#!/usr/bin/env python3
"""Load SAM + LoRA on CPU and produce P0 masks/overlays."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from common import json_dump, load_rgb, overlay_mask, read_csv, relpath, repo_root, save_mask, save_rgb, stable_hash, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--mode", choices=("p0", "showcase", "all"), default="all")
    parser.add_argument("--p0-limit", type=int, default=12)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    return parser.parse_args()


def import_runtime(root: Path, runtime_dir: Path | None) -> None:
    runtime = runtime_dir or (root / ".research_runtime")
    sys.path.insert(0, str(runtime.resolve()))
    sys.path.insert(0, str((root / "code").resolve()))


def load_lora_cpu(net, checkpoint: Path, torch) -> None:
    """Same loading logic as the supplied wrapper, with explicit CPU mapping."""
    from torch.nn.parameter import Parameter

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for index, module in enumerate(net.w_As):
        module.weight = Parameter(state_dict[f"w_a_{index:03d}"].detach().cpu())
    for index, module in enumerate(net.w_Bs):
        module.weight = Parameter(state_dict[f"w_b_{index:03d}"].detach().cpu())
    sam_dict = net.sam.state_dict()
    prompt_keys = [key for key in sam_dict if "prompt_encoder" in key]
    mask_keys = [key for key in sam_dict if "mask_decoder" in key]
    for key in prompt_keys + mask_keys:
        if key not in state_dict:
            raise KeyError(f"LoRA checkpoint is missing {key}")
        sam_dict[key] = state_dict[key].detach().cpu()
    net.sam.load_state_dict(sam_dict)


def build_model(root: Path, runtime_dir: Path | None):
    import_runtime(root, runtime_dir)
    import torch
    from segment_anything import sam_model_registry
    from sam_lora_image_encoder import LoRA_Sam

    torch.set_num_threads(max(1, min(8, (os_cpu_count() or 4))))
    base = root / "sam_vit_b_01ec64.pth"
    lora = root / "checkpoint_best.pth"
    base_state = torch.load(base, map_location="cpu", weights_only=False)
    lora_state = torch.load(lora, map_location="cpu", weights_only=False)
    sam, embedding_size = sam_model_registry["vit_b"](
        image_size=512,
        num_classes=1,
        checkpoint=str(base),
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    net = LoRA_Sam(sam, 4)
    load_lora_cpu(net, lora, torch)
    net.eval()
    return net, torch, {
        "base_checkpoint_bytes": base.stat().st_size,
        "lora_checkpoint_bytes": lora.stat().st_size,
        "base_state_keys": len(base_state),
        "lora_state_keys": len(lora_state),
        "embedding_size": embedding_size,
        "device": "cpu",
    }


def os_cpu_count() -> int | None:
    import os

    return os.cpu_count()


def predict(net, torch, rgb: np.ndarray) -> tuple[np.ndarray, float]:
    original_h, original_w = rgb.shape[:2]
    resized = np.asarray(Image.fromarray(rgb, mode="RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        output = net(tensor, False, 512)
    logits = output["masks"][0]
    prediction = logits.argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
    mask = np.asarray(Image.fromarray(prediction * 255, mode="L").resize((original_w, original_h), Image.Resampling.NEAREST)) > 0
    iou_score = float(output["iou_predictions"].detach().cpu().numpy().reshape(-1)[-1])
    return mask, iou_score


def select_p0(root: Path, limit: int) -> list[tuple[Path, Path]]:
    ann = root / "research" / "assets" / "distortion_segmentation_annotation_dataset" / "for_segmentation"
    pairs = []
    for image in sorted((ann / "images").glob("*.png")):
        mask = ann / "mask_for_sam" / image.name
        if mask.exists():
            with Image.open(mask) as opened:
                area = float(np.asarray(opened.convert("L"), dtype=np.uint8).mean() / 255.0)
            pairs.append((area, image, mask))
    pairs.sort(key=lambda row: (row[0], stable_hash(row[1].name)))
    if not pairs:
        return []
    selected = []
    for quantile in np.linspace(0.0, 1.0, min(4, limit), endpoint=True):
        selected.append(pairs[min(len(pairs) - 1, int(round(quantile * (len(pairs) - 1))))])
    selected.extend(pairs)
    unique = []
    seen = set()
    for area, image, mask in selected:
        if image.name in seen:
            continue
        unique.append((image, mask))
        seen.add(image.name)
        if len(unique) >= limit:
            break
    return unique


def run_p0(root: Path, net, torch, limit: int) -> dict:
    output_root = root / "research" / "outputs" / "restoration" / "sam_p0"
    rows = []
    for index, (image_path, annotation_path) in enumerate(select_p0(root, limit)):
        rgb = load_rgb(image_path)
        annotation = np.asarray(Image.open(annotation_path).convert("L")) > 0
        predicted, iou_score = predict(net, torch, rgb)
        sample = output_root / f"{index:02d}_{image_path.stem}"
        save_rgb(sample / "input.png", rgb)
        save_mask(sample / "annotation_mask.png", annotation)
        save_mask(sample / "sam_mask.png", predicted)
        save_rgb(sample / "overlay.png", overlay_mask(rgb, predicted))
        save_rgb(sample / "annotation_overlay.png", overlay_mask(rgb, annotation, color=(50, 180, 255)))
        intersection = int(np.logical_and(predicted, annotation).sum())
        union = int(np.logical_or(predicted, annotation).sum())
        dice_den = int(predicted.sum() + annotation.sum())
        rows.append(
            {
                "sample_id": image_path.stem,
                "image_path": relpath(image_path, root),
                "annotation_path": relpath(annotation_path, root),
                "sam_mask_path": relpath(sample / "sam_mask.png", root),
                "annotation_ratio": float(annotation.mean()),
                "sam_ratio": float(predicted.mean()),
                "iou": float(intersection / union) if union else 1.0,
                "dice": float(2 * intersection / dice_den) if dice_den else 1.0,
                "model_iou_score": iou_score,
            }
        )
    write_csv(root / "research" / "data" / "sam_p0_manifest.csv", rows, list(rows[0].keys()) if rows else ["sample_id"])
    return {"count": len(rows), "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else None, "mean_dice": float(np.mean([r["dice"] for r in rows])) if rows else None}


def run_showcase(root: Path, net, torch) -> dict:
    manifest = root / "research" / "data" / "open3dhk_showcase.csv"
    rows = read_csv(manifest)
    output_root = root / "research" / "outputs" / "restoration" / "showcase_sam"
    for index, row in enumerate(rows):
        image_rel = row.get("input_path") or row["image_path"]
        image_path = root / image_rel
        rgb = load_rgb(image_path)
        predicted, iou_score = predict(net, torch, rgb)
        sample = output_root / f"{int(row.get('showcase_index', index)):02d}_{Path(image_rel).stem}"
        save_mask(sample / "sam_mask.png", predicted)
        save_rgb(sample / "overlay.png", overlay_mask(rgb, predicted))
        annotation_rel = row.get("annotation_mask_path", "")
        if annotation_rel and (root / annotation_rel).exists():
            annotation = np.asarray(Image.open(root / annotation_rel).convert("L"), dtype=np.uint8) > 0
            intersection = int(np.logical_and(annotation, predicted).sum())
            union = int(np.logical_or(annotation, predicted).sum())
            dice_den = int(annotation.sum() + predicted.sum())
            row["manual_sam_iou"] = f"{(intersection / union) if union else 1.0:.8f}"
            row["manual_sam_dice"] = f"{(2 * intersection / dice_den) if dice_den else 1.0:.8f}"
            row["annotation_mask_ratio"] = f"{float(annotation.mean()):.8f}"
        row["sam_mask_path"] = relpath(sample / "sam_mask.png", root)
        row["sam_mask_ratio"] = f"{float(predicted.mean()):.8f}"
        row["sam_model_iou_score"] = f"{iou_score:.8f}"
        row["mask_source"] = "SAM_LoRA_cpu"
    write_csv(manifest, rows, list(rows[0].keys()) if rows else [])
    return {"count": len(rows), "mean_sam_ratio": float(np.mean([float(r["sam_mask_ratio"]) for r in rows])) if rows else None}


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    net, torch, load_summary = build_model(root, args.runtime_dir)
    summary = {"checkpoints_loaded": True, **load_summary}
    if args.mode in ("p0", "all"):
        summary["p0"] = run_p0(root, net, torch, args.p0_limit)
    if args.mode in ("showcase", "all"):
        summary["showcase"] = run_showcase(root, net, torch)
    json_dump(root / "research" / "outputs" / "restoration" / "sam_status.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
