#!/usr/bin/env python3
"""Run the official RoMa v2 fast/512 matcher on a small Open3DHK subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))
from common import load_rgb, save_rgb  # noqa: E402
from reference_consistency import _lab_adjust  # noqa: E402
from roma_reference import _load_model  # noqa: E402


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _mask_for_image(image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    candidate = image_path.parents[0].parent / "mask_for_sam" / image_path.name
    raw = np.asarray(Image.open(candidate).convert("L"), dtype=np.uint8)
    return np.asarray(Image.fromarray(raw).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def _zoom(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return image
    y0, y1 = max(0, int(ys.min()) - 24), min(image.shape[0], int(ys.max()) + 25)
    x0, x1 = max(0, int(xs.min()) - 24), min(image.shape[1], int(xs.max()) + 25)
    return image[y0:y1, x0:x1]


def _warp_reference(model, reference: np.ndarray, warp, overlap, output_shape: tuple[int, int]):
    import torch
    import torch.nn.functional as F

    h, w = output_shape
    ref = torch.from_numpy(np.array(reference, copy=True)).permute(2, 0, 1).float().div(255.0)[None]
    ref = F.interpolate(ref, size=warp.shape[:2], mode="bilinear", align_corners=False, antialias=True)
    sampled = F.grid_sample(ref.cuda(), warp[None].cuda(), mode="bilinear", padding_mode="zeros", align_corners=False)
    sampled = F.interpolate(sampled, size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0]
    conf = F.interpolate(overlap[None, None].float().cuda(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return (sampled.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy(), conf.cpu().numpy())


def run(args: argparse.Namespace) -> None:
    import torch

    rows = []
    with _resolve(args.metrics_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["mask_source"] == "manual"
                and row["reference_path"]
                and _resolve(row["input_path"]).is_file()
                and _resolve(row["reference_path"]).is_file()
                and row["showcase_index"] not in {r["showcase_index"] for r in rows}
            ):
                rows.append(row)
    rows = rows[: args.limit]
    out = _resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = _load_model()
    records = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            target_path = _resolve(row["input_path"])
            reference_path = _resolve(row["reference_path"])
            target = load_rgb(target_path)
            reference = load_rgb(reference_path)
            mask = _mask_for_image(target_path, target.shape[:2])
            preds = model.match(str(target_path), str(reference_path))
            warp = preds["warp_AB"][0]
            overlap = preds["overlap_AB"][0, ..., 0]
            warped, confidence = _warp_reference(model, reference, warp, overlap, target.shape[:2])
            valid = confidence >= float(args.confidence)
            photo_region = valid & ~cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1).astype(bool)
            adjusted = _lab_adjust(warped, target, photo_region)
            allowed = mask & valid
            composite = target.copy()
            composite[allowed] = adjusted[allowed]
            difference = np.abs(composite.astype(np.int16) - target.astype(np.int16)).clip(0, 255).astype(np.uint8)
            stem = out / f"{index:02d}"
            stem.mkdir(parents=True, exist_ok=True)
            save_rgb(stem / "input.png", target)
            save_rgb(stem / "reference.png", reference)
            save_rgb(stem / "roma_warp.png", warped)
            save_rgb(stem / "confidence.png", np.repeat((confidence.clip(0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2))
            save_rgb(stem / "composite.png", composite)
            save_rgb(stem / "zoom.png", _zoom(composite, mask))
            save_rgb(stem / "difference.png", difference)
            records.append({
                "showcase_index": row["showcase_index"],
                "unique_image_id": row["unique_image_id"],
                "target_path": str(target_path),
                "reference_path": str(reference_path),
                "overlap_mean": float(overlap.mean().item()),
                "overlap_median": float(overlap.median().item()),
                "mask_support_ratio": float((mask & valid).sum() / max(1, mask.sum())),
                "mask_confidence_mean": float(confidence[mask].mean()) if mask.any() else 0.0,
                "mask_outside_max_abs": int(difference[~mask].max()) if np.any(~mask) else 0,
                "output_dir": str(stem),
            })
    (out / "roma_showcase_metrics.csv").write_text("", encoding="utf-8")
    if records:
        with (out / "roma_showcase_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "status": "completed",
        "setting": "official RoMa v2 fast / 512 / batch=1",
        "images": len(records),
        "mean_mask_support_ratio": float(np.mean([r["mask_support_ratio"] for r in records])) if records else 0.0,
        "outside_max_abs": int(max((r["mask_outside_max_abs"] for r in records), default=0)),
        "official_repo": "https://github.com/Parskatt/RoMaV2",
    }
    (out / "roma_showcase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", default="research/outputs/restoration_v2/reference_consistency/reference_consistency_metrics.csv")
    parser.add_argument("--output", default="research/outputs/restoration_v2/reference_consistency/roma_fast")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.5)
    run(parser.parse_args())
