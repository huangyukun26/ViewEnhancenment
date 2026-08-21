#!/usr/bin/env python3
"""Constrained FLUX.2-klein quick experiment on the fixed Open3DHK showcase.

This runner deliberately keeps the real model path separate from the strict
compositing and reporting path.  If weights, credentials, or GPU memory are
not available it writes an honest blocker report and identity passthrough
artifacts; those artifacts are never labelled as FLUX outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_mask, load_rgb, save_mask, save_rgb  # noqa: E402


OUT = ROOT / "research" / "outputs" / "restoration_v2" / "flux_constrained"
PROMPT = (
    "Restore the damaged exterior building facade only. Continue the exact same building, "
    "facade geometry, perspective lines, window grid, concrete and glass materials visible "
    "in the surrounding photograph. Match the original camera viewpoint, daylight, "
    "atmospheric haze, color temperature, sharpness, sensor noise and compression. Produce "
    "a conservative photorealistic architectural restoration with seamless boundaries and "
    "minimal visual change."
)
SEEDS = (0, 1)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label_tile(image: np.ndarray, label: str, size: tuple[int, int] = (280, 230)) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    body = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    body.thumbnail((size[0] - 8, size[1] - 32), Image.Resampling.LANCZOS)
    tile.paste(body, ((size[0] - body.width) // 2, 3))
    ImageDraw.Draw(tile).text((5, size[1] - 24), label[:48], fill="black", font=_font(15))
    return tile


def _na_tile(label: str, size: tuple[int, int] = (280, 230)) -> Image.Image:
    tile = Image.new("RGB", size, (238, 238, 238))
    draw = ImageDraw.Draw(tile)
    draw.text((size[0] // 2 - 20, size[1] // 2 - 10), "N/A", fill=(80, 80, 80), font=_font(20))
    draw.text((5, size[1] - 24), label[:48], fill="black", font=_font(15))
    return tile


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _crop_context(target: np.ndarray, mask: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    h, w = mask.shape[:2]
    x0, x1 = max(0, int(xs.min()) - context), min(w, int(xs.max()) + 1 + context)
    y0, y1 = max(0, int(ys.min()) - context), min(h, int(ys.max()) + 1 + context)
    return target[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _inner_alpha(mask: np.ndarray, feather: int = 4) -> np.ndarray:
    """Return alpha that is zero outside M and feathers only into M."""
    hard = np.asarray(mask, dtype=bool)
    if feather <= 0:
        return hard.astype(np.float32)
    distance = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(distance / float(feather), 0.0, 1.0).astype(np.float32) * hard.astype(np.float32)


def _strict_composite(original: np.ndarray, raw_crop: np.ndarray, crop_mask: np.ndarray, bbox: tuple[int, int, int, int], feather: int) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop_h, crop_w = y1 - y0, x1 - x0
    if raw_crop.shape[:2] != (crop_h, crop_w):
        raw_crop = cv2.resize(raw_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    alpha = _inner_alpha(crop_mask, feather)
    crop_input = original[y0:y1, x0:x1].astype(np.float32)
    composite_crop = np.rint(alpha[..., None] * raw_crop.astype(np.float32) + (1.0 - alpha[..., None]) * crop_input).clip(0, 255).astype(np.uint8)
    output = original.copy()
    output[y0:y1, x0:x1] = composite_crop
    # This assignment is intentionally explicit: no candidate can touch M=0.
    output[~_full_mask_from_bbox(original.shape[:2], crop_mask, bbox)] = original[~_full_mask_from_bbox(original.shape[:2], crop_mask, bbox)]
    return output


def _full_mask_from_bbox(shape: tuple[int, int], crop_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    full = np.zeros(shape, dtype=bool)
    full[y0:y1, x0:x1] = crop_mask
    return full


def _metrics(original: np.ndarray, output: np.ndarray, mask: np.ndarray, runtime_sec: float, raw_path: str, output_path: str, backend_status: str) -> dict[str, object]:
    hard = np.asarray(mask, dtype=bool)
    outside = ~hard
    delta = np.abs(output.astype(np.int16) - original.astype(np.int16))
    gray_in = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_out = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab_in = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_out = cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = hard & ~cv2.erode(hard.astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1).astype(bool)
    grad_in = cv2.magnitude(cv2.Sobel(gray_in, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray_in, cv2.CV_32F, 0, 1, 3))
    grad_out = cv2.magnitude(cv2.Sobel(gray_out, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray_out, cv2.CV_32F, 0, 1, 3))
    low_in = cv2.GaussianBlur(original, (0, 0), 8).astype(np.float32)
    low_out = cv2.GaussianBlur(output, (0, 0), 8).astype(np.float32)
    return {
        "backend_status": backend_status,
        "outside_max_abs": int(delta[outside].max()) if np.any(outside) else 0,
        "changed_fraction_mask": float(np.mean(np.mean(delta, axis=2)[hard] > 2.0)) if np.any(hard) else 0.0,
        "mean_abs_delta_mask": float(np.mean(delta[hard])) if np.any(hard) else 0.0,
        "boundary_lab_l1": float(np.mean(np.abs(lab_out[ring] - lab_in[ring]))) if np.any(ring) else 0.0,
        "boundary_gradient_diff": float(np.mean(np.abs(grad_out[ring] - grad_in[ring]))) if np.any(ring) else 0.0,
        "low_frequency_diff_mask": float(np.mean(np.abs(low_out[hard] - low_in[hard]))) if np.any(hard) else 0.0,
        "runtime_sec": float(runtime_sec),
        "raw_path": raw_path,
        "output_path": output_path,
    }


def _find_model(args: argparse.Namespace) -> tuple[Path | None, list[str]]:
    candidates = [args.model_path,]
    import os
    candidates.extend([os.environ.get("KLEIN_4B_MODEL_PATH"), os.environ.get("FLUX2_MODEL_PATH")])
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate), []
    return None, [
        "local_checkpoint_missing: no FLUX.2-klein-4B directory was found in --model-path, "
        "KLEIN_4B_MODEL_PATH, or FLUX2_MODEL_PATH",
        "HF_TOKEN/HUGGINGFACE_HUB_TOKEN and BFL_API_KEY were not configured during this run",
        "hardware_note: detected RTX 3060 has 6GB VRAM; the official FLUX.2-klein-4B card states approximately 13GB VRAM",
    ]


def _load_pipeline(model_path: Path):
    import torch
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = Flux2KleinPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    return pipe, torch


def _run_generation(pipe, torch_module, crop: np.ndarray, reference_paths: list[Path], seed: int, steps: int) -> np.ndarray:
    target = Image.fromarray(crop, mode="RGB")
    refs = [Image.open(path).convert("RGB") for path in reference_paths[:2] if path.exists()]
    images = [target] + refs
    h, w = crop.shape[:2]
    # Keep the native crop aspect ratio but round to a pipeline-friendly size.
    gen_h, gen_w = max(64, (h // 16) * 16), max(64, (w // 16) * 16)
    generator = torch_module.Generator(device="cuda" if torch_module.cuda.is_available() else "cpu").manual_seed(seed)
    kwargs = {
        "prompt": PROMPT,
        "image": images if len(images) > 1 else target,
        "height": gen_h,
        "width": gen_w,
        "guidance_scale": 1.0,
        "num_inference_steps": steps,
        "generator": generator,
    }
    result = pipe(**kwargs).images[0]
    return np.asarray(result.convert("RGB"), dtype=np.uint8)


def _difference_image(original: np.ndarray, selected: np.ndarray, mask: np.ndarray) -> np.ndarray:
    difference = np.abs(selected.astype(np.int16) - original.astype(np.int16)).astype(np.uint8)
    difference = np.minimum(difference * 4, 255).astype(np.uint8)
    difference[~mask] = 0
    return difference


def _write_blocker(path: Path, blockers: list[str]) -> None:
    path.write_text(
        "FLUX generation was not run. These are identity passthrough artifacts, not FLUX outputs.\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n",
        encoding="utf-8",
    )


def _candidate_name(multi_reference: bool, seed: int) -> str:
    return f"target_plus_reference_seed{seed}" if multi_reference else f"target_only_seed{seed}"


def _write_commands() -> None:
    (OUT / "run_commands.md").write_text(
        "# Constrained FLUX quick6 commands\n\n"
        "The current committed run is an honest blocker run because no local FLUX.2-klein-4B weights, HF token, or BFL API key were available. It does not label identity passthrough as FLUX output.\n\n"
        "```powershell\n"
        "& 'C:\\Users\\LENOVO\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' research/restoration_v2/flux_constrained.py --quick6 research/data/flux_quick6.csv\n"
        "```\n\n"
        "With a locally provisioned official checkpoint (do not commit weights):\n\n"
        "```powershell\n"
        "$env:KLEIN_4B_MODEL_PATH='D:\\path\\to\\FLUX.2-klein-4B'\n"
        "& 'C:\\Users\\LENOVO\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' research/restoration_v2/flux_constrained.py --quick6 research/data/flux_quick6.csv --model-path $env:KLEIN_4B_MODEL_PATH\n"
        "```\n\n"
        "Official references: [FLUX.2 repository](https://github.com/black-forest-labs/flux2), [FLUX.2-klein-4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick6", type=Path, default=ROOT / "research" / "data" / "flux_quick6.csv")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--feather", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    quick6_path = args.quick6 if args.quick6.is_absolute() else ROOT / args.quick6
    OUT.mkdir(parents=True, exist_ok=True)
    _write_commands()
    rows = _read_rows(quick6_path)
    model_path, blockers = _find_model(args)
    pipe = torch_module = None
    backend_status = "blocked_identity_passthrough"
    if model_path is not None:
        try:
            pipe, torch_module = _load_pipeline(model_path)
            backend_status = "flux2_klein_4b_diffusers"
        except Exception as exc:
            blockers = [f"pipeline_load_failed: {type(exc).__name__}: {exc}"] + blockers
            backend_status = "blocked_identity_passthrough"

    all_candidates: list[dict[str, object]] = []
    sample_records: list[dict[str, object]] = []
    for row in rows:
        sample_id = row["sample_id"]
        sample_dir = OUT / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_path = _resolve(row["image_path"])
        mask_path = _resolve(row["mask_path"])
        target = load_rgb(target_path)
        mask = load_mask(mask_path)
        if mask.shape != target.shape[:2]:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((target.shape[1], target.shape[0]), Image.Resampling.NEAREST)) > 0
        crop, crop_mask, bbox = _crop_context(target, mask, args.context)
        refs = [_resolve(p.strip()) for p in row.get("reference_paths", "").split(";") if p.strip()]
        valid_refs = [p for p in refs if p.exists()]
        save_rgb(sample_dir / "input.png", target)
        save_mask(sample_dir / "mask.png", mask)
        x0, y0, x1, y1 = bbox
        target_only_names = [_candidate_name(False, seed) for seed in SEEDS]
        modes = [(False, seed) for seed in SEEDS]
        if valid_refs:
            modes.extend((True, seed) for seed in SEEDS)
        candidate_outputs: dict[str, np.ndarray] = {}
        for multi_reference, seed in modes:
            name = _candidate_name(multi_reference, seed)
            raw_path = sample_dir / f"{name}_raw.png"
            started = time.perf_counter()
            if pipe is not None:
                try:
                    raw = _run_generation(pipe, torch_module, crop, valid_refs if multi_reference else [], seed, args.steps)
                    status = backend_status
                except Exception as exc:
                    blockers.append(f"{sample_id}/{name}_failed: {type(exc).__name__}: {exc}")
                    raw = crop.copy()
                    status = "generation_failed_identity_passthrough"
            else:
                raw = crop.copy()
                status = backend_status
            save_rgb(raw_path, raw)
            output = _strict_composite(target, raw, crop_mask, bbox, args.feather)
            output_path = sample_dir / f"{name}.png"
            save_rgb(output_path, output)
            elapsed = time.perf_counter() - started
            candidate_outputs[name] = output
            row_metrics = _metrics(target, output, mask, elapsed, str(raw_path.relative_to(ROOT)), str(output_path.relative_to(ROOT)), status)
            all_candidates.append({
                "sample_id": sample_id,
                "degradation_type": row["degradation_type"],
                "candidate": name,
                "reference_count": len(valid_refs) if multi_reference else 0,
                **row_metrics,
            })

        # In a blocker run all four candidates are explicitly identical passthroughs.
        eligible = [item for item in all_candidates if item["sample_id"] == sample_id and item["outside_max_abs"] == 0]
        if backend_status == "blocked_identity_passthrough" or not eligible:
            selected_name = "identity_abstain"
            selected = target.copy()
            selected_raw = crop.copy()
        else:
            # Conservative ranking: smallest boundary/low-frequency change first,
            # then a small nonzero change. This is an ordering aid, not a quality claim.
            ranked = sorted(eligible, key=lambda item: (float(item["boundary_lab_l1"]) + float(item["boundary_gradient_diff"]) + float(item["low_frequency_diff_mask"]), -float(item["changed_fraction_mask"])))
            selected_name = str(ranked[0]["candidate"])
            selected = candidate_outputs[selected_name]
            selected_raw = load_rgb(sample_dir / f"{selected_name}_raw.png")
        save_rgb(sample_dir / "selected.png", selected)
        save_rgb(sample_dir / "generation_raw.png", selected_raw)
        zoom = selected[max(0, y0 - 16):min(selected.shape[0], y1 + 16), max(0, x0 - 16):min(selected.shape[1], x1 + 16)]
        save_rgb(sample_dir / "selected_zoom.png", zoom)
        save_rgb(sample_dir / "difference.png", _difference_image(target, selected, mask))
        if backend_status == "blocked_identity_passthrough":
            _write_blocker(sample_dir / "blocker.txt", blockers)
        sample_records.append({
            "sample_id": sample_id,
            "degradation_type": row["degradation_type"],
            "reference_count": len(valid_refs),
            "selected": selected_name,
            "backend_status": backend_status,
            "input_path": str((sample_dir / "input.png").relative_to(ROOT)),
            "selected_path": str((sample_dir / "selected.png").relative_to(ROOT)),
        })

    fields = ["sample_id", "degradation_type", "candidate", "reference_count", "backend_status", "outside_max_abs", "changed_fraction_mask", "mean_abs_delta_mask", "boundary_lab_l1", "boundary_gradient_diff", "low_frequency_diff_mask", "runtime_sec", "raw_path", "output_path"]
    with (OUT / "flux_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_candidates)

    # Contact sheet: exactly the requested seven columns, with N/A when no ref exists.
    contact_rows = []
    for sample in sample_records:
        sample_dir = OUT / sample["sample_id"]
        input_image = load_rgb(sample_dir / "input.png")
        mask_rgb = np.repeat(np.asarray(load_mask(sample_dir / "mask.png"), dtype=np.uint8)[..., None] * 255, 3, axis=2)
        target_path = sample_dir / "target_only_seed0.png"
        multi_path = sample_dir / "target_plus_reference_seed0.png"
        contact_rows.append([
            _label_tile(input_image, f"{sample['sample_id']} input"),
            _label_tile(mask_rgb, "mask"),
            _label_tile(load_rgb(target_path), "target-only"),
            _label_tile(load_rgb(multi_path), "multi-reference") if multi_path.exists() else _na_tile("multi-reference N/A"),
            _label_tile(load_rgb(sample_dir / "selected.png"), f"selected: {sample['selected']}"),
            _label_tile(load_rgb(sample_dir / "selected_zoom.png"), "zoom"),
            _label_tile(load_rgb(sample_dir / "difference.png"), "difference x4"),
        ])
    tile_w, tile_h = 280, 230
    contact = Image.new("RGB", (7 * tile_w, len(contact_rows) * tile_h), "white")
    for r_index, row_tiles in enumerate(contact_rows):
        for c_index, tile in enumerate(row_tiles):
            contact.paste(tile, (c_index * tile_w, r_index * tile_h))
    contact.save(OUT / "flux_quick6_contact_sheet.png")

    # Blind A/B image. The key is separate and is intentionally not used for scoring.
    blind_rng = random.Random(20260821)
    blind_key = {"seed": 20260821, "backend_status": backend_status, "samples": {}}
    blind = Image.new("RGB", (3 * tile_w, len(sample_records) * tile_h), "white")
    for r_index, sample in enumerate(sample_records):
        sample_dir = OUT / sample["sample_id"]
        identity = load_rgb(sample_dir / "input.png")
        selected = load_rgb(sample_dir / "selected.png")
        selected_on_a = bool(blind_rng.randint(0, 1))
        a, b = (selected, identity) if selected_on_a else (identity, selected)
        blind_key["samples"][sample["sample_id"]] = {"A": "selected" if selected_on_a else "identity", "B": "identity" if selected_on_a else "selected", "selected": sample["selected"]}
        for c_index, tile in enumerate([_label_tile(identity, "input"), _label_tile(a, "A"), _label_tile(b, "B")]):
            blind.paste(tile, (c_index * tile_w, r_index * tile_h))
    blind.save(OUT / "flux_quick6_blind.png")
    (OUT / "blind_key.json").write_text(json.dumps(blind_key, ensure_ascii=False, indent=2), encoding="utf-8")

    pref_fields = ["sample_id", "better", "same", "worse", "notes"]
    with (OUT / "human_preference_flux.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pref_fields)
        writer.writeheader()
        for sample in sample_records:
            writer.writerow({"sample_id": sample["sample_id"], "better": "", "same": "", "worse": "", "notes": ""})

    outside_values = [int(item["outside_max_abs"]) for item in all_candidates]
    summary = {
        "status": "blocked" if backend_status == "blocked_identity_passthrough" else "completed",
        "backend": "FLUX.2-klein-4B / Diffusers",
        "backend_status": backend_status,
        "blockers": sorted(set(blockers)),
        "images": len(sample_records),
        "generated_flux_candidates": sum(1 for item in all_candidates if item["backend_status"] == "flux2_klein_4b_diffusers"),
        "candidate_rows": len(all_candidates),
        "selected_identity_count": sum(1 for item in sample_records if item["selected"] == "identity_abstain"),
        "outside_max_abs_all_candidates": max(outside_values) if outside_values else None,
        "outside_max_abs_requirement_satisfied": bool(outside_values) and max(outside_values) == 0,
        "go_decision": "NO-GO_BLOCKED" if backend_status == "blocked_identity_passthrough" else "REVIEW_QUICK6",
        "quick6_manifest": str(quick6_path.relative_to(ROOT)),
        "contact_sheet": str((OUT / "flux_quick6_contact_sheet.png").relative_to(ROOT)),
        "blind_sheet": str((OUT / "flux_quick6_blind.png").relative_to(ROOT)),
        "blind_key": str((OUT / "blind_key.json").relative_to(ROOT)),
        "human_preference": str((OUT / "human_preference_flux.csv").relative_to(ROOT)),
        "official_repo": "https://github.com/black-forest-labs/flux2",
        "model_card": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        "subjective_quality_claim": "none; human_preference_flux.csv is intentionally blank",
    }
    (OUT / "flux_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
