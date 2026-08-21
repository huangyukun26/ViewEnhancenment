#!/usr/bin/env python3
"""Fidelity-constrained post-processing and FLUX.1 Fill quick6 experiment.

The existing FLUX.2 outputs are treated as a fixed, high-quality/low-fidelity
baseline.  This script keeps the input responsible for low-frequency appearance
and uses Fill candidates only where seed agreement and structural evidence allow.
All final composites are explicitly copied from the input outside the allowed
mask; no automatic metric is presented as a perceptual quality claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


SOURCE_DEFAULT = ROOT / "research" / "outputs" / "restoration_v2" / "flux_constrained_cloud_v100"
OUT_DEFAULT = ROOT / "research" / "outputs" / "restoration_v2" / "flux_fidelity_cloud_v100"
PROMPT = (
    "Restore the damaged exterior building facade only. Continue the exact same building, "
    "facade geometry, perspective lines, window grid, concrete and glass materials visible "
    "in the surrounding photograph. Match the original camera viewpoint, daylight, "
    "atmospheric haze, color temperature, sharpness, sensor noise and compression. Produce "
    "a conservative photorealistic architectural restoration with seamless boundaries and "
    "minimal visual change."
)
FILL_SEEDS = (0, 1, 2, 3)


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _font(size: int = 15):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _crop_context(target: np.ndarray, mask: np.ndarray, context: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    h, w = mask.shape[:2]
    x0, x1 = max(0, int(xs.min()) - context), min(w, int(xs.max()) + 1 + context)
    y0, y1 = max(0, int(ys.min()) - context), min(h, int(ys.max()) + 1 + context)
    return target[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _full_from_crop(shape: tuple[int, int], crop: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    full = np.zeros(shape, dtype=crop.dtype)
    full[y0:y1, x0:x1] = crop
    return full


def _paste_crop(target: np.ndarray, raw_crop: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = target.copy()
    raw = raw_crop
    if raw.shape[:2] != (y1 - y0, x1 - x0):
        raw = cv2.resize(raw, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)
    out[y0:y1, x0:x1] = raw
    return out


def _inner_alpha(mask: np.ndarray, feather: int = 4) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    if feather <= 0:
        return hard.astype(np.float32)
    distance = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(distance / float(feather), 0.0, 1.0).astype(np.float32) * hard.astype(np.float32)


def _composite(original: np.ndarray, generated: np.ndarray, mask: np.ndarray, feather: int = 4, confidence: np.ndarray | None = None) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    alpha = _inner_alpha(hard, feather)
    if confidence is not None:
        alpha = alpha * np.asarray(confidence, dtype=np.float32)
    out = np.rint(alpha[..., None] * generated.astype(np.float32) + (1.0 - alpha[..., None]) * original.astype(np.float32)).clip(0, 255).astype(np.uint8)
    out[~hard] = original[~hard]
    return out


def _lab(image: np.ndarray) -> np.ndarray:
    # OpenCV's 8-bit Lab representation is stable here: L,a,b are all in
    # [0,255] (with the chroma midpoint encoded around 128). Using float RGB
    # without normalizing to [0,1] silently clips and creates black outputs.
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def _lab_rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _lab_anchor(target: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Anchor generated Lab median/MAD to target, with bounded scale."""
    hard = np.asarray(mask, dtype=bool)
    target_lab, generated_lab = _lab(target), _lab(generated)
    if not np.any(hard):
        return target.copy()
    target_values, generated_values = target_lab[hard], generated_lab[hard]
    target_median = np.median(target_values, axis=0)
    generated_median = np.median(generated_values, axis=0)
    target_mad = np.median(np.abs(target_values - target_median), axis=0)
    generated_mad = np.median(np.abs(generated_values - generated_median), axis=0)
    l_scale = float(np.clip((target_mad[0] + 1e-3) / (generated_mad[0] + 1e-3), 0.85, 1.15))
    ab_scale = np.clip((target_mad[1:] + 1e-3) / (generated_mad[1:] + 1e-3), 0.7, 1.3)
    anchored = generated_lab.copy()
    anchored[..., 0] = target_median[0] + (generated_lab[..., 0] - generated_median[0]) * l_scale
    anchored[..., 1:] = target_median[1:] + (generated_lab[..., 1:] - generated_median[1:]) * ab_scale
    return _composite(target, _lab_rgb(anchored), hard, feather=4)


def _frequency_fuse(target: np.ndarray, generated: np.ndarray, mask: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Keep target low frequency and inject bounded generated high frequency."""
    target_lab, generated_lab = _lab(target), _lab(generated)
    target_low = cv2.GaussianBlur(target_lab, (0, 0), sigmaX=sigma)
    generated_low = cv2.GaussianBlur(generated_lab, (0, 0), sigmaX=sigma)
    fused = target_low.copy()
    fused[..., 0] = target_low[..., 0] + 0.75 * (generated_lab[..., 0] - generated_low[..., 0])
    fused[..., 1:] = target_low[..., 1:] + 0.25 * (generated_lab[..., 1:] - generated_low[..., 1:])
    return _composite(target, _lab_rgb(fused), np.asarray(mask, dtype=bool), feather=4)


def _difference_image(original: np.ndarray, selected: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta = np.abs(selected.astype(np.int16) - original.astype(np.int16)).astype(np.float32)
    difference = np.clip(delta * 4.0, 0.0, 255.0).astype(np.uint8)
    difference[~np.asarray(mask, dtype=bool)] = 0
    return difference


def _edge_f1(target: np.ndarray, output: np.ndarray, mask: np.ndarray) -> float:
    target_edge = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    output_edge = cv2.Canny(cv2.cvtColor(output, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    hard = np.asarray(mask, dtype=bool)
    target_edge &= hard
    output_edge &= hard
    if not np.any(target_edge) and not np.any(output_edge):
        return 1.0
    target_near = cv2.dilate(target_edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    output_near = cv2.dilate(output_edge.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    precision = float(np.sum(output_edge & target_near)) / max(1, int(np.sum(output_edge)))
    recall = float(np.sum(target_edge & output_near)) / max(1, int(np.sum(target_edge)))
    return 2.0 * precision * recall / max(1e-6, precision + recall)


def _seed_stats(candidates: list[np.ndarray], mask: np.ndarray) -> dict[str, object]:
    if len(candidates) < 2:
        return {"pairwise_seed_mad": 0.0, "seed_edge_iou": 1.0, "medoid_index": 0, "mean_distances": [0.0]}
    hard = np.asarray(mask, dtype=bool)
    distances = np.zeros(len(candidates), dtype=np.float64)
    mads, edge_ious = [], []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            rgb_distance = float(np.mean(np.abs(candidates[i][hard].astype(np.float32) - candidates[j][hard].astype(np.float32))) / 255.0)
            lab_distance = float(np.mean(np.abs(_lab(candidates[i])[hard] - _lab(candidates[j])[hard])) / 100.0)
            low_i = cv2.GaussianBlur(candidates[i].astype(np.float32), (0, 0), 12.0)
            low_j = cv2.GaussianBlur(candidates[j].astype(np.float32), (0, 0), 12.0)
            low_distance = float(np.mean(np.abs(low_i[hard] - low_j[hard])) / 255.0)
            distance = (rgb_distance + lab_distance + low_distance) / 3.0
            distances[i] += distance
            distances[j] += distance
            mads.append(float(np.mean(np.abs(candidates[i][hard].astype(np.float32) - candidates[j][hard].astype(np.float32)))))
            edge_i = cv2.Canny(cv2.cvtColor(candidates[i], cv2.COLOR_RGB2GRAY), 80, 160) > 0
            edge_j = cv2.Canny(cv2.cvtColor(candidates[j], cv2.COLOR_RGB2GRAY), 80, 160) > 0
            edge_i &= hard
            edge_j &= hard
            union = np.sum(edge_i | edge_j)
            edge_ious.append(float(np.sum(edge_i & edge_j) / union) if union else 1.0)
    distances /= max(1, len(candidates) - 1)
    return {
        "pairwise_seed_mad": float(np.mean(mads)),
        "seed_edge_iou": float(np.mean(edge_ious)),
        "medoid_index": int(np.argmin(distances)),
        "mean_distances": distances.tolist(),
    }


def _uncertainty(candidates: list[np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    stack = np.stack([x.astype(np.float32) for x in candidates], axis=0)
    uncertainty = cv2.GaussianBlur(np.mean(np.std(stack, axis=0), axis=2), (0, 0), 3.0)
    normalized = np.clip(uncertainty / 32.0, 0.0, 1.0)
    confidence = (1.0 - normalized).astype(np.float32)
    hard = np.asarray(mask, dtype=bool)
    confidence[~hard] = 0.0
    uncertain_ratio = float(np.mean((confidence[hard] < 0.5))) if np.any(hard) else 1.0
    return (normalized * 255.0).astype(np.uint8), confidence, uncertain_ratio


def _structural_anchor(crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    anchor = np.zeros(crop_mask.shape, dtype=bool)
    line_segments: list[tuple[float, float, float, float]] = []
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = detector.detect(gray)[0]
        if lines is not None:
            line_segments.extend(tuple(map(float, line[0])) for line in lines)
    except Exception:
        pass
    if not line_segments:
        hough = cv2.HoughLinesP(gray, 1, np.pi / 180.0, threshold=35, minLineLength=28, maxLineGap=8)
        if hough is not None:
            line_segments.extend(tuple(map(float, line)) for line in np.asarray(hough).reshape(-1, 4))
    h, w = crop_mask.shape
    for x0, y0, x1, y1 in line_segments:
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 28:
            continue
        xs = np.clip(np.rint(np.linspace(x0, x1, max(16, int(length * 1.5)))), 0, w - 1).astype(int)
        ys = np.clip(np.rint(np.linspace(y0, y1, max(16, int(length * 1.5)))), 0, h - 1).astype(int)
        inside = crop_mask[ys, xs]
        if int(np.sum(inside)) >= 3 and int(np.sum(~inside)) >= 3:
            cv2.line(anchor, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, 3)
    return anchor.astype(bool)


def _metric_row(sample_id: str, degradation: str, method: str, seed: str, target: np.ndarray, output: np.ndarray, mask: np.ndarray, runtime: float, pairwise_mad: float, edge_iou: float, uncertain_ratio: float, selected: bool, rejection_reason: str) -> dict[str, object]:
    hard = np.asarray(mask, dtype=bool)
    outside = ~hard
    delta = np.abs(output.astype(np.int16) - target.astype(np.int16))
    target_lab, output_lab = _lab(target), _lab(output)
    low_target = cv2.GaussianBlur(target_lab, (0, 0), 12.0)
    low_output = cv2.GaussianBlur(output_lab, (0, 0), 12.0)
    eroded = cv2.erode(hard.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    ring = hard & ~eroded
    median_shift = float(np.mean(np.abs(np.median(output_lab[hard], axis=0) - np.median(target_lab[hard], axis=0)))) if np.any(hard) else 0.0
    boundary_lab = float(np.mean(np.abs(output_lab[ring] - target_lab[ring]))) if np.any(ring) else 0.0
    gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_target = cv2.magnitude(cv2.Sobel(gray_target, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray_target, cv2.CV_32F, 0, 1, 3))
    grad_output = cv2.magnitude(cv2.Sobel(gray_output, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray_output, cv2.CV_32F, 0, 1, 3))
    return {
        "sample_id": sample_id,
        "degradation_type": degradation,
        "method": method,
        "seed": seed,
        "outside_max_abs": int(delta[outside].max()) if np.any(outside) else 0,
        "changed_fraction_mask": float(np.mean(np.mean(delta, axis=2)[hard] > 2.0)) if np.any(hard) else 0.0,
        "mean_abs_delta_mask": float(np.mean(delta[hard])) if np.any(hard) else 0.0,
        "low_frequency_lab_l1": float(np.mean(np.abs(low_output[hard] - low_target[hard]))) if np.any(hard) else 0.0,
        "mask_median_lab_shift": median_shift,
        "boundary_lab_l1": boundary_lab,
        "boundary_gradient_discontinuity": float(np.mean(np.abs(grad_output[ring] - grad_target[ring]))) if np.any(ring) else 0.0,
        "input_edge_retention": _edge_f1(target, output, hard),
        "pairwise_seed_mad": pairwise_mad,
        "seed_edge_iou": edge_iou,
        "uncertain_mask_ratio": uncertain_ratio,
        "runtime_sec": float(runtime),
        "selected": int(bool(selected)),
        "rejection_reason": rejection_reason,
    }


def _zoom(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 20) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin]


def _tile(full: np.ndarray, zoom: np.ndarray, label: str, size: tuple[int, int] = (320, 300)) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    top_h, bottom_h = size[1] - 105, 88
    full_img = Image.fromarray(np.asarray(full, dtype=np.uint8)).convert("RGB")
    full_img.thumbnail((size[0] - 8, top_h - 6), Image.Resampling.LANCZOS)
    tile.paste(full_img, ((size[0] - full_img.width) // 2, 3))
    zoom_img = Image.fromarray(np.asarray(zoom, dtype=np.uint8)).convert("RGB")
    zoom_img.thumbnail((size[0] - 8, bottom_h - 5), Image.Resampling.LANCZOS)
    tile.paste(zoom_img, ((size[0] - zoom_img.width) // 2, top_h + 2))
    ImageDraw.Draw(tile).text((5, size[1] - 15), label[:52], fill="black", font=_font(14))
    return tile


def _na_tile(label: str, size: tuple[int, int] = (320, 300)) -> Image.Image:
    tile = Image.new("RGB", size, (238, 238, 238))
    draw = ImageDraw.Draw(tile)
    draw.text((size[0] // 2 - 18, size[1] // 2 - 10), "N/A", fill=(80, 80, 80), font=_font(20))
    draw.text((5, size[1] - 15), label[:52], fill="black", font=_font(14))
    return tile


def _load_fill_pipeline(model_path: Path):
    import torch
    from diffusers import FluxFillPipeline

    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    pipe = FluxFillPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    if torch.cuda.is_available():
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gib >= 20:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
    return pipe, torch


def _run_fill(pipe, torch_module, crop: np.ndarray, fill_mask: np.ndarray, seed: int, steps: int) -> np.ndarray:
    h, w = crop.shape[:2]
    gen_h, gen_w = max(64, (h // 16) * 16), max(64, (w // 16) * 16)
    image = Image.fromarray(crop, mode="RGB").resize((gen_w, gen_h), Image.Resampling.LANCZOS)
    mask = Image.fromarray((fill_mask.astype(np.uint8) * 255), mode="L").resize((gen_w, gen_h), Image.Resampling.NEAREST)
    result = pipe(
        prompt=PROMPT,
        image=image,
        mask_image=mask,
        height=gen_h,
        width=gen_w,
        guidance_scale=30,
        num_inference_steps=steps,
        max_sequence_length=512,
        generator=torch_module.Generator("cpu").manual_seed(seed),
    ).images[0]
    return np.asarray(result.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), dtype=np.uint8)


def _write_commands(out: Path, source: Path) -> None:
    out.joinpath("run_commands.md").write_text(
        "# FLUX fidelity quick6 commands\n\n"
        "P0 post-processing from the committed FLUX.2 cloud output:\n\n"
        "```bash\n"
        "python research/restoration_v2/flux_fidelity.py --no-fill\n"
        "```\n\n"
        "Official FLUX.1-Fill-dev smoke/full run on the V100 (weights are not committed):\n\n"
        "```bash\n"
        "FLUX_FILL_MODEL_PATH=/home/vipuser/models/FLUX.1-Fill-dev \\\n+  /home/vipuser/fluxenv/bin/python research/restoration_v2/flux_fidelity.py \\\n+  --fill-model-path /home/vipuser/models/FLUX.1-Fill-dev --fill-steps 50\n"
        "```\n\n"
        f"Source baseline: `{source}`\n"
        "Official implementation: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev\n",
        encoding="utf-8",
    )


def _smoke_only(rows: list[dict[str, str]], source: Path, out: Path, pipe, torch_module, context: int, feather: int, steps: int) -> dict[str, object]:
    row = rows[0]
    sample_dir = source / row["sample_id"]
    target = load_rgb(sample_dir / "input.png")
    manual = load_mask(sample_dir / "mask.png")
    crop, crop_mask, bbox = _crop_context(target, manual, context)
    anchor_crop = _structural_anchor(crop, crop_mask)
    fill_crop = cv2.erode(crop_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool) & ~anchor_crop
    smoke_dir = out / "fill_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(smoke_dir / "input.png", target)
    save_mask(smoke_dir / "manual_mask.png", manual)
    save_mask(smoke_dir / "structural_anchor.png", _full_from_crop(manual.shape, anchor_crop, bbox))
    save_mask(smoke_dir / "fill_mask.png", _full_from_crop(manual.shape, fill_crop, bbox))
    started = time.perf_counter()
    raw_crop = _run_fill(pipe, torch_module, crop, fill_crop, 0, steps)
    raw_full = _paste_crop(target, raw_crop, bbox)
    allowed = _full_from_crop(manual.shape, fill_crop, bbox)
    composite = _composite(target, raw_full, allowed, feather)
    save_rgb(smoke_dir / "raw.png", raw_full)
    save_rgb(smoke_dir / "composite.png", composite)
    result = {"status": "completed", "sample_id": row["sample_id"], "outside_max_abs": int(np.abs(composite.astype(np.int16) - target.astype(np.int16))[~manual].max()), "runtime_sec": time.perf_counter() - started}
    (out / "fill_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick6", type=Path, default=ROOT / "research" / "data" / "flux_quick6.csv")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--fill-model-path", type=Path, default=None)
    parser.add_argument("--fill-blocker-note", type=str, default="", help="Record an externally verified Fill access blocker without running a fake backend.")
    parser.add_argument("--fill-steps", type=int, default=50)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--feather", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-fill", action="store_true")
    args = parser.parse_args()
    source = _resolve(args.source_dir)
    out = _resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(_resolve(args.quick6))
    _write_commands(out, source)

    fill_pipe = fill_torch = None
    fill_status = "not_requested"
    fill_blockers: list[str] = []
    fill_env = os.environ.get("FLUX_FILL_MODEL_PATH", "")
    fill_path = args.fill_model_path or (Path(fill_env) if fill_env else None)
    if args.fill_blocker_note:
        fill_status = "blocked_external_access"
        fill_blockers.append(args.fill_blocker_note)
    elif not args.no_fill and fill_path is not None and fill_path.exists():
        try:
            fill_pipe, fill_torch = _load_fill_pipeline(fill_path)
            fill_status = "flux1_fill_diffusers"
        except Exception as exc:
            fill_status = "blocked_pipeline_load"
            fill_blockers.append(f"pipeline_load_failed: {type(exc).__name__}: {exc}")
    elif not args.no_fill:
        fill_status = "blocked_checkpoint_missing"
        fill_blockers.append("FLUX.1-Fill-dev checkpoint missing; pass --fill-model-path or FLUX_FILL_MODEL_PATH")

    if args.smoke_only:
        if fill_pipe is None:
            result = {"status": fill_status, "blockers": fill_blockers}
            (out / "fill_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
            return
        result = _smoke_only(rows, source, out, fill_pipe, fill_torch, args.context, args.feather, args.fill_steps)
        print(json.dumps(result, indent=2))
        return

    all_metrics: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    anchor_overlay_tiles = []
    for sample_index, row in enumerate(rows):
        sample_id, degradation = row["sample_id"], row["degradation_type"]
        source_sample = source / sample_id
        sample_out = out / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        target = load_rgb(source_sample / "input.png")
        manual = load_mask(source_sample / "mask.png")
        crop, crop_mask, bbox = _crop_context(target, manual, args.context)
        anchor_crop = _structural_anchor(crop, crop_mask)
        eroded_crop = cv2.erode(crop_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        fill_crop = eroded_crop & ~anchor_crop
        anchor_full = _full_from_crop(manual.shape, anchor_crop, bbox)
        fill_full = _full_from_crop(manual.shape, fill_crop, bbox)
        save_rgb(sample_out / "input.png", target)
        save_mask(sample_out / "manual_mask.png", manual)
        save_mask(sample_out / "structural_anchor.png", anchor_full)
        save_mask(sample_out / "fill_mask.png", fill_full)
        overlay = np.zeros_like(target)
        overlay[manual] = (255, 80, 30)
        overlay[anchor_full] = (30, 220, 70)
        overlay[~manual] = target[~manual]
        anchor_overlay_tiles.append((overlay, _zoom(overlay, bbox), f"{sample_index}: mask/anchor"))

        current = load_rgb(source_sample / "selected.png")
        baseline_seed_images = [load_rgb(source_sample / f"target_only_seed{seed}.png") for seed in (0, 1) if (source_sample / f"target_only_seed{seed}.png").exists()]
        baseline_stats = _seed_stats(baseline_seed_images, manual)
        baseline_mad = float(baseline_stats["pairwise_seed_mad"])
        baseline_edge_iou = float(baseline_stats["seed_edge_iou"])
        started = time.perf_counter()
        lab_anchored = _lab_anchor(target, current, manual)
        frequency_fused = _frequency_fuse(target, lab_anchored, manual)
        p0_runtime = time.perf_counter() - started
        save_rgb(sample_out / "lab_anchored.png", lab_anchored)
        save_rgb(sample_out / "frequency_fused.png", frequency_fused)
        all_metrics.append(_metric_row(sample_id, degradation, "identity", "", target, target, manual, 0.0, baseline_mad, baseline_edge_iou, 0.0, False, "baseline"))
        all_metrics.append(_metric_row(sample_id, degradation, "current_FLUX2", "", target, current, manual, 0.0, baseline_mad, baseline_edge_iou, 0.0, False, "fixed_baseline"))
        all_metrics.append(_metric_row(sample_id, degradation, "Lab_anchor", "", target, lab_anchored, manual, p0_runtime, baseline_mad, baseline_edge_iou, 0.0, False, "postprocess_only"))
        all_metrics.append(_metric_row(sample_id, degradation, "frequency_fusion", "", target, frequency_fused, manual, p0_runtime, baseline_mad, baseline_edge_iou, 0.0, False, "postprocess_only"))

        fill_raws: list[np.ndarray] = []
        fill_composites: list[np.ndarray] = []
        constrained: list[np.ndarray] = []
        fill_runtime = 0.0
        if fill_pipe is not None:
            for seed in FILL_SEEDS:
                started = time.perf_counter()
                raw_crop = _run_fill(fill_pipe, fill_torch, crop, fill_crop, seed, args.fill_steps)
                fill_runtime += time.perf_counter() - started
                raw_full = _paste_crop(target, raw_crop, bbox)
                composite = _composite(target, raw_full, fill_full, args.feather)
                anchored = _lab_anchor(target, composite, fill_full)
                constrained_output = _frequency_fuse(target, anchored, fill_full)
                fill_raws.append(raw_full)
                fill_composites.append(composite)
                constrained.append(constrained_output)
                save_rgb(sample_out / f"fill_seed_{seed}_raw.png", raw_full)
                save_rgb(sample_out / f"fill_seed_{seed}_composite.png", composite)
                save_rgb(sample_out / f"fill_seed_{seed}_constrained.png", constrained_output)
                all_metrics.append(_metric_row(sample_id, degradation, "Fill_seed", str(seed), target, composite, manual, time.perf_counter() - started, 0.0, 1.0, 0.0, False, "candidate"))

            seed_stats = _seed_stats(constrained, fill_full)
            uncertainty_image, confidence, uncertain_ratio = _uncertainty(constrained, fill_full)
            medoid = constrained[int(seed_stats["medoid_index"])]
            fill_lab = _lab_anchor(target, medoid, fill_full)
            fill_frequency = _frequency_fuse(target, fill_lab, fill_full)
            full_output = _composite(target, fill_frequency, fill_full, args.feather, confidence)
            save_rgb(sample_out / "seed_uncertainty.png", np.repeat(uncertainty_image[..., None], 3, axis=2))
            save_rgb(sample_out / "selected_medoid.png", medoid)
            save_rgb(sample_out / "fill_lab_anchored.png", fill_lab)
            save_rgb(sample_out / "fill_frequency_fused.png", fill_frequency)
            save_rgb(sample_out / "full_output.png", full_output)
            save_rgb(sample_out / "full_zoom.png", _zoom(full_output, bbox))
            seed_mad, seed_edge_iou = float(seed_stats["pairwise_seed_mad"]), float(seed_stats["seed_edge_iou"])
            full_row = _metric_row(sample_id, degradation, "Full", "medoid", target, full_output, manual, fill_runtime, seed_mad, seed_edge_iou, uncertain_ratio, False, "candidate")
            full_low = float(full_row["low_frequency_lab_l1"])
            baseline_row = _metric_row(sample_id, degradation, "current_FLUX2", "", target, current, manual, 0.0, baseline_mad, baseline_edge_iou, 0.0, False, "fixed_baseline")
            reasons = []
            if int(full_row["outside_max_abs"]) != 0:
                reasons.append("outside_changed")
            if seed_mad >= max(1e-6, baseline_mad):
                reasons.append("seed_instability_not_reduced")
            if full_low > float(baseline_row["low_frequency_lab_l1"]):
                reasons.append("low_frequency_drift_not_reduced")
            if uncertain_ratio > 0.95:
                reasons.append("reliable_confidence_too_small")
            if float(full_row["input_edge_retention"]) < 0.5:
                reasons.append("edge_retention_low")
            rejection = ";".join(reasons)
            accepted = not rejection
            all_metrics.extend([
                _metric_row(sample_id, degradation, "Fill_medoid", "medoid", target, medoid, manual, fill_runtime, seed_mad, seed_edge_iou, uncertain_ratio, False, "candidate"),
                _metric_row(sample_id, degradation, "Fill_Lab_anchor", "medoid", target, fill_lab, manual, fill_runtime, seed_mad, seed_edge_iou, uncertain_ratio, False, "candidate"),
                _metric_row(sample_id, degradation, "Fill_frequency_fusion", "medoid", target, fill_frequency, manual, fill_runtime, seed_mad, seed_edge_iou, uncertain_ratio, False, "candidate"),
                full_row,
            ])
            selected = full_output if accepted else target.copy()
            selected_method = "Full" if accepted else "identity"
            records.append({"sample_id": sample_id, "degradation_type": degradation, "selected": selected_method, "rejection_reason": rejection, "baseline_seed_mad": baseline_mad, "full_seed_mad": seed_mad, "baseline_low_frequency_lab_l1": float(baseline_row["low_frequency_lab_l1"]), "full_low_frequency_lab_l1": full_low, "uncertain_mask_ratio": uncertain_ratio})
            stability_rows.append({"sample_id": sample_id, "degradation_type": degradation, "method": "current_FLUX2", "pairwise_seed_mad": baseline_mad, "seed_edge_iou": baseline_edge_iou, "uncertain_mask_ratio": ""})
            stability_rows.append({"sample_id": sample_id, "degradation_type": degradation, "method": "Full", "pairwise_seed_mad": seed_mad, "seed_edge_iou": seed_edge_iou, "uncertain_mask_ratio": uncertain_ratio})
            fill_raw_for_sheet = fill_raws[0]
            full_for_sheet = selected
            uncertainty_for_sheet = np.repeat(uncertainty_image[..., None], 3, axis=2)

            # Two extra calls total: compare a full manual mask with the anchored mask on two samples.
            if sample_index < 2:
                ablation_raw_crop = _run_fill(fill_pipe, fill_torch, crop, crop_mask, 0, args.fill_steps)
                ablation_raw = _paste_crop(target, ablation_raw_crop, bbox)
                save_rgb(sample_out / "full_mask_fill_raw.png", ablation_raw)
                save_rgb(sample_out / "full_mask_fill.png", _composite(target, ablation_raw, manual, args.feather))
                save_rgb(sample_out / "anchor_mask_fill.png", fill_composites[0])
        else:
            seed_stats = {"pairwise_seed_mad": 0.0, "seed_edge_iou": 1.0}
            uncertain_ratio = 1.0
            selected = target.copy()
            selected_method = "identity"
            rejection = ";".join(fill_blockers) if fill_blockers else "fill_not_run"
            records.append({"sample_id": sample_id, "degradation_type": degradation, "selected": selected_method, "rejection_reason": rejection, "baseline_seed_mad": baseline_mad, "full_seed_mad": "", "baseline_low_frequency_lab_l1": float(_metric_row(sample_id, degradation, "current_FLUX2", "", target, current, manual, 0, baseline_mad, baseline_edge_iou, 0, False, "")["low_frequency_lab_l1"]), "full_low_frequency_lab_l1": "", "uncertain_mask_ratio": uncertain_ratio})
            stability_rows.append({"sample_id": sample_id, "degradation_type": degradation, "method": "current_FLUX2", "pairwise_seed_mad": baseline_mad, "seed_edge_iou": baseline_edge_iou, "uncertain_mask_ratio": ""})
            fill_raw_for_sheet = None
            full_for_sheet = target
            uncertainty_for_sheet = np.zeros_like(target)
        save_rgb(sample_out / "difference.png", _difference_image(target, selected, manual))
        save_rgb(sample_out / "selected.png", selected)

    metric_fields = ["sample_id", "degradation_type", "method", "seed", "outside_max_abs", "changed_fraction_mask", "mean_abs_delta_mask", "low_frequency_lab_l1", "mask_median_lab_shift", "boundary_lab_l1", "boundary_gradient_discontinuity", "input_edge_retention", "pairwise_seed_mad", "seed_edge_iou", "uncertain_mask_ratio", "runtime_sec", "selected", "rejection_reason"]
    with (out / "fidelity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    with (out / "seed_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "degradation_type", "method", "pairwise_seed_mad", "seed_edge_iou", "uncertain_mask_ratio"])
        writer.writeheader()
        writer.writerows(stability_rows)

    # The contact sheet keeps both full view and enlarged mask crop in every tile.
    contact_rows = []
    for record, anchor_info in zip(records, anchor_overlay_tiles):
        sample_out = out / record["sample_id"]
        input_image = load_rgb(sample_out / "input.png")
        current_image = load_rgb(source / record["sample_id"] / "selected.png")
        lab_image = load_rgb(sample_out / "lab_anchored.png")
        freq_image = load_rgb(sample_out / "frequency_fused.png")
        full_path = sample_out / "full_output.png"
        full_image = load_rgb(full_path) if full_path.exists() else input_image
        uncertainty_path = sample_out / "seed_uncertainty.png"
        uncertainty_image = load_rgb(uncertainty_path) if uncertainty_path.exists() else np.zeros_like(input_image)
        bbox = _crop_context(input_image, load_mask(sample_out / "manual_mask.png"), 0)[2]
        fill_raw_path = sample_out / "fill_seed_0_raw.png"
        contact_rows.append([
            _tile(input_image, _zoom(input_image, bbox), f"{record['sample_id']} input"),
            _tile(anchor_info[0], anchor_info[1], "mask/anchor"),
            _tile(current_image, _zoom(current_image, bbox), "current FLUX2"),
            _tile(load_rgb(fill_raw_path), _zoom(load_rgb(fill_raw_path), bbox), "Fill raw") if fill_raw_path.exists() else _na_tile("Fill raw blocked"),
            _tile(lab_image, _zoom(lab_image, bbox), "Lab anchor"),
            _tile(freq_image, _zoom(freq_image, bbox), "frequency fusion"),
            _tile(full_image, _zoom(full_image, bbox), f"Full / {record['selected']}"),
            _tile(uncertainty_image, _zoom(uncertainty_image, bbox), "seed uncertainty"),
        ])
    tile_w, tile_h = 320, 300
    contact = Image.new("RGB", (8 * tile_w, len(contact_rows) * tile_h), "white")
    for r_index, tiles in enumerate(contact_rows):
        for c_index, tile in enumerate(tiles):
            contact.paste(tile, (c_index * tile_w, r_index * tile_h))
    contact.save(out / "fidelity_ablation_contact_sheet.png")

    rng = random.Random(20260821)
    blind_key, fidelity_rows, identity_rows = {}, [], []
    for record in records:
        sid = record["sample_id"]
        sample_out = out / sid
        current = load_rgb(source / sid / "selected.png")
        full_path = sample_out / "full_output.png"
        full = load_rgb(full_path) if full_path.exists() else load_rgb(sample_out / "input.png")
        swap = bool(rng.randrange(2))
        a_name, b_name = ("Full", "current_FLUX2") if swap else ("current_FLUX2", "Full")
        fidelity_rows.append((sid, a_name, b_name, current if a_name == "current_FLUX2" else full, full if b_name == "Full" else current))
        swap_identity = bool(rng.randrange(2))
        ia_name, ib_name = ("Full", "identity") if swap_identity else ("identity", "Full")
        identity = load_rgb(sample_out / "input.png")
        identity_rows.append((sid, ia_name, ib_name, full if ia_name == "Full" else identity, full if ib_name == "Full" else identity))
        blind_key[sid] = {"fidelity_A": a_name, "fidelity_B": b_name, "identity_A": ia_name, "identity_B": ib_name}

    def save_blind(path: Path, rows_to_draw: list[tuple[str, str, str, np.ndarray, np.ndarray]], title: str) -> None:
        tile_w, tile_h = 430, 290
        canvas = Image.new("RGB", (3 * tile_w, len(rows_to_draw) * tile_h), "white")
        for idx, (sid, a_name, b_name, a_img, b_img) in enumerate(rows_to_draw):
            for col, (label, image) in enumerate((("input", load_rgb(out / sid / "input.png")), ("A", a_img), ("B", b_img))):
                tile = _tile(image, _zoom(image, _crop_context(image, load_mask(out / sid / "manual_mask.png"), 0)[2]), f"{sid[:28]} {label}", (tile_w, tile_h))
                canvas.paste(tile, (col * tile_w, idx * tile_h))
        canvas.save(path)

    save_blind(out / "fidelity_vs_quality_blind.png", fidelity_rows, "fidelity")
    save_blind(out / "identity_vs_full_blind.png", identity_rows, "identity")
    (out / "blind_key.json").write_text(json.dumps(blind_key, indent=2), encoding="utf-8")
    with (out / "human_preference_fidelity.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "overall_preference", "sharpness_improvement", "wall_color_fidelity", "material_fidelity", "geometry_window_fidelity", "visible_seam", "acceptable_for_Open3DHK_showcase", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"sample_id": row["sample_id"], **{field: "" for field in fields[1:]}})

    identity_count = sum(record["selected"] == "identity" for record in records)
    summary = {
        "status": "completed",
        "go_decision": "AWAIT_HUMAN_REVIEW",
        "source_baseline": str(source.relative_to(ROOT)),
        "fill_backend_status": fill_status,
        "fill_blockers": fill_blockers,
        "images": len(records),
        "formal_fill_calls": len(records) * len(FILL_SEEDS) if fill_pipe is not None else 0,
        "anchor_ablation_extra_calls": min(2, len(records)) if fill_pipe is not None else 0,
        "identity_selected_count": identity_count,
        "full_selected_count": len(records) - identity_count,
        "outside_max_abs_requirement": "see fidelity_metrics.csv; strict composite required",
        "subjective_quality_claim": "none; human_preference_fidelity.csv is intentionally blank",
        "outputs": {
            "contact_sheet": str((out / "fidelity_ablation_contact_sheet.png").relative_to(ROOT)),
            "fidelity_blind": str((out / "fidelity_vs_quality_blind.png").relative_to(ROOT)),
            "identity_blind": str((out / "identity_vs_full_blind.png").relative_to(ROOT)),
            "metrics": str((out / "fidelity_metrics.csv").relative_to(ROOT)),
            "stability": str((out / "seed_stability.csv").relative_to(ROOT)),
            "human_preference": str((out / "human_preference_fidelity.csv").relative_to(ROOT)),
        },
        "records": records,
    }
    (out / "fidelity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "server_runtime.log").write_text("Runner output is captured by the invoking shell.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
