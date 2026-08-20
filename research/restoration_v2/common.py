#!/usr/bin/env python3
"""Shared utilities for the second restoration cycle."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: object) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate((np.asarray(mask) > 0).astype(np.uint8), kernel) > 0


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode((np.asarray(mask) > 0).astype(np.uint8), kernel) > 0


def soft_composite(input_rgb: np.ndarray, candidate_rgb: np.ndarray, mask: np.ndarray, dilate_px: int = 3, feather: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Composite only in mask plus a narrow feather support."""
    hard = np.asarray(mask) > 0
    allowed = dilate(hard, dilate_px)
    alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), feather)
    alpha[~allowed] = 0.0
    alpha[hard] = np.maximum(alpha[hard], 0.5)
    result = np.rint(alpha[..., None] * candidate_rgb.astype(np.float32) + (1.0 - alpha[..., None]) * input_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    result[~allowed] = input_rgb[~allowed]
    return result, allowed


def strict_proxy_composite(clean: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact P0 paired-data construction: degraded only where M=1."""
    result = clean.copy()
    result[np.asarray(mask) > 0] = degraded[np.asarray(mask) > 0]
    return result


def masked_psnr(output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(mask) > 0
    if not np.any(region):
        return float("nan")
    delta = output.astype(np.float32)[region] - target.astype(np.float32)[region]
    mse = float(np.mean(delta * delta))
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def masked_ssim(output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(mask) > 0
    if np.count_nonzero(region) < 16:
        return float("nan")
    _, ssim_map = structural_similarity(output, target, channel_axis=2, data_range=255, full=True)
    return float(np.asarray(ssim_map).mean(axis=2)[region].mean())


def lpips_distance(metric, output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """LPIPS over the mask bounding crop; metric is a loaded lpips.LPIPS model."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) < 16:
        return float("nan")
    y0, y1 = max(0, int(ys.min()) - 4), min(output.shape[0], int(ys.max()) + 5)
    x0, x1 = max(0, int(xs.min()) - 4), min(output.shape[1], int(xs.max()) + 5)
    m = np.asarray(mask[y0:y1, x0:x1], dtype=np.float32)
    a = output[y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    b = target[y0:y1, x0:x1].astype(np.float32) / 127.5 - 1.0
    import torch

    with torch.no_grad():
        value = metric(torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda(), torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).cuda())
    # LPIPS is computed on the crop; report it only as a masked-region proxy.
    return float(value.mean().item())


def mean_edge(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def metric_row(input_rgb: np.ndarray, output_rgb: np.ndarray, mask: np.ndarray, target: np.ndarray | None = None) -> dict[str, float]:
    hard = np.asarray(mask) > 0
    allowed = dilate(hard, 3)
    outside_delta = np.abs(output_rgb.astype(np.int16) - input_rgb.astype(np.int16))
    ring = allowed & ~hard
    output_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    input_lab = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    result = {
        "mask_ratio": float(hard.mean()),
        "allowed_outside_max_abs": int(outside_delta[~allowed].max()) if np.any(~allowed) else 0,
        "allowed_outside_mean_abs": float(outside_delta[~allowed].mean()) if np.any(~allowed) else 0.0,
        "boundary_color_lab_l1": float(np.abs(output_lab[ring] - input_lab[ring]).mean()) if np.any(ring) else 0.0,
        "boundary_edge_l1": float(np.abs(mean_edge(output_rgb)[ring] - mean_edge(input_rgb)[ring]).mean()) if np.any(ring) else 0.0,
        "masked_psnr": float("nan"),
        "masked_ssim": float("nan"),
        "masked_lpips": float("nan"),
    }
    if target is not None:
        result["masked_psnr"] = masked_psnr(output_rgb, target, hard)
        result["masked_ssim"] = masked_ssim(output_rgb, target, hard)
    return result


def resize_keep(rgb: np.ndarray, size: int = 512) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)


def make_grid(tiles: list[tuple[str, np.ndarray]], output: Path, columns: int = 6, tile_size: tuple[int, int] = (220, 190)) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tw, th = tile_size
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tw, rows * th), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for index, (label, tile) in enumerate(tiles):
        x, y = (index % columns) * tw, (index // columns) * th
        image = Image.fromarray(np.asarray(tile, dtype=np.uint8)).convert("RGB")
        image.thumbnail((tw - 8, th - 26), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (tw - image.width) // 2, y + 2))
        draw.text((x + 4, y + th - 20), label[:42], fill="black")
    canvas.save(output)
