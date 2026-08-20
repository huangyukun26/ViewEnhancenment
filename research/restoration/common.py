#!/usr/bin/env python3
"""Small, dependency-light utilities for the short restoration cycle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path | None = None) -> str:
    root = root or repo_root()
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(path)


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def odd_kernel(radius: int) -> int:
    return max(3, 2 * int(radius) + 1)


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if radius <= 0:
        return mask > 0
    return np.asarray(Image.fromarray(mask, mode="L").filter(ImageFilter.MaxFilter(odd_kernel(radius)))) > 0


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if radius <= 0:
        return mask > 0
    return np.asarray(Image.fromarray(mask, mode="L").filter(ImageFilter.MinFilter(odd_kernel(radius)))) > 0


def soft_alpha(mask: np.ndarray, dilate: int = 0, feather: float = 2.0) -> np.ndarray:
    """Create an alpha that is zero outside mask/dilated-mask."""
    hard = (np.asarray(mask) > 0).astype(np.uint8) * 255
    allowed = binary_dilate(hard, dilate)
    image = Image.fromarray(hard, mode="L")
    if feather > 0:
        image = image.filter(ImageFilter.GaussianBlur(float(feather)))
    alpha = np.asarray(image, dtype=np.float32) / 255.0
    alpha[~allowed] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def strict_composite(input_rgb: np.ndarray, candidate_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    input_rgb = np.asarray(input_rgb, dtype=np.uint8)
    candidate_rgb = np.asarray(candidate_rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.float32)
    result = np.rint(
        alpha[..., None] * candidate_rgb.astype(np.float32)
        + (1.0 - alpha[..., None]) * input_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    result[alpha <= 0] = input_rgb[alpha <= 0]
    return result


def deterministic_candidate(input_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """A conservative CPU restoration proxy: local contrast/detail recovery."""
    image = Image.fromarray(input_rgb, mode="RGB")
    radius = 1.5 + 0.6 * float(strength)
    percent = int(70 + 45 * float(strength))
    sharp = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
    return np.asarray(sharp, dtype=np.uint8)


def stochastic_candidate(input_rgb: np.ndarray, seed: int, strength: float = 0.22) -> np.ndarray:
    """Low-intensity stochastic candidate used when diffusion is unavailable.

    This is deliberately labelled a proxy, not a diffusion result. It adds a
    tiny seed-controlled texture perturbation to a conservative sharpened image.
    """
    rng = np.random.default_rng(int(seed))
    sharp = deterministic_candidate(input_rgb, strength=0.75).astype(np.float32)
    smooth = np.asarray(Image.fromarray(input_rgb, mode="RGB").filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    noise = rng.normal(0.0, 1.8 * float(strength) / 0.22, size=sharp.shape[:2] + (1,))
    result = 0.75 * sharp + 0.25 * smooth + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    import io

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def scale_roundtrip(rgb: np.ndarray, factor: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    small = (max(8, int(round(w * factor))), max(8, int(round(h * factor))))
    image = Image.fromarray(rgb, mode="RGB").resize(small, Image.Resampling.BICUBIC)
    return np.asarray(image.resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)


def infer_diff_mask(input_rgb: np.ndarray, reference_rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
    diff = np.abs(reference_rgb.astype(np.float32) - input_rgb.astype(np.float32)).mean(axis=2)
    nonzero = diff[diff > 0]
    if nonzero.size == 0:
        return np.zeros(diff.shape, dtype=bool), 0.0, 0.0
    threshold = max(18.0, float(np.percentile(nonzero, 80)))
    raw = (diff >= threshold).astype(np.uint8) * 255
    closed = Image.fromarray(raw, mode="L").filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    return np.asarray(closed) > 0, float(diff.mean()), threshold


def psnr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(mask) > 0
    if not np.any(region):
        return float("nan")
    delta = a.astype(np.float32)[region] - b.astype(np.float32)[region]
    mse = float(np.mean(delta * delta))
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(mask) > 0
    if np.count_nonzero(region) < 2:
        return float("nan")
    x = a.astype(np.float64).mean(axis=2)[region]
    y = b.astype(np.float64).mean(axis=2)[region]
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))


def gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def boundary_features(input_rgb: np.ndarray, output_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    hard = np.asarray(mask) > 0
    ring_out = binary_dilate(hard, 3) & ~hard
    ring_in = hard & ~binary_erode(hard, 3)
    delta = np.abs(output_rgb.astype(np.float32) - input_rgb.astype(np.float32)).mean(axis=2)
    input_grad = gradient_magnitude(input_rgb)
    output_grad = gradient_magnitude(output_rgb)
    grad_delta = np.abs(output_grad - input_grad)
    region = hard if np.any(hard) else np.ones(hard.shape, dtype=bool)
    input_sharp = float(input_grad[region].mean())
    output_sharp = float(output_grad[region].mean())
    if np.any(ring_out):
        boundary_color = float(delta[ring_out].mean())
        boundary_edge = float(grad_delta[ring_out].mean())
    else:
        boundary_color = 0.0
        boundary_edge = 0.0
    if np.any(ring_in):
        inner_color = float(delta[ring_in].mean())
    else:
        inner_color = 0.0
    in_pixels = output_rgb[region].astype(np.float32)
    ref_pixels = input_rgb[region].astype(np.float32)
    mean_delta = float(np.abs(in_pixels.mean(axis=0) - ref_pixels.mean(axis=0)).mean())
    std_delta = float(np.abs(in_pixels.std(axis=0) - ref_pixels.std(axis=0)).mean())
    return {
        "boundary_color_l1": boundary_color,
        "boundary_edge_l1": boundary_edge,
        "inner_color_l1": inner_color,
        "color_mean_delta": mean_delta,
        "color_std_delta": std_delta,
        "sharpness_gain": output_sharp - input_sharp,
        "input_sharpness": input_sharp,
        "output_sharpness": output_sharp,
    }


def metric_row(input_rgb: np.ndarray, output_rgb: np.ndarray, mask: np.ndarray, gt: np.ndarray | None = None) -> dict[str, float]:
    hard = np.asarray(mask) > 0
    # Candidates are allowed to blend only inside the hard mask dilated by
    # the fixed 3 px feather boundary. Report strict preservation outside
    # that allowed support; also keep the original-mask statistic separately
    # so boundary blending is not mistaken for an unconstrained edit.
    allowed = binary_dilate(hard, 3)
    outside = ~allowed
    original_outside = ~hard
    outside_delta = np.abs(output_rgb.astype(np.int16) - input_rgb.astype(np.int16))
    features = boundary_features(input_rgb, output_rgb, hard)
    row: dict[str, float] = {
        "mask_ratio": float(hard.mean()),
        "mask_outside_max_abs": int(outside_delta[outside].max()) if np.any(outside) else 0,
        "mask_outside_mean_abs": float(outside_delta[outside].mean()) if np.any(outside) else 0.0,
        "original_mask_outside_max_abs": int(outside_delta[original_outside].max()) if np.any(original_outside) else 0,
        "original_mask_outside_mean_abs": float(outside_delta[original_outside].mean()) if np.any(original_outside) else 0.0,
        "masked_psnr": float("nan"),
        "masked_ssim": float("nan"),
        "masked_lpips": float("nan"),
    }
    row.update(features)
    if gt is not None:
        row["masked_psnr"] = psnr_masked(output_rgb, gt, hard)
        row["masked_ssim"] = ssim_masked(output_rgb, gt, hard)
        row["masked_l1"] = float(np.abs(output_rgb.astype(np.float32)[hard] - gt.astype(np.float32)[hard]).mean()) if np.any(hard) else 0.0
    return row


def masked_output_std(outputs: list[np.ndarray], mask: np.ndarray) -> float:
    """Mean per-pixel RGB standard deviation across fixed random seeds."""
    hard = np.asarray(mask) > 0
    if not outputs or not np.any(hard):
        return 0.0
    stack = np.stack([np.asarray(output, dtype=np.float32) for output in outputs], axis=0)
    pixel_std = stack.std(axis=0).mean(axis=2)
    return float(pixel_std[hard].mean())


def feature_vector(metrics: dict[str, float]) -> np.ndarray:
    """No-reference guard features; lower predicted error is preferred."""
    return np.asarray(
        [
            float(metrics.get("boundary_color_l1", 0.0)) / 255.0,
            float(metrics.get("boundary_edge_l1", 0.0)) / 255.0,
            float(metrics.get("color_mean_delta", 0.0)) / 255.0,
            float(metrics.get("color_std_delta", 0.0)) / 255.0,
            abs(float(metrics.get("sharpness_gain", 0.0))) / max(float(metrics.get("input_sharpness", 1.0)), 1.0),
            float(metrics.get("inner_color_l1", 0.0)) / 255.0,
        ],
        dtype=np.float64,
    )


def fit_guard_model(train_rows: list[dict]) -> dict:
    """Fit a tiny linear no-reference selector using proxy-train GT only."""
    x, y = [], []
    for row in train_rows:
        if row.get("method") == "B0_identity":
            continue
        target = row.get("masked_l1")
        if target in (None, "") or not np.isfinite(float(target)):
            continue
        x.append(np.r_[1.0, feature_vector(row)])
        y.append(float(target) / 255.0)
    if len(x) < 3:
        return {"intercept": 0.0, "weights": [0.0] * 6, "trained_rows": len(x), "fallback": True}
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    ridge = 1e-3 * np.eye(x_arr.shape[1])
    ridge[0, 0] = 0.0
    coef = np.linalg.solve(x_arr.T @ x_arr + ridge, x_arr.T @ y_arr)
    return {"intercept": float(coef[0]), "weights": [float(v) for v in coef[1:]], "trained_rows": len(x), "fallback": False}


def guard_score(metrics: dict[str, float], model: dict) -> float:
    return float(model.get("intercept", 0.0) + np.dot(np.asarray(model.get("weights", [0.0] * 6)), feature_vector(metrics)))


def fit_tile(image: Image.Image, width: int, height: int, background: str = "white") -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (width, height), background)
    fitted = ImageOps.contain(image, (width - 8, height - 8))
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40), opacity=0.45) -> np.ndarray:
    image = rgb.astype(np.float32).copy()
    hard = np.asarray(mask) > 0
    image[hard] = (1.0 - opacity) * image[hard] + opacity * np.asarray(color, dtype=np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def crop_zoom(rgb: np.ndarray, mask: np.ndarray, size: int = 192, margin: int = 24) -> np.ndarray:
    ys, xs = np.where(np.asarray(mask) > 0)
    if ys.size == 0:
        h, w = rgb.shape[:2]
        crop = rgb[max(0, h // 2 - size // 2):h // 2 + size // 2, max(0, w // 2 - size // 2):w // 2 + size // 2]
    else:
        x0, x1 = max(0, int(xs.min()) - margin), min(rgb.shape[1], int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(rgb.shape[0], int(ys.max()) + margin + 1)
        crop = rgb[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(crop, mode="RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8)


def grid(images: list[tuple[str, np.ndarray]], path: Path, columns: int = 5, tile_size: tuple[int, int] = (220, 190)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tw, th = tile_size
    label_h = 28
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * tw, rows * (th + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (label, array) in enumerate(images):
        x, y = (i % columns) * tw, (i // columns) * (th + label_h)
        canvas.paste(fit_tile(Image.fromarray(array), tw, th), (x, y))
        draw.text((x + 5, y + th + 5), str(label)[:36], fill="black", font=font)
    canvas.save(path)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
