#!/usr/bin/env python3
"""Build strictly paired, domain-proxy Open3DHK restoration data."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

from common import load_mask, load_rgb, make_grid, repo_root, save_mask, save_rgb, strict_proxy_composite, write_csv


DEGRADATIONS = ("blur_downsample", "smear_warp", "repeat_missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    return parser.parse_args()


def source_group(stem: str) -> str:
    parts = stem.split("_")
    for part in parts:
        if len(part) == 10 and part.isalpha() and part.isupper():
            return part
    return parts[3] if len(parts) > 3 else stem


def patch_quality(image: np.ndarray, available: np.ndarray, x: int, y: int, size: int) -> tuple[bool, float]:
    region = image[y:y + size, x:x + size]
    valid = available[y:y + size, x:x + size]
    if region.shape[0] != size or region.shape[1] != size or valid.mean() < 0.985:
        return False, 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    nonwhite = (gray < 245).mean()
    texture_std = float(gray.std())
    edge_mean = float(cv2.Canny(region, 60, 140).mean())
    score = min(1.0, nonwhite) * min(1.0, texture_std / 35.0) * min(1.0, edge_mean / 12.0)
    return bool(nonwhite > 0.82 and texture_std > 12.0 and edge_mean > 2.5), score


def make_mask(size: int, severity: str, rng: random.Random) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if severity == "small":
        h = rng.randint(max(24, size // 8), max(30, size // 4))
        w = rng.randint(max(24, size // 8), max(30, size // 4))
    elif severity == "medium":
        h = rng.randint(size // 4, size // 2)
        w = rng.randint(size // 4, size // 2)
    else:
        h = rng.randint(size // 2, int(size * 0.78))
        w = rng.randint(size // 2, int(size * 0.78))
    x = rng.randint(6, max(6, size - w - 6))
    y = rng.randint(6, max(6, size - h - 6))
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    if rng.random() < 0.55:
        # A notch keeps masks closer to the irregular Open3DHK annotation shapes.
        nx = min(size - 1, x + rng.randint(0, max(1, w // 2)))
        ny = min(size - 1, y + rng.randint(0, max(1, h // 2)))
        cv2.ellipse(mask, (nx, ny), (max(5, w // 4), max(5, h // 5)), rng.randint(0, 180), 0, 360, 0, -1)
    return mask > 0


def degrade(clean: np.ndarray, mask: np.ndarray, degradation: str, rng: random.Random) -> np.ndarray:
    degraded = clean.copy()
    if degradation == "blur_downsample":
        sigma = rng.uniform(1.4, 3.2)
        blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=sigma)
        scale = rng.choice((0.35, 0.5, 0.65))
        small = cv2.resize(blurred, (max(8, int(clean.shape[1] * scale)), max(8, int(clean.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_LINEAR)
    elif degradation == "smear_warp":
        h, w = clean.shape[:2]
        dx = rng.uniform(-0.12, 0.12) * w
        dy = rng.uniform(-0.08, 0.08) * h
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst = np.float32([[dx, dy], [w - 1 + dx, dy * 0.4], [dx * 0.3, h - 1 + dy]])
        matrix = cv2.getAffineTransform(src, dst)
        degraded = cv2.warpAffine(clean, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)
        degraded = cv2.GaussianBlur(degraded, (0, 0), rng.uniform(0.4, 1.2))
    else:
        h, w = clean.shape[:2]
        mode = rng.choice(("repeat", "missing"))
        if mode == "repeat":
            strip_w = max(8, w // 12)
            source_x = rng.randint(0, max(0, w - strip_w))
            strip = clean[:, source_x:source_x + strip_w]
            tiled = np.tile(strip, (1, int(np.ceil(w / strip_w)), 1))[:, :w]
            degraded = tiled
        else:
            blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=14)
            degraded = blurred
    return strict_proxy_composite(clean, degraded, mask)


def choose_patch(image: np.ndarray, annotation_mask: np.ndarray, rng: random.Random, size: int = 224) -> tuple[np.ndarray, int, int, float] | None:
    available = ~cv2.dilate(annotation_mask.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    h, w = image.shape[:2]
    candidates = []
    for _ in range(120):
        x = rng.randint(0, max(0, w - size))
        y = rng.randint(0, max(0, h - size))
        ok, score = patch_quality(image, available, x, y, size)
        if ok:
            candidates.append((score, x, y))
    if not candidates:
        return None
    score, x, y = max(candidates)
    return image[y:y + size, x:x + size].copy(), x, y, score


def split_for_group(group: str) -> str:
    digest = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "val" if digest < 2 else "train"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    rng = random.Random(args.seed)
    image_dir = root / "research/assets/distortion_segmentation_annotation_dataset/for_segmentation/images"
    mask_dir = root / "research/assets/distortion_segmentation_annotation_dataset/for_segmentation/mask_for_sam"
    data_root = root / "research/data/restoration_v2_proxy_pairs"
    output_root = root / "research/outputs/restoration_v2"
    data_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.png"))
    rng.shuffle(image_paths)
    rows: list[dict] = []
    tries = 0
    target_each = max(1, args.count // len(DEGRADATIONS))
    counts = {name: 0 for name in DEGRADATIONS}
    while len(rows) < args.count and tries < args.count * 80:
        tries += 1
        degradation = DEGRADATIONS[len(rows) % len(DEGRADATIONS)]
        if counts[degradation] >= target_each and len(rows) + (args.count - len(rows)) > args.count - (args.count % len(DEGRADATIONS)):
            degradation = min(DEGRADATIONS, key=lambda name: counts[name])
        image_path = image_paths[tries % len(image_paths)]
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        image = load_rgb(image_path)
        annotation_mask = load_mask(mask_path)
        selected = choose_patch(image, annotation_mask, rng)
        if selected is None:
            continue
        clean, x, y, quality = selected
        severity = rng.choices(("small", "medium", "large"), weights=(0.4, 0.4, 0.2))[0]
        patch_mask = make_mask(clean.shape[0], severity, rng)
        if patch_mask.mean() < 0.04 or patch_mask.mean() > 0.72:
            continue
        distorted = degrade(clean, patch_mask, degradation, rng)
        diff = np.abs(distorted.astype(np.int16) - clean.astype(np.int16)).mean(axis=2)
        masked_diff = float(diff[patch_mask].mean())
        if masked_diff < 5.0 or float(diff[~patch_mask].mean()) > 0.01:
            continue
        sample_id = f"v2_{len(rows):04d}_{image_path.stem}_{degradation}"
        clean_rel = Path("research/data/restoration_v2_proxy_pairs") / sample_id / "clean.png"
        distorted_rel = Path("research/data/restoration_v2_proxy_pairs") / sample_id / "distorted.png"
        mask_rel = Path("research/data/restoration_v2_proxy_pairs") / sample_id / "mask.png"
        save_rgb(root / clean_rel, clean)
        save_rgb(root / distorted_rel, distorted)
        save_mask(root / mask_rel, patch_mask)
        rows.append({
            "sample_id": sample_id,
            "source_group": source_group(image_path.stem),
            "source_image": str(image_path.relative_to(root)).replace("\\", "/"),
            "degradation_type": degradation,
            "severity": severity,
            "clean_path": str(clean_rel).replace("\\", "/"),
            "distorted_path": str(distorted_rel).replace("\\", "/"),
            "mask_path": str(mask_rel).replace("\\", "/"),
            "mask_ratio": f"{patch_mask.mean():.8f}",
            "clean_quality_score": f"{quality:.6f}",
            "masked_input_clean_l1": f"{masked_diff:.6f}",
            "split": split_for_group(source_group(image_path.stem)),
            "seed": args.seed + len(rows),
        })
        counts[degradation] += 1
    if len(rows) < args.count:
        raise RuntimeError(f"could only construct {len(rows)} strict pairs out of requested {args.count}")
    fields = list(rows[0].keys())
    write_csv(root / "research/data/restoration_v2_proxy_pairs.csv", rows, fields)
    tiles: list[tuple[str, np.ndarray]] = []
    for row in rows[:30]:
        clean = load_rgb(root / row["clean_path"])
        distorted = load_rgb(root / row["distorted_path"])
        mask = load_mask(root / row["mask_path"])
        tiles.extend([(f"{row['sample_id'][-12:]} input", distorted), ("mask", np.repeat(mask[..., None] * 255, 3, axis=2)), ("clean", clean)])
    make_grid(tiles, output_root / f"proxy_inspection_{min(30, len(rows))}.png", columns=6, tile_size=(210, 190))
    print({"count": len(rows), "degradation_counts": counts, "split_counts": {split: sum(r["split"] == split for r in rows) for split in ("train", "val")}, "output": str(root / "research/data/restoration_v2_proxy_pairs.csv")})


if __name__ == "__main__":
    main()
