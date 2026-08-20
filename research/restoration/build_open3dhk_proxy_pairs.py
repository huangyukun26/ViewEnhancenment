#!/usr/bin/env python3
"""Build a small, strictly paired Open3DHK-domain restoration proxy set.

The clean patch is cropped from a visually usable region outside the supplied
distortion mask. The clean patch is only a domain proxy, not a real clean GT.
Each degradation is applied inside an exact synthetic mask, and all outputs are
written under research/ so source data remain read-only.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import load_rgb, repo_root, relpath, save_mask, save_rgb, stable_hash, write_csv


KINDS = ("blur", "downsample", "smear", "repeat", "missing", "warp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def annotation_root(root: Path) -> Path:
    return root / "research" / "assets" / "distortion_segmentation_annotation_dataset" / "for_segmentation"


def source_group(name: str) -> str:
    tokens = Path(name).stem.split("_")
    if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isdigit():
        return tokens[2]
    return "_".join(tokens[:3])


def choose_crop(rgb: np.ndarray, existing_mask: np.ndarray, rng: random.Random, size: int = 256) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for _ in range(80):
        x0 = rng.randint(0, max(0, w - size))
        y0 = rng.randint(0, max(0, h - size))
        crop = rgb[y0:y0 + size, x0:x0 + size]
        mask_crop = existing_mask[y0:y0 + size, x0:x0 + size]
        gray = crop.astype(np.float32).mean(axis=2)
        nonwhite = float((gray < 245).mean())
        texture = float(gray.std())
        occupied = float(mask_crop.mean())
        # Prefer non-white facade/scene regions with little pre-existing mask.
        score = occupied * 3.0 + max(0.0, 0.45 - nonwhite) + max(0.0, 5.0 - texture) / 20.0
        candidates.append((score, (x0, y0, x0 + size, y0 + size)))
        if occupied < 0.03 and nonwhite > 0.35 and texture > 6:
            return crop.copy(), (x0, y0, x0 + size, y0 + size)
    candidates.sort(key=lambda item: item[0])
    box = candidates[0][1]
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1].copy(), box


def synthetic_mask(size: int, kind: str, rng: random.Random) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    w = rng.randint(max(24, size // 8), max(32, size // 3))
    h = rng.randint(max(24, size // 8), max(32, size // 3))
    x0 = rng.randint(8, max(8, size - w - 8))
    y0 = rng.randint(8, max(8, size - h - 8))
    if kind in {"smear", "warp"}:
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=255)
        if rng.random() < 0.5:
            draw.rectangle((max(0, x0 - w // 3), y0 + h // 3, min(size - 1, x0 + w + w // 3), y0 + 2 * h // 3), fill=255)
    elif kind == "repeat":
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=255)
        draw.rectangle((x0 + w // 3, max(0, y0 - h // 5), x0 + 2 * w // 3, min(size - 1, y0 + h + h // 5)), fill=255)
    else:
        draw.ellipse((x0, y0, x0 + w, y0 + h), fill=255)
    return np.asarray(image) > 0


def region_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def degrade(clean: np.ndarray, mask: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    source = Image.fromarray(clean, mode="RGB")
    x0, y0, x1, y1 = region_bbox(mask)
    margin = 8
    bx0, by0, bx1, by1 = max(0, x0 - margin), max(0, y0 - margin), min(clean.shape[1], x1 + margin), min(clean.shape[0], y1 + margin)
    region = source.crop((bx0, by0, bx1, by1))
    rw, rh = region.size

    if kind == "blur":
        altered = region.filter(ImageFilter.GaussianBlur(radius=rng.choice([1.5, 2.5, 3.5])))
    elif kind == "downsample":
        small = region.resize((max(8, rw // 3), max(8, rh // 3)), Image.Resampling.BILINEAR)
        altered = small.resize((rw, rh), Image.Resampling.BILINEAR)
    elif kind == "smear":
        stretched = region.resize((max(8, int(rw * rng.choice([1.35, 1.6]))), rh), Image.Resampling.BILINEAR)
        left = max(0, (stretched.width - rw) // 2)
        altered = stretched.crop((left, 0, left + rw, rh))
    elif kind == "warp":
        shift = rng.choice([-12, -8, 8, 12])
        altered = region.transform(region.size, Image.Transform.AFFINE, (1.0, 0.0, shift, 0.0, 1.0, -shift / 2), resample=Image.Resampling.BILINEAR)
    elif kind == "repeat":
        altered = region.copy()
        strip_w = max(4, rw // 4)
        strip = region.crop((max(0, rw // 2 - strip_w // 2), 0, min(rw, rw // 2 + strip_w // 2), rh))
        for x in range(0, rw, strip.width):
            altered.paste(strip.resize((min(strip.width, rw - x), rh), Image.Resampling.BILINEAR), (x, 0))
    else:  # missing / local occlusion
        border = np.asarray(region, dtype=np.float32)
        border_pixels = np.concatenate([border[: max(2, rh // 8)].reshape(-1, 3), border[-max(2, rh // 8):].reshape(-1, 3)], axis=0)
        color = np.clip(border_pixels.mean(axis=0), 0, 255).astype(np.uint8)
        altered = Image.new("RGB", (rw, rh), tuple(int(v) for v in color))
        altered = altered.filter(ImageFilter.GaussianBlur(5.0))

    full = source.copy()
    full.paste(altered, (bx0, by0))
    return np.asarray(full, dtype=np.uint8)


def select_sources(root: Path, count: int, seed: int) -> list[Path]:
    ann = annotation_root(root)
    candidates = sorted((ann / "images").glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[Path] = []
    groups: set[str] = set()
    # First pass favors different building codes; second pass fills the quota.
    for path in candidates:
        group = source_group(path.name)
        if group not in groups:
            selected.append(path)
            groups.add(group)
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected.extend([p for p in candidates if p not in selected][: count - len(selected)])
    return selected[:count]


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    ann = annotation_root(root)
    output = root / "research" / "data" / "open3dhk_proxy_pairs"
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records: list[dict] = []
    sources = select_sources(root, args.count, args.seed)
    for index, image_path in enumerate(sources):
        mask_path = ann / "mask_for_sam" / image_path.name
        if not mask_path.exists():
            continue
        rgb = load_rgb(image_path)
        existing_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        clean, crop_box = choose_crop(rgb, existing_mask, rng)
        kind = KINDS[index % len(KINDS)]
        syn_mask = synthetic_mask(clean.shape[0], kind, rng)
        distorted = degrade(clean, syn_mask, kind, rng)
        sample_id = f"proxy_{index:03d}_{image_path.stem}"
        sample_dir = output / sample_id
        save_rgb(sample_dir / "clean.png", clean)
        save_rgb(sample_dir / "distorted.png", distorted)
        save_mask(sample_dir / "mask.png", syn_mask)
        record = {
            "sample_id": sample_id,
            "source_image": relpath(image_path, root),
            "source_mask": relpath(mask_path, root),
            "source_group": source_group(image_path.name),
            "split": "pending",
            "kind": kind,
            "crop_x0": crop_box[0],
            "crop_y0": crop_box[1],
            "crop_x1": crop_box[2],
            "crop_y1": crop_box[3],
            "clean_path": relpath(sample_dir / "clean.png", root),
            "distorted_path": relpath(sample_dir / "distorted.png", root),
            "mask_path": relpath(sample_dir / "mask.png", root),
            "mask_ratio": float(syn_mask.mean()),
            "seed": args.seed + index,
        }
        records.append(record)

    # Group-level split: the same source building never appears on both sides.
    groups = sorted({r["source_group"] for r in records}, key=lambda x: stable_hash(f"{args.seed}:{x}"))
    n_val = max(1, int(round(len(groups) * 0.2))) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])
    for record in records:
        record["split"] = "val" if record["source_group"] in val_groups else "train"

    fields = list(records[0].keys()) if records else ["sample_id"]
    write_csv(root / "research" / "data" / "open3dhk_proxy_pairs.csv", records, fields)
    summary = {
        "count": len(records),
        "train": sum(r["split"] == "train" for r in records),
        "val": sum(r["split"] == "val" for r in records),
        "source_groups": len(groups),
        "kinds": {kind: sum(r["kind"] == kind for r in records) for kind in KINDS},
        "note": "Clean is a crop from a visually usable Open3DHK region outside the supplied mask; it is a domain proxy, not a real clean GT.",
    }
    (output / "summary.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
