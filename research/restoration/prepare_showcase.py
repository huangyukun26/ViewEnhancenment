#!/usr/bin/env python3
"""Create a fixed 24-image Open3DHK showcase from the supplied annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import load_rgb, repo_root, relpath, stable_hash, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--count", type=int, default=24)
    return parser.parse_args()


def annotation_root(root: Path) -> Path:
    return root / "research" / "assets" / "distortion_segmentation_annotation_dataset" / "for_segmentation"


def source_group(name: str) -> str:
    tokens = Path(name).stem.split("_")
    if len(tokens) >= 3 and tokens[0].isdigit() and tokens[1].isdigit():
        return tokens[2]
    return "_".join(tokens[:3])


def room_category(name: str) -> str:
    tokens = Path(name).stem.split("_")
    for token in ("Living", "Kitchen", "Bed", "MBed", "Bath", "Balcony", "Utility", "Store", "Null"):
        if token in tokens:
            return token
    return "other"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    ann = annotation_root(root)
    pairs = []
    for image_path in sorted((ann / "images").glob("*.png")):
        mask_path = ann / "mask_for_sam" / image_path.name
        if not mask_path.exists():
            continue
        image = load_rgb(image_path)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        if image.shape[:2] != mask.shape:
            continue
        pairs.append(
            {
                "image_path": relpath(image_path, root),
                "annotation_mask_path": relpath(mask_path, root),
                "source_group": source_group(image_path.name),
                "category": room_category(image_path.name),
                "manual_mask_ratio": float(mask.mean()),
            }
        )
    if not pairs:
        raise RuntimeError("No image/mask pairs found in the annotation asset")

    ratios = np.asarray([row["manual_mask_ratio"] for row in pairs], dtype=np.float64)
    q1, q2 = np.quantile(ratios, [0.33, 0.66])
    for row in pairs:
        ratio = row["manual_mask_ratio"]
        row["severity"] = "small" if ratio <= q1 else "medium" if ratio <= q2 else "large"

    selected: list[dict] = []
    used_groups: set[str] = set()
    each = max(1, args.count // 3)
    for severity in ("small", "medium", "large"):
        candidates = [row for row in pairs if row["severity"] == severity]
        candidates.sort(key=lambda row: stable_hash(f"showcase:{severity}:{row['source_group']}:{row['image_path']}"))
        for row in candidates:
            if row["source_group"] in used_groups:
                continue
            selected.append(row)
            used_groups.add(row["source_group"])
            if sum(item["severity"] == severity for item in selected) >= each:
                break
    if len(selected) < args.count:
        for row in sorted(pairs, key=lambda item: stable_hash(f"fill:{item['image_path']}")):
            if row not in selected:
                selected.append(row)
            if len(selected) >= args.count:
                break
    selected = selected[: args.count]
    selected.sort(key=lambda row: (row["severity"], stable_hash(row["image_path"])))
    for index, row in enumerate(selected):
        row["sample_id"] = f"showcase_{index:02d}_{Path(row['image_path']).stem}"
        row["showcase_index"] = index
        row["sam_mask_path"] = ""
        row["sam_mask_ratio"] = ""
        row["sam_model_iou_score"] = ""
        row["mask_source"] = "manual_annotation_for_fixed_selection"
        row["previous_generated_path"] = ""

    output = root / "research" / "data" / "open3dhk_showcase.csv"
    fields = [
        "sample_id", "showcase_index", "source_group", "category", "severity",
        "manual_mask_ratio", "image_path", "annotation_mask_path",
        "sam_mask_path", "sam_mask_ratio", "sam_model_iou_score", "mask_source",
        "previous_generated_path",
    ]
    write_csv(output, selected, fields)
    counts = {severity: sum(row["severity"] == severity for row in selected) for severity in ("small", "medium", "large")}
    print(f"wrote {output} with {len(selected)} fixed samples; severity_counts={counts}")


if __name__ == "__main__":
    main()
