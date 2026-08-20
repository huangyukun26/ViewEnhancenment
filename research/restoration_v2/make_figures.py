#!/usr/bin/env python3
"""Make fixed-showcase, zoom, and failure-case figures without cherry-picking."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))
from common import load_mask, load_rgb, make_grid, read_csv  # noqa: E402


OUT_ROOT = ROOT / "research" / "outputs" / "restoration_v2"


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def _placeholder(rgb: np.ndarray) -> np.ndarray:
    return np.full_like(rgb, 128, dtype=np.uint8)


def _zoom(rgb: np.ndarray, mask: np.ndarray, size: int = 240) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        crop = rgb
    else:
        pad = max(20, int(round(max(rgb.shape[:2]) * 0.035)))
        x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
        y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
        crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    return cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _get_images(row: dict[str, str], backend_map: dict, refs: dict) -> tuple[dict[str, np.ndarray], str]:
    index = row["showcase_index"]
    input_rgb = load_rgb(_path(row["image_path"]))
    mask = load_mask(_path(row["annotation_mask_path"]))
    entry = backend_map.get((index, "manual"), {})
    images = {"input": input_rgb, "mask": _mask_rgb(mask)}
    for label, key in (("A", "R1_NAFNet_path"), ("B", "R2_LaMa_path")):
        value = entry.get(key, "")
        images[label] = load_rgb(_path(value)) if value and _path(value).exists() else _placeholder(input_rgb)
    reference_row = refs.get(index, {})
    ref_value = reference_row.get("composite_path", "")
    images["C"] = load_rgb(_path(ref_value)) if reference_row.get("reliable") == "True" and ref_value and _path(ref_value).exists() else _placeholder(input_rgb)
    severity = row["severity"]
    if severity in {"small", "medium"} and entry.get("R1_NAFNet_path") and _path(entry["R1_NAFNet_path"]).exists():
        route = "A"
    elif severity == "large" and reference_row.get("reliable") == "True" and ref_value and _path(ref_value).exists():
        route = "C"
    elif entry.get("R2_LaMa_path") and _path(entry["R2_LaMa_path"]).exists():
        route = "B"
    else:
        route = "abstain"
    images["final"] = images[route] if route != "abstain" else input_rgb
    return images, route


def main() -> None:
    showcase = read_csv(ROOT / "research" / "data" / "open3dhk_showcase.csv")
    backend_rows = read_csv(OUT_ROOT / "real_backend_manifest_showcase.csv") if (OUT_ROOT / "real_backend_manifest_showcase.csv").exists() else []
    backend_map = {(r.get("showcase_index", ""), r.get("mask_source", "")): r for r in backend_rows}
    refs = {r["showcase_index"]: r for r in read_csv(OUT_ROOT / "reference_coverage.csv")} if (OUT_ROOT / "reference_coverage.csv").exists() else {}
    full_tiles, zoom_tiles, failures = [], [], []
    for row in showcase:
        images, route = _get_images(row, backend_map, refs)
        index = row["showcase_index"]
        mask = load_mask(_path(row["annotation_mask_path"]))
        full_tiles.extend([(f"{index} input", images["input"]), (f"{index} mask", images["mask"]), (f"{index} A", images["A"]), (f"{index} B", images["B"]), (f"{index} C", images["C"]), (f"{index} final", images["final"])])
        zoom_tiles.extend([(f"{index} input", _zoom(images["input"], mask)), (f"{index} mask", _zoom(images["mask"], mask)), (f"{index} A", _zoom(images["A"], mask)), (f"{index} B", _zoom(images["B"], mask)), (f"{index} C", _zoom(images["C"], mask)), (f"{index} final", _zoom(images["final"], mask))])
        sam_iou = float(row.get("manual_sam_iou", "0"))
        ref_bad = refs.get(index, {}).get("reliable") != "True"
        if route == "abstain" or sam_iou < 0.8 or ref_bad or not backend_map.get((index, "manual"), {}).get("R2_LaMa_path"):
            failures.extend([(f"{index} input", images["input"]), (f"{index} mask", images["mask"]), (f"{index} A", images["A"]), (f"{index} B", images["B"]), (f"{index} C", images["C"]), (f"{index} final", images["final"])])
    make_grid(full_tiles, OUT_ROOT / "open3dhk_full_grid.png", columns=6, tile_size=(220, 190))
    make_grid(zoom_tiles, OUT_ROOT / "open3dhk_zoom_grid.png", columns=6, tile_size=(220, 220))
    make_grid(failures[: 6 * 8], OUT_ROOT / "failure_cases.png", columns=6, tile_size=(220, 210))
    print(f"full showcase samples: {len(showcase)}, failure-case tiles: {min(len(failures), 48)}")


if __name__ == "__main__":
    main()
