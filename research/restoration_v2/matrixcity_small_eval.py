#!/usr/bin/env python3
"""Small, reproducible MatrixCity reference-view sanity experiment.

This script deliberately works with a locally cached *partial* uncompressed tar
range.  It never downloads or commits the multi-GB MatrixCity archives.  The
clean target is the MatrixCity RGB image itself; an Open3DHK mask shape is used
to synthesize a strictly masked UV/resolution degradation.  A nearby clean
MatrixCity view is then used as the reference for the SIFT+homography baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))
from common import load_rgb, lpips_distance, masked_psnr, masked_ssim, save_rgb  # noqa: E402
from reference_consistency import _lab_adjust, _sift_alignment  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _extract_partial_pngs(archive: Path, output_dir: Path, limit: int) -> list[Path]:
    """Extract complete PNG members from a partial tar range."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with tarfile.open(archive, "r:") as handle:
            for member in handle:
                name = Path(member.name)
                if not member.isfile() or name.suffix.lower() != ".png":
                    continue
                try:
                    frame_id = int(name.stem)
                except ValueError:
                    continue
                if frame_id >= limit:
                    break
                destination = output_dir / f"{frame_id:04d}.png"
                if not destination.exists():
                    source = handle.extractfile(member)
                    if source is None:
                        continue
                    destination.write_bytes(source.read())
                extracted.append(destination)
    except (tarfile.ReadError, EOFError):
        # A range intentionally ends in the middle of a tar member.  All
        # complete members before that point are valid and are retained.
        pass
    return sorted(set(extracted))


def _load_frames(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for frame in data.get("frames", []):
        if "frame_index" in frame and "rot_mat" in frame:
            result[int(frame["frame_index"])] = np.asarray(frame["rot_mat"], dtype=np.float32)
    return result


def _nearest_reference(target_id: int, frame_ids: list[int], poses: dict[int, np.ndarray]) -> int | None:
    candidates = [i for i in frame_ids if i != target_id and i in poses]
    if not candidates:
        return None
    target = poses[target_id][:3, 3]
    return min(candidates, key=lambda i: float(np.linalg.norm(poses[i][:3, 3] - target)))


def _mask_pool() -> list[Path]:
    return sorted((ROOT / "research" / "assets").glob("**/mask_for_sam/*.png"))


def _resize_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    # The annotation PNGs are binary images encoded as either {0,1} or
    # {0,255}; use >0 rather than assuming an 8-bit 255 foreground.
    mask = np.asarray(Image.fromarray(raw).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    # Avoid mostly-white/empty masks while preserving the Open3DHK shape.
    if mask.mean() > 0.55:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((19, 19), np.uint8), iterations=1) > 0
    return mask


def _masked_degradation(clean: np.ndarray, mask: np.ndarray, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    low = cv2.resize(cv2.resize(clean, (max(32, w // 4), max(32, h // 4)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "uv_stretch":
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        shift = 0.18 * w * np.sin(grid_y / max(1.0, h) * math.pi * 2.0)
        map_x = np.mod(grid_x * 0.52 + shift, max(1.0, w - 1)).astype(np.float32)
        map_y = grid_y
        degraded = cv2.remap(clean, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif kind == "repeat_seam":
        degraded = low.copy()
        split = int(w * (0.42 + 0.12 * rng.random()))
        source = np.roll(low[:, : max(1, split)], int(w * 0.08), axis=1)
        repeats = int(np.ceil((w - split) / max(1, source.shape[1])))
        repeated = np.tile(source, (1, repeats, 1))[:, : w - split]
        degraded[:, split:] = repeated
    else:
        degraded = cv2.GaussianBlur(low, (0, 0), 3.2)
    result = clean.copy()
    result[mask] = degraded[mask]
    return result


def _homography_composite(distorted: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    info = _sift_alignment(reference, distorted)
    if info.get("warped") is None:
        return distorted.copy(), info
    support = np.asarray(info["support"], dtype=bool)
    photo_region = support & ~cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1).astype(bool)
    adjusted = _lab_adjust(info["warped"], distorted, photo_region)
    # MatrixCity evaluation uses the exact synthetic mask: pixels outside it
    # are byte-identical, so the outside maximum is an actual hard guarantee.
    output = distorted.copy()
    output[mask] = adjusted[mask]
    info["support_in_mask"] = float((support & mask).sum() / max(1, mask.sum()))
    info["outside_max"] = int(np.abs(output.astype(np.int16) - distorted.astype(np.int16))[~mask].max())
    return output, info


def _load_lpips():
    try:
        import lpips

        return lpips.LPIPS(net="alex").cuda().eval()
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"LPIPS unavailable: {exc}")
        return None


def _small_lpips(metric, output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if metric is None:
        return float("nan")
    ys, xs = np.where(mask)
    if len(xs) < 16:
        return float("nan")
    y0, y1 = max(0, int(ys.min()) - 4), min(output.shape[0], int(ys.max()) + 5)
    x0, x1 = max(0, int(xs.min()) - 4), min(output.shape[1], int(xs.max()) + 5)
    a = cv2.resize(output[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
    b = cv2.resize(target[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_AREA)
    full = np.ones((256, 256), dtype=np.uint8)
    return lpips_distance(metric, a, b, full)


def run(args: argparse.Namespace) -> None:
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    image_dir = ROOT / args.image_dir
    paths = _extract_partial_pngs(ROOT / args.archive, image_dir, args.max_images)
    poses = _load_frames(ROOT / args.pose_json)
    frame_ids = [int(p.stem) for p in paths if int(p.stem) in poses]
    frame_ids.sort()
    masks = _mask_pool()
    if len(frame_ids) < 4 or not masks:
        status = {"status": "blocked", "reason": "insufficient cached RGB frames or Open3DHK masks", "frames": len(frame_ids), "masks": len(masks)}
        (output / "matrixcity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return

    probe_shape = load_rgb(paths[0]).shape[:2]
    usable_masks = []
    for mask_path in masks:
        candidate = _resize_mask(mask_path, probe_shape)
        ratio = float(candidate.mean())
        if 0.015 <= ratio <= 0.45:
            usable_masks.append(mask_path)
    if not usable_masks:
        status = {"status": "blocked", "reason": "no usable Open3DHK mask shape after resize", "frames": len(frame_ids), "masks": len(masks)}
        (output / "matrixcity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return

    rows: list[dict] = []
    methods = {"identity": [], "homography": []}
    lpips_metric = _load_lpips()
    pair_count = min(args.pairs, len(frame_ids) // 2)
    for pair_index in range(pair_count):
        target_id = frame_ids[2 * pair_index]
        reference_id = _nearest_reference(target_id, frame_ids, poses)
        if reference_id is None:
            continue
        target_path = image_dir / f"{target_id:04d}.png"
        reference_path = image_dir / f"{reference_id:04d}.png"
        clean = load_rgb(target_path)
        reference = load_rgb(reference_path)
        mask = _resize_mask(usable_masks[pair_index % len(usable_masks)], clean.shape[:2])
        kind = ("uv_stretch", "repeat_seam", "resolution_blur")[pair_index % 3]
        distorted = _masked_degradation(clean, mask, kind, pair_index)
        homography, info = _homography_composite(distorted, reference, mask)
        for name, candidate in (("identity", distorted), ("homography", homography)):
            region = mask
            delta = np.abs(candidate.astype(np.int16) - distorted.astype(np.int16))
            row = {
                "pair_id": f"matrixcity_down_{pair_index:04d}",
                "target_frame": target_id,
                "reference_frame": reference_id,
                "degradation": kind,
                "method": name,
                "mask_ratio": float(mask.mean()),
                "masked_psnr": masked_psnr(candidate, clean, mask),
                "masked_ssim": masked_ssim(candidate, clean, mask),
                "masked_lpips": _small_lpips(lpips_metric, candidate, clean, mask),
                "mask_outside_max_abs": int(delta[~mask].max()),
                "homography_mask_support_ratio": float(info.get("support_in_mask", 0.0)) if name == "homography" else "",
                "matches": int(info.get("matches", 0)) if name == "homography" else "",
                "inliers": int(info.get("inliers", 0)) if name == "homography" else "",
            }
            rows.append(row)
            methods[name].append(row)
        if pair_index < args.save_examples:
            stem = output / "examples" / f"{pair_index:03d}"
            stem.mkdir(parents=True, exist_ok=True)
            save_rgb(stem / "clean_gt.png", clean)
            save_rgb(stem / "distorted_input.png", distorted)
            save_rgb(stem / "reference.png", reference)
            save_rgb(stem / "mask.png", np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2))
            save_rgb(stem / "homography.png", homography)

    _write_csv(output / "matrixcity_metrics.csv", rows)
    summary = {"status": "completed", "pairs": len(methods["identity"]), "source": "MatrixCity small_city street train_dense / road_down", "mask_source": "Open3DHK annotation masks", "methods": {}}
    for name, values in methods.items():
        if not values:
            continue
        for field in ("masked_psnr", "masked_ssim", "masked_lpips"):
            nums = np.asarray([float(v[field]) for v in values], dtype=float)
            summary["methods"].setdefault(name, {})[field + "_mean"] = float(nums.mean())
            summary["methods"][name][field + "_median"] = float(np.median(nums))
    (output / "matrixcity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=".research_cache/matrixcity/down_dense_256mb.bin")
    parser.add_argument("--pose-json", default=".research_cache/matrixcity/down_transforms.json")
    parser.add_argument("--image-dir", default=".research_cache/matrixcity/images")
    parser.add_argument("--output", default="research/outputs/restoration_v2/matrixcity_small")
    parser.add_argument("--max-images", type=int, default=180)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--save-examples", type=int, default=12)
    run(parser.parse_args())
