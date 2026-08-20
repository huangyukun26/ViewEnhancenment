#!/usr/bin/env python3
"""Same-building reference retrieval and SIFT/RANSAC alignment.

The local checkout contains the annotation image set but not a separately
exported multi-view source directory.  Therefore this script searches all
available Open3DHK annotation images with the same ``source_group``.  It uses
SIFT as the available feature fallback for both ranking and alignment and
records the resulting coverage instead of claiming DINOv2 retrieval.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))
from common import (  # noqa: E402
    load_mask,
    load_rgb,
    make_grid,
    read_csv,
    save_rgb,
    soft_composite,
    write_csv,
)


OUT_ROOT = ROOT / "research" / "outputs" / "restoration_v2"
IMAGE_ROOT = ROOT / "research" / "assets" / "distortion_segmentation_annotation_dataset" / "for_segmentation" / "images"


def source_group(stem: str) -> str:
    match = re.search(r"_([A-Z]{10})_", stem)
    return match.group(1) if match else "unknown"


def _resize_for_matching(rgb: np.ndarray, side: int = 512) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, side / max(h, w))
    if scale == 1.0:
        return rgb, 1.0
    return cv2.resize(rgb, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA), scale


def _sift_alignment(reference: np.ndarray, target: np.ndarray) -> dict:
    ref_small, ref_scale = _resize_for_matching(reference)
    target_small, target_scale = _resize_for_matching(target)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.02)
    ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)
    target_kp, target_desc = sift.detectAndCompute(target_gray, None)
    if ref_desc is None or target_desc is None or len(ref_kp) < 4 or len(target_kp) < 4:
        return {"ok": False, "reason": "insufficient_sift_features", "matches": 0}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(ref_desc, target_desc, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    if len(good) < 4:
        return {"ok": False, "reason": "few_ratio_test_matches", "matches": len(good)}
    src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        return {"ok": False, "reason": "homography_failed", "matches": len(good)}
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = float(inlier_count / max(1, len(good)))
    ref_h, ref_w = reference.shape[:2]
    target_h, target_w = target.shape[:2]
    # H is estimated in resized coordinates; convert it back to native pixels.
    scale_ref = np.diag([1.0 / ref_scale, 1.0 / ref_scale, 1.0])
    scale_target = np.diag([target_scale, target_scale, 1.0])
    homography_native = scale_target @ homography @ scale_ref
    projected = cv2.perspectiveTransform(np.float32([[[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]]), homography_native)[0]
    valid_polygon = projected.copy()
    valid_polygon[:, 0] = np.clip(valid_polygon[:, 0], 0, target_w - 1)
    valid_polygon[:, 1] = np.clip(valid_polygon[:, 1], 0, target_h - 1)
    coverage = float(abs(cv2.contourArea(valid_polygon)) / max(1.0, target_w * target_h))
    warped = cv2.warpPerspective(reference, homography_native, (target_w, target_h), flags=cv2.INTER_LINEAR)
    support = cv2.warpPerspective(np.ones((ref_h, ref_w), np.uint8), homography_native, (target_w, target_h)) > 0
    return {
        "ok": bool(inlier_count >= 8 and inlier_ratio >= 0.2 and coverage >= 0.03),
        "reason": "ok" if inlier_count >= 8 and inlier_ratio >= 0.2 and coverage >= 0.03 else "weak_ransac",
        "matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "coverage": coverage,
        "homography": homography_native,
        "warped": warped,
        "support": support,
        "keypoints_ref": ref_kp,
        "keypoints_target": target_kp,
        "good_matches": good,
        "inlier_mask": inlier_mask,
    }


def _lab_match(warped: np.ndarray, target: np.ndarray, support: np.ndarray) -> np.ndarray:
    if np.count_nonzero(support) < 32:
        return warped
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    warped_lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    result = warped_lab.copy()
    for channel in range(3):
        ref_values = warped_lab[..., channel][support]
        target_values = target_lab[..., channel][support]
        ref_std = float(ref_values.std())
        target_std = float(target_values.std())
        result[..., channel] = (warped_lab[..., channel] - float(ref_values.mean())) * (target_std / max(ref_std, 1.0)) + float(target_values.mean())
    return cv2.cvtColor(result.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _match_visual(reference: np.ndarray, target: np.ndarray, info: dict) -> np.ndarray:
    if not info.get("good_matches"):
        return target.copy()
    ref_small, _ = _resize_for_matching(reference)
    target_small, _ = _resize_for_matching(target)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY)
    return cv2.drawMatches(
        cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB), info["keypoints_ref"],
        cv2.cvtColor(target_gray, cv2.COLOR_GRAY2RGB), info["keypoints_target"],
        info["good_matches"], None,
        matchesMask=info["inlier_mask"].ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def run(args) -> None:
    showcase = read_csv(ROOT / "research" / "data" / "open3dhk_showcase.csv")
    all_images = sorted(IMAGE_ROOT.glob("*.png"))
    by_group: dict[str, list[Path]] = {}
    for path in all_images:
        by_group.setdefault(source_group(path.stem), []).append(path)
    out_dir = OUT_ROOT / "reference_results"
    rows = []
    grid_tiles = []
    grid_samples = 0
    for row in showcase:
        target_path = ROOT / row["image_path"]
        target = load_rgb(target_path)
        mask = load_mask(ROOT / row["annotation_mask_path"])
        if args.only_large and float(row["manual_mask_ratio"]) < 0.6:
            continue
        candidates = [p for p in by_group.get(row["source_group"], []) if p.resolve() != target_path.resolve()]
        ranked = []
        for candidate_path in candidates:
            reference = load_rgb(candidate_path)
            info = _sift_alignment(reference, target)
            ranked.append((float(info.get("inliers", 0)) * max(0.01, float(info.get("inlier_ratio", 0))), candidate_path, reference, info))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best = ranked[0] if ranked else None
        sample_dir = out_dir / str(row["showcase_index"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        if best is None:
            ref_path = ""
            warped = target.copy()
            composite = target.copy()
            matches_visual = target.copy()
            info = {"ok": False, "reason": "no_same_group_candidate", "matches": 0, "inliers": 0, "inlier_ratio": 0, "coverage": 0}
            reference = target.copy()
        elif "warped" not in best[3]:
            _, ref_path_obj, reference, info = best
            ref_path = str(ref_path_obj.relative_to(ROOT))
            # A weak descriptor match is still recorded, but cannot produce
            # a warp; abstain rather than fabricating an alignment.
            warped = target.copy()
            composite = target.copy()
            matches_visual = target.copy()
            info = dict(info)
            info["ok"] = False
            info["reason"] = info.get("reason", "alignment_not_available")
        else:
            _, ref_path_obj, reference, info = best
            ref_path = str(ref_path_obj.relative_to(ROOT))
            warped = info["warped"]
            adjusted = _lab_match(warped, target, info["support"])
            composite, _ = soft_composite(target, adjusted, mask, dilate_px=3, feather=1.5)
            matches_visual = _match_visual(reference, target, info)
        save_rgb(sample_dir / "reference.png", reference)
        save_rgb(sample_dir / "matches.png", matches_visual)
        save_rgb(sample_dir / "warped.png", warped)
        save_rgb(sample_dir / "composite.png", composite)
        save_rgb(sample_dir / "target.png", target)
        save_rgb(sample_dir / "mask_rgb.png", np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2))
        rows.append({
            "sample_id": row["sample_id"],
            "showcase_index": row["showcase_index"],
            "source_group": row["source_group"],
            "target_path": str(target_path.relative_to(ROOT)),
            "reference_path": ref_path,
            "candidate_count": str(len(candidates)),
            "reliable": str(bool(info.get("ok", False))),
            "reason": info.get("reason", ""),
            "matches": str(info.get("matches", 0)),
            "inliers": str(info.get("inliers", 0)),
            "inlier_ratio": f"{float(info.get('inlier_ratio', 0)):.6f}",
            "coverage": f"{float(info.get('coverage', 0)):.6f}",
            "warped_path": str((sample_dir / "warped.png").relative_to(ROOT)),
            "composite_path": str((sample_dir / "composite.png").relative_to(ROOT)),
        })
        if grid_samples < args.grid_limit:
            grid_tiles.extend([
                (f"{row['showcase_index']} target", target),
                (f"{row['showcase_index']} mask", np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2)),
                (f"{row['showcase_index']} reference", reference),
                (f"{row['showcase_index']} matches", matches_visual),
                (f"{row['showcase_index']} warped", warped),
                (f"{row['showcase_index']} composite", composite),
            ])
            grid_samples += 1
    fields = list(rows[0].keys()) if rows else ["sample_id"]
    write_csv(OUT_ROOT / "reference_coverage.csv", rows, fields)
    make_grid(grid_tiles, OUT_ROOT / "reference_alignment_grid.png", columns=6, tile_size=(250, 220))
    print(f"reference candidates: {len(all_images)}, showcase rows: {len(rows)}, reliable: {sum(r['reliable'] == 'True' for r in rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-large", action="store_true")
    parser.add_argument("--grid-limit", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
