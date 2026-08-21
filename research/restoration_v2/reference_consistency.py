#!/usr/bin/env python3
"""Conservative reference-view evaluation for the next restoration cycle.

The output of this script is deliberately separate from the previous
restoration-v2 outputs.  It fixes the resized-coordinate homography transform,
keeps the top five same-building candidates, and only composites pixels where
the reference has valid support and sufficient correspondence confidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))
from common import dilate, load_mask, load_rgb, make_grid, read_csv, save_rgb, soft_composite, write_csv, json_dump  # noqa: E402


OUT_ROOT = ROOT / "research" / "outputs" / "restoration_v2" / "reference_consistency"
IMAGE_ROOT = ROOT / "research" / "assets" / "distortion_segmentation_annotation_dataset" / "for_segmentation" / "images"


def source_group(stem: str) -> str:
    match = re.search(r"_([A-Z]{10})_", stem)
    return match.group(1) if match else "unknown"


def _resize_for_matching(rgb: np.ndarray, side: int = 512) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, float(side) / max(h, w))
    if scale == 1.0:
        return rgb, scale
    return cv2.resize(rgb, (max(32, int(round(w * scale))), max(32, int(round(h * scale)))), interpolation=cv2.INTER_AREA), scale


def _small_to_native(h_small: np.ndarray, ref_scale: float, target_scale: float) -> np.ndarray:
    """Convert H: x_target_small = H x_reference_small to native pixels.

    x_reference_small = S_ref x_reference_native and
    x_target_native = inv(S_target) x_target_small, hence
    H_native = inv(S_target) @ H_small @ S_ref.
    """
    s_ref = np.diag([ref_scale, ref_scale, 1.0])
    inv_s_target = np.diag([1.0 / target_scale, 1.0 / target_scale, 1.0])
    result = inv_s_target @ h_small @ s_ref
    return result / max(abs(result[2, 2]), 1e-12)


def _find_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        return cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0, confidence=0.999, maxIters=10000)
    except (AttributeError, cv2.error):
        return cv2.findHomography(src, dst, cv2.RANSAC, 5.0)


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography).reshape(-1, 2)


def _empty_info(reason: str) -> dict:
    return {
        "ok": False,
        "reason": reason,
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "retrieval_score": 0.0,
        "warped": None,
        "support": None,
        "confidence": None,
        "ref_points": np.empty((0, 2), np.float32),
        "target_points": np.empty((0, 2), np.float32),
        "inlier_points_ref": np.empty((0, 2), np.float32),
        "inlier_points_target": np.empty((0, 2), np.float32),
        "forward_errors": np.empty((0,), np.float32),
        "backward_errors": np.empty((0,), np.float32),
        "fb_errors": np.empty((0,), np.float32),
        "local_reprojection_error_px": float("nan"),
        "forward_backward_consistency_px": float("nan"),
    }


def _sift_alignment(reference: np.ndarray, target: np.ndarray) -> dict:
    ref_small, ref_scale = _resize_for_matching(reference)
    target_small, target_scale = _resize_for_matching(target)
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.02)
    ref_kp, ref_desc = sift.detectAndCompute(cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY), None)
    target_kp, target_desc = sift.detectAndCompute(cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY), None)
    if ref_desc is None or target_desc is None or len(ref_kp) < 4 or len(target_kp) < 4:
        return _empty_info("insufficient_sift_features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_desc, target_desc, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    if len(good) < 4:
        info = _empty_info("few_ratio_test_matches")
        info["matches"] = len(good)
        return info
    src_small = np.float32([ref_kp[m.queryIdx].pt for m in good])
    dst_small = np.float32([target_kp[m.trainIdx].pt for m in good])
    h_small, inlier_mask = _find_homography(src_small, dst_small)
    h_back_small, _ = _find_homography(dst_small, src_small)
    if h_small is None or h_back_small is None or inlier_mask is None:
        info = _empty_info("homography_failed")
        info["matches"] = len(good)
        return info
    h_native = _small_to_native(h_small, ref_scale, target_scale)
    h_back_native = _small_to_native(h_back_small, target_scale, ref_scale)
    src_native = src_small / ref_scale
    dst_native = dst_small / target_scale
    inliers = inlier_mask.ravel().astype(bool)
    src_in = src_native[inliers]
    dst_in = dst_native[inliers]
    forward_points = _transform_points(src_in, h_native)
    backward_points = _transform_points(dst_in, h_back_native)
    round_trip_points = _transform_points(forward_points, h_back_native)
    forward_errors = np.linalg.norm(forward_points - dst_in, axis=1)
    backward_errors = np.linalg.norm(backward_points - src_in, axis=1)
    fb_errors = np.linalg.norm(round_trip_points - src_in, axis=1)
    target_h, target_w = target.shape[:2]
    ref_h, ref_w = reference.shape[:2]
    support = cv2.warpPerspective(np.ones((ref_h, ref_w), np.uint8), h_native, (target_w, target_h), flags=cv2.INTER_NEAREST) > 0
    confidence = np.zeros((target_h, target_w), dtype=np.float32)
    for point, error in zip(dst_in, forward_errors):
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= x < target_w and 0 <= y < target_h:
            cv2.circle(confidence, (x, y), 14, float(np.exp(-error / 6.0)), -1)
    confidence = cv2.GaussianBlur(confidence, (0, 0), 10)
    if confidence.max() > 0:
        confidence /= float(confidence.max())
    confidence *= support.astype(np.float32)
    warped = cv2.warpPerspective(reference, h_native, (target_w, target_h), flags=cv2.INTER_LINEAR)
    inlier_ratio = float(inliers.mean())
    projected = _transform_points(np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]), h_native)
    projected[:, 0] = np.clip(projected[:, 0], 0, target_w - 1)
    projected[:, 1] = np.clip(projected[:, 1], 0, target_h - 1)
    coverage = float(abs(cv2.contourArea(projected.astype(np.float32))) / max(1.0, target_w * target_h))
    return {
        "ok": False,
        "reason": "candidate",
        "matches": len(good),
        "inliers": int(inliers.sum()),
        "inlier_ratio": inlier_ratio,
        "retrieval_score": float(inliers.sum() * inlier_ratio),
        "coverage": coverage,
        "homography": h_native,
        "homography_back": h_back_native,
        "warped": warped,
        "support": support,
        "confidence": confidence,
        "ref_points": src_native,
        "target_points": dst_native,
        "inlier_points_ref": src_in,
        "inlier_points_target": dst_in,
        "forward_errors": forward_errors,
        "backward_errors": backward_errors,
        "fb_errors": fb_errors,
        "keypoints_ref": ref_kp,
        "keypoints_target": target_kp,
        "good_matches": good,
        "inlier_mask": inlier_mask,
        "local_reprojection_error_px": float("nan"),
        "forward_backward_consistency_px": float(np.median(fb_errors)) if len(fb_errors) else float("nan"),
    }


def _lab_adjust(warped: np.ndarray, target: np.ndarray, photo_region: np.ndarray) -> np.ndarray:
    """Color match using only clean target pixels outside the edit mask."""
    if np.count_nonzero(photo_region) < 64:
        return warped.copy()
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    warped_lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    output = warped_lab.copy()
    for channel in range(3):
        source_values = warped_lab[..., channel][photo_region]
        target_values = target_lab[..., channel][photo_region]
        output[..., channel] = (warped_lab[..., channel] - source_values.mean()) * (target_values.std() / max(source_values.std(), 1.0)) + target_values.mean()
    return cv2.cvtColor(output.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _score_mask(info: dict, target: np.ndarray, mask: np.ndarray) -> dict:
    hard = np.asarray(mask) > 0
    if info.get("warped") is None or info.get("support") is None:
        return {"reliable": False, "reason": info.get("reason", "no_alignment"), "mask_support_ratio": 0.0, "mask_high_confidence_ratio": 0.0, "mask_confidence_mean": 0.0, "mask_outside_photometric_residual_lab_l1": float("nan"), "local_reprojection_error_px": float("nan"), "forward_backward_consistency_px": float("nan"), "composite": target.copy(), "color_adjusted": info.get("warped")}
    support = info["support"]
    confidence = info["confidence"]
    high = confidence >= 0.25
    mask_count = max(1, int(hard.sum()))
    valid_in_mask = hard & support
    high_in_mask = valid_in_mask & high
    photo_region = support & ~dilate(hard, 12) & (confidence >= 0.05)
    adjusted = _lab_adjust(info["warped"], target, photo_region)
    composite, allowed = soft_composite(target, adjusted, hard, dilate_px=3, feather=1.5)
    target_points = info["inlier_points_target"]
    local = np.zeros((len(target_points),), dtype=bool)
    if len(target_points):
        yi = np.clip(np.rint(target_points[:, 1]).astype(int), 0, hard.shape[0] - 1)
        xi = np.clip(np.rint(target_points[:, 0]).astype(int), 0, hard.shape[1] - 1)
        local = hard[yi, xi]
    local_error_values = info["forward_errors"][local] if len(local) else np.empty((0,), dtype=np.float32)
    local_error = float(np.median(local_error_values)) if len(local_error_values) else float("nan")
    lab_target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_adjusted = cv2.cvtColor(adjusted, cv2.COLOR_RGB2LAB).astype(np.float32)
    residual = float(np.abs(lab_adjusted[photo_region] - lab_target[photo_region]).mean()) if np.any(photo_region) else float("nan")
    allowed_outside = ~allowed
    outside_delta = np.abs(composite.astype(np.int16) - target.astype(np.int16))
    outside_max = int(outside_delta[allowed_outside].max()) if np.any(allowed_outside) else 0
    mask_support_ratio = float(valid_in_mask.sum() / mask_count)
    mask_high_ratio = float(high_in_mask.sum() / mask_count)
    mask_conf = float(confidence[hard].mean()) if np.any(hard) else 0.0
    fb = float(info.get("forward_backward_consistency_px", float("nan")))
    reliable = bool(
        info.get("inliers", 0) >= 8
        and info.get("inlier_ratio", 0.0) >= 0.2
        and mask_support_ratio >= 0.4
        and mask_high_ratio >= 0.2
        and (not np.isfinite(local_error) or local_error <= 10.0)
        and (not np.isfinite(fb) or fb <= 12.0)
    )
    reason = "ok" if reliable else "low_mask_confidence_or_consistency"
    return {
        "reliable": reliable,
        "reason": reason,
        "mask_support_ratio": mask_support_ratio,
        "mask_high_confidence_ratio": mask_high_ratio,
        "mask_confidence_mean": mask_conf,
        "mask_outside_photometric_residual_lab_l1": residual,
        "local_reprojection_error_px": local_error,
        "forward_backward_consistency_px": fb,
        "composite": composite if reliable else target.copy(),
        "color_adjusted": adjusted,
        "support": support,
        "confidence": confidence,
        "allowed": allowed,
        "outside_max": outside_max,
        "photo_region": photo_region,
    }


def _confidence_rgb(confidence: np.ndarray | None) -> np.ndarray:
    if confidence is None:
        return np.zeros((900, 900, 3), dtype=np.uint8)
    color = cv2.applyColorMap(np.rint(np.clip(confidence, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _diff_rgb(input_rgb: np.ndarray, output_rgb: np.ndarray) -> np.ndarray:
    diff = np.abs(input_rgb.astype(np.int16) - output_rgb.astype(np.int16)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(np.maximum(diff.max(axis=2), 0), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def _zoom(rgb: np.ndarray, mask: np.ndarray, size: int = 220) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        crop = rgb
    else:
        pad = max(24, int(round(max(rgb.shape[:2]) * 0.04)))
        crop = rgb[max(0, ys.min() - pad):min(rgb.shape[0], ys.max() + pad + 1), max(0, xs.min() - pad):min(rgb.shape[1], xs.max() + pad + 1)]
    h, w = crop.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    return cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


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
        info["good_matches"], None, matchesMask=info["inlier_mask"].ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _candidate_pool() -> list[Path]:
    paths = []
    for path in (ROOT / "research" / "assets").rglob("*.png"):
        if "mask_for_sam" in path.parts or "outputs" in path.parts:
            continue
        try:
            if path.resolve().is_file() and path.parent.name == "images":
                paths.append(path.resolve())
        except OSError:
            continue
    return sorted(set(paths))


def run(args) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    showcase = read_csv(ROOT / "research" / "data" / "open3dhk_showcase.csv")
    pool = _candidate_pool()
    by_group: dict[str, list[Path]] = {}
    for path in pool:
        by_group.setdefault(source_group(path.stem), []).append(path)
    metric_rows = []
    candidate_rows = []
    manual_tiles, sam_tiles, blind_tiles = [], [], []
    reliable_by_source = {"manual": [], "sam": []}
    out_dir = OUT_ROOT / "per_image"
    for item in showcase:
        target_path = ROOT / item["image_path"]
        target = load_rgb(target_path)
        manual = load_mask(ROOT / item["annotation_mask_path"])
        sam = load_mask(ROOT / item["sam_mask_path"])
        candidates = [path for path in by_group.get(item["source_group"], []) if path != target_path.resolve()]
        scored = []
        for candidate_path in candidates:
            reference = load_rgb(candidate_path)
            info = _sift_alignment(reference, target)
            scored.append((info.get("retrieval_score", 0.0), candidate_path, reference, info))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:5]
        for rank, (score, candidate_path, reference, info) in enumerate(top, start=1):
            candidate_rows.append({"unique_image_id": item["sample_id"], "showcase_index": item["showcase_index"], "rank": str(rank), "reference_path": str(candidate_path.relative_to(ROOT)), "retrieval_score": f"{score:.6f}", "matches": str(info.get("matches", 0)), "inliers": str(info.get("inliers", 0)), "inlier_ratio": f"{info.get('inlier_ratio', 0.0):.6f}", "forward_backward_consistency_px": f"{info.get('forward_backward_consistency_px', float('nan')):.6f}"})
        for source, mask in (("manual", manual), ("sam", sam)):
            if top:
                selected = None
                selected_score = None
                selected_rank = None
                selected_reference = None
                selected_info = None
                selected_metrics = None
                for rank, (score, candidate_path, reference, info) in enumerate(top, start=1):
                    metrics = _score_mask(info, target, mask)
                    selection_score = metrics["mask_support_ratio"] * max(metrics["mask_confidence_mean"], 1e-6) / max(1.0, metrics["local_reprojection_error_px"] if np.isfinite(metrics["local_reprojection_error_px"]) else 1.0)
                    if selected is None or (metrics["reliable"], selection_score) > (selected_metrics["reliable"], selected_score):
                        selected, selected_score, selected_rank, selected_reference, selected_info, selected_metrics = candidate_path, selection_score, rank, reference, info, metrics
            else:
                selected = None
                selected_score = 0.0
                selected_rank = 0
                selected_reference = target.copy()
                selected_info = _empty_info("no_same_group_candidate")
                selected_metrics = _score_mask(selected_info, target, mask)
            image_dir = out_dir / str(item["showcase_index"]) / source
            image_dir.mkdir(parents=True, exist_ok=True)
            composite = selected_metrics["composite"]
            confidence_rgb = _confidence_rgb(selected_metrics.get("confidence"))
            if confidence_rgb.shape[:2] != target.shape[:2]:
                confidence_rgb = cv2.resize(confidence_rgb, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            diff = _diff_rgb(target, composite)
            save_rgb(image_dir / "input.png", target)
            save_rgb(image_dir / "mask.png", np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2))
            save_rgb(image_dir / "reference.png", selected_reference)
            save_rgb(image_dir / "sift_warp.png", selected_info.get("warped") if selected_info.get("warped") is not None else target)
            save_rgb(image_dir / "confidence.png", confidence_rgb)
            save_rgb(image_dir / "composite.png", composite)
            save_rgb(image_dir / "zoom.png", _zoom(composite, mask))
            save_rgb(image_dir / "difference.png", diff)
            allowed_max = int(np.abs(composite.astype(np.int16) - target.astype(np.int16))[~dilate(mask, 3)].max()) if np.any(~dilate(mask, 3)) else 0
            row = {
                "unique_image_id": item["sample_id"],
                "showcase_index": item["showcase_index"],
                "source_group": item["source_group"],
                "mask_source": source,
                "reference_rank": str(selected_rank),
                "reference_path": str(selected.relative_to(ROOT)) if selected else "",
                "candidate_pool_size": str(len(candidates)),
                "top5_count": str(len(top)),
                "reliable": str(bool(selected_metrics["reliable"])),
                "reason": selected_metrics["reason"],
                "mask_support_ratio": f"{selected_metrics['mask_support_ratio']:.6f}",
                "mask_high_confidence_ratio": f"{selected_metrics['mask_high_confidence_ratio']:.6f}",
                "mask_confidence_mean": f"{selected_metrics['mask_confidence_mean']:.6f}",
                "forward_backward_consistency_px": f"{selected_metrics['forward_backward_consistency_px']:.6f}",
                "local_reprojection_error_px": f"{selected_metrics['local_reprojection_error_px']:.6f}",
                "mask_outside_photometric_residual_lab_l1": f"{selected_metrics['mask_outside_photometric_residual_lab_l1']:.6f}",
                "allowed_outside_max_abs": str(allowed_max),
                "input_path": str(target_path.relative_to(ROOT)),
                "composite_path": str((image_dir / "composite.png").relative_to(ROOT)),
                "contact_sheet_link": "open3dhk_reference_consistency_contact_sheet.png",
            }
            metric_rows.append(row)
            reliable_by_source[source].append(bool(selected_metrics["reliable"]))
            if source == "manual":
                manual_tiles.extend([(f"{item['showcase_index']} input", target), (f"{item['showcase_index']} mask", np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2)), (f"{item['showcase_index']} reference", selected_reference), (f"{item['showcase_index']} SIFT warp", selected_info.get("warped") if selected_info.get("warped") is not None else target), (f"{item['showcase_index']} confidence", confidence_rgb), (f"{item['showcase_index']} composite", composite), (f"{item['showcase_index']} zoom", _zoom(composite, mask)), (f"{item['showcase_index']} difference", diff)])
                blind_tiles.extend([(f"{item['showcase_index']} input", target), (f"{item['showcase_index']} A", target), (f"{item['showcase_index']} B", composite)])
            else:
                sam_tiles.extend([(f"{item['showcase_index']} input", target), (f"{item['showcase_index']} mask", np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2)), (f"{item['showcase_index']} reference", selected_reference), (f"{item['showcase_index']} SIFT warp", selected_info.get("warped") if selected_info.get("warped") is not None else target), (f"{item['showcase_index']} confidence", confidence_rgb), (f"{item['showcase_index']} composite", composite), (f"{item['showcase_index']} zoom", _zoom(composite, mask)), (f"{item['showcase_index']} difference", diff)])
    write_csv(OUT_ROOT / "reference_consistency_metrics.csv", metric_rows, list(metric_rows[0].keys()))
    write_csv(OUT_ROOT / "reference_consistency_candidates.csv", candidate_rows, list(candidate_rows[0].keys()) if candidate_rows else ["unique_image_id"])
    make_grid(manual_tiles, OUT_ROOT / "open3dhk_reference_consistency_contact_sheet.png", columns=8, tile_size=(220, 190))
    make_grid(sam_tiles, OUT_ROOT / "open3dhk_reference_consistency_contact_sheet_sam.png", columns=8, tile_size=(220, 190))
    make_grid(blind_tiles, OUT_ROOT / "open3dhk_reference_identity_vs_method_blind.png", columns=3, tile_size=(260, 220))
    preference = [{"unique_image_id": item["sample_id"], "mask_source": source, "better": "", "same": "", "worse": "", "notes": ""} for item in showcase for source in ("manual", "sam")]
    write_csv(OUT_ROOT / "human_preference.csv", preference, ["unique_image_id", "mask_source", "better", "same", "worse", "notes"])
    summary = {"unique_images": len(showcase), "candidate_pool_unique_rgb": len(pool), "manual": {}, "sam": {}, "contact_sheets": {"manual": "open3dhk_reference_consistency_contact_sheet.png", "sam": "open3dhk_reference_consistency_contact_sheet_sam.png", "blind": "open3dhk_reference_identity_vs_method_blind.png"}, "metrics_csv": "reference_consistency_metrics.csv", "candidates_csv": "reference_consistency_candidates.csv", "human_preference_csv": "human_preference.csv", "homography_coordinate_fix": "H_native=inv(S_target) @ H_small @ S_reference", "roma_status": "not_run_in_this_script"}
    for source in ("manual", "sam"):
        subset = [r for r in metric_rows if r["mask_source"] == source]
        support = np.asarray([float(r["mask_support_ratio"]) for r in subset])
        reliable = np.asarray([r["reliable"] == "True" for r in subset])
        summary[source] = {"images": len(subset), "reliable_images": int(reliable.sum()), "coverage_ratio": float(reliable.mean()) if len(reliable) else 0.0, "median_mask_support_ratio": float(np.median(support)) if len(support) else 0.0, "median_mask_confidence": float(np.median([float(r["mask_confidence_mean"]) for r in subset])) if subset else 0.0, "mask_outside_max_abs": int(max(int(r["allowed_outside_max_abs"]) for r in subset)) if subset else 0}
    json_dump(OUT_ROOT / "reference_consistency_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-only", action="store_true", help="reserved for reproducible reruns; the full contact sheet is always rebuilt")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
