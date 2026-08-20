#!/usr/bin/env python3
"""Evaluate real restoration-v2 outputs on proxy pairs and fixed Open3DHK."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))
from common import (  # noqa: E402
    dilate,
    json_dump,
    load_mask,
    load_rgb,
    lpips_distance,
    masked_psnr,
    masked_ssim,
    metric_row,
    read_csv,
    save_rgb,
    soft_composite,
    write_csv,
)


OUT_ROOT = ROOT / "research" / "outputs" / "restoration_v2"


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _unsharp(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).filter(ImageFilter.UnsharpMask(radius=2.0, percent=100, threshold=3)), dtype=np.uint8)


def _load_lpips():
    import lpips

    return lpips.LPIPS(net="alex").cuda().eval()


def _evaluate_output(input_rgb, output_rgb, mask, target, lpips_metric, method, runtime=""):
    row = metric_row(input_rgb, output_rgb, mask, target)
    row["method"] = method
    row["runtime_sec"] = runtime
    row["masked_lpips"] = lpips_distance(lpips_metric, output_rgb, target, mask)
    return row


def _proxy_evaluation() -> list[dict]:
    rows = read_csv(ROOT / "research" / "data" / "restoration_v2_proxy_pairs.csv")
    manifest_path = OUT_ROOT / "real_backend_manifest_proxy.csv"
    if not manifest_path.exists():
        manifest_path = OUT_ROOT / "real_backend_manifest.csv"
    backend = {r["sample_id"]: r for r in read_csv(manifest_path)} if manifest_path.exists() else {}
    metric = _load_lpips()
    records = []
    for item in rows:
        sample_id = item["sample_id"]
        input_rgb = load_rgb(_path(item["distorted_path"]))
        target = load_rgb(_path(item["clean_path"]))
        mask = load_mask(_path(item["mask_path"]))
        entry = backend.get(sample_id, {})
        candidates = {
            "Identity": (input_rgb, "0"),
            "B1_unsharp_weak": (soft_composite(input_rgb, _unsharp(input_rgb), mask, 3, 1.5)[0], ""),
            "reference_oracle": (target, ""),
        }
        for name, key in (("R1_NAFNet", "R1_NAFNet_path"), ("R2_LaMa", "R2_LaMa_path")):
            value = entry.get(key, "")
            if value and _path(value).exists():
                candidates[name] = (load_rgb(_path(value)), entry.get(name[3:].lower() + "_runtime_sec", entry.get("r1_runtime_sec" if name.startswith("R1") else "r2_runtime_sec", "")))
        for method, (output, runtime) in candidates.items():
            values = _evaluate_output(input_rgb, output, mask, target, metric, method, runtime)
            values.update({
                "row_type": "sample",
                "sample_id": sample_id,
                "split": item["split"],
                "degradation": item["degradation_type"],
                "severity": item["severity"],
            })
            records.append(values)
    # Summary uses median as the primary robust statistic and also retains mean.
    sample_methods = sorted({r["method"] for r in records})
    sample_degradations = sorted({r["degradation"] for r in records})
    for method in sample_methods:
        for degradation in ["all"] + sample_degradations:
            subset = [r for r in records if r.get("row_type") == "sample" and r["method"] == method and (degradation == "all" or r["degradation"] == degradation)]
            if not subset:
                continue
            identity = {r["sample_id"]: r for r in records if r.get("row_type") == "sample" and r["method"] == "Identity" and (degradation == "all" or r["degradation"] == degradation)}
            psnr = np.asarray([float(r["masked_psnr"]) for r in subset], dtype=float)
            ssim = np.asarray([float(r["masked_ssim"]) for r in subset], dtype=float)
            lp = np.asarray([float(r["masked_lpips"]) for r in subset], dtype=float)
            wins = []
            for r in subset:
                base = identity.get(r["sample_id"])
                if base is not None:
                    wins.append(float(r["masked_psnr"]) > float(base["masked_psnr"]) + 1e-6 and float(r["masked_ssim"]) >= float(base["masked_ssim"]) - 1e-6 and float(r["masked_lpips"]) < float(base["masked_lpips"]) - 1e-5)
            records.append({
                "row_type": "summary",
                "method": method,
                "degradation": degradation,
                "sample_id": "",
                "split": "all",
                "severity": "",
                "masked_psnr": f"{np.mean(psnr):.6f}",
                "masked_psnr_median": f"{np.median(psnr):.6f}",
                "masked_ssim": f"{np.mean(ssim):.6f}",
                "masked_ssim_median": f"{np.median(ssim):.6f}",
                "masked_lpips": f"{np.mean(lp):.6f}",
                "masked_lpips_median": f"{np.median(lp):.6f}",
                "win_rate_vs_identity": f"{np.mean(wins):.6f}" if wins else "",
                "n": str(len(subset)),
            })
    return records


def _reference_candidate(input_rgb: np.ndarray, mask: np.ndarray, ref_row: dict[str, str]) -> np.ndarray | None:
    warped_value = ref_row.get("warped_path", "")
    if not ref_row.get("reliable") == "True" or not warped_value or not _path(warped_value).exists():
        return None
    warped = load_rgb(_path(warped_value))
    return soft_composite(input_rgb, warped, mask, dilate_px=3, feather=1.5)[0]


def _showcase_evaluation() -> list[dict]:
    showcase = read_csv(ROOT / "research" / "data" / "open3dhk_showcase.csv")
    backend_path = OUT_ROOT / "real_backend_manifest_showcase.csv"
    backend = read_csv(backend_path) if backend_path.exists() else []
    backend_map = {(r.get("showcase_index", ""), r.get("mask_source", "")): r for r in backend}
    ref_rows = {r["showcase_index"]: r for r in read_csv(OUT_ROOT / "reference_coverage.csv")} if (OUT_ROOT / "reference_coverage.csv").exists() else {}
    records = []
    for item in showcase:
        input_rgb = load_rgb(_path(item["image_path"]))
        ref_row = ref_rows.get(item["showcase_index"], {})
        for source in ("manual", "sam"):
            mask = load_mask(_path(item["annotation_mask_path"] if source == "manual" else item["sam_mask_path"]))
            if mask.shape != input_rgb.shape[:2]:
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((input_rgb.shape[1], input_rgb.shape[0]), Image.Resampling.NEAREST)) > 127
            entry = backend_map.get((item["showcase_index"], source), {})
            candidates = {"Identity_abstain": input_rgb}
            for name, key in (("R1_NAFNet", "R1_NAFNet_path"), ("R2_LaMa", "R2_LaMa_path")):
                value = entry.get(key, "")
                if value and _path(value).exists():
                    candidates[name] = load_rgb(_path(value))
            reference = _reference_candidate(input_rgb, mask, ref_row)
            if reference is not None:
                candidates["reference"] = reference
            severity = item["severity"]
            if severity in {"small", "medium"} and "R1_NAFNet" in candidates:
                final_method = "R1_NAFNet"
            elif severity == "large" and reference is not None:
                final_method = "reference"
            elif "R2_LaMa" in candidates:
                final_method = "R2_LaMa"
            else:
                final_method = "abstain"
            if final_method == "abstain":
                candidates["final"] = input_rgb
            else:
                candidates["final"] = candidates[final_method]
            for method, output in candidates.items():
                values = metric_row(input_rgb, output, mask, None)
                values.update({
                    "row_type": "sample",
                    "sample_id": item["sample_id"],
                    "showcase_index": item["showcase_index"],
                    "mask_source": source,
                    "severity": severity,
                    "category": item["category"],
                    "method": method,
                    "final_route": final_method if method == "final" else "",
                    "manual_sam_iou": item.get("manual_sam_iou", ""),
                    "sam_mask_ratio": item.get("sam_mask_ratio", ""),
                    "reference_reliable": ref_row.get("reliable", "False"),
                    "reference_inlier_ratio": ref_row.get("inlier_ratio", ""),
                    "reference_coverage": ref_row.get("coverage", ""),
                })
                records.append(values)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--showcase-only", action="store_true")
    args = parser.parse_args()
    proxy = [] if args.showcase_only else _proxy_evaluation()
    showcase = [] if args.proxy_only else _showcase_evaluation()
    fields = sorted({key for row in proxy for key in row} | {key for row in showcase for key in row})
    if proxy:
        write_csv(OUT_ROOT / "proxy_metrics.csv", proxy, fields)
    if showcase:
        write_csv(OUT_ROOT / "open3dhk_metrics.csv", showcase, fields)
    preference_rows = []
    for row in read_csv(ROOT / "research" / "data" / "open3dhk_showcase.csv"):
        for source in ("manual", "sam"):
            preference_rows.append({"sample_id": row["sample_id"], "mask_source": source, "better": "", "same": "", "worse": "", "notes": ""})
    write_csv(OUT_ROOT / "human_preference.csv", preference_rows, ["sample_id", "mask_source", "better", "same", "worse", "notes"])
    summary = {}
    if proxy:
        summary_rows = [r for r in proxy if r.get("row_type") == "summary" and r.get("degradation") == "all"]
        summary["proxy_summary_all"] = summary_rows
    if showcase:
        finals = [r for r in showcase if r.get("method") == "final"]
        summary["showcase"] = {
            "rows": len(showcase),
            "samples": len({r["sample_id"] for r in finals}),
            "final_abstain": sum(r.get("final_route") == "abstain" for r in finals),
            "final_route_counts": {route: sum(r.get("final_route") == route for r in finals) for route in sorted({r.get("final_route") for r in finals})},
            "sam_manual_iou_mean": float(np.mean([float(r["manual_sam_iou"]) for r in finals])) if finals else None,
            "reference_reliable_samples": sum(r.get("reference_reliable") == "True" for r in finals),
        }
    json_dump(OUT_ROOT / "short_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
