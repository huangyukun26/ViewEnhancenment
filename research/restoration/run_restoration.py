#!/usr/bin/env python3
"""Run the short Fidelity-Guarded Masked Restoration cycle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import (
    binary_dilate,
    binary_erode,
    crop_zoom,
    deterministic_candidate,
    feature_vector,
    fit_guard_model,
    guard_score,
    grid,
    json_dump,
    jpeg_roundtrip,
    load_rgb,
    metric_row,
    masked_output_std,
    overlay_mask,
    read_csv,
    relpath,
    repo_root,
    save_rgb,
    soft_alpha,
    stochastic_candidate,
    strict_composite,
    write_csv,
    scale_roundtrip,
)


METHODS = ("B0_identity", "B1_deterministic", "G_lite_stochastic_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--skip-robustness", action="store_true")
    return parser.parse_args()


def mask_from_row(root: Path, row: dict, key: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(root / row[key]).convert("L"), dtype=np.uint8) > 0


def candidate_images(input_rgb: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    # All candidates share the same narrow feathered alpha. Outside the mask
    # (and its 3-pixel allowed dilation) output pixels are exact input pixels.
    alpha = soft_alpha(mask, dilate=3, feather=2.0)
    return {
        "B0_identity": input_rgb.copy(),
        "B1_deterministic": strict_composite(input_rgb, deterministic_candidate(input_rgb, 1.0), alpha),
        "G_lite_stochastic_proxy": strict_composite(input_rgb, stochastic_candidate(input_rgb, seed), alpha),
    }


def metrics_for_candidates(input_rgb: np.ndarray, candidates: dict[str, np.ndarray], mask: np.ndarray, gt: np.ndarray | None, sample_id: str, split: str, seed: int) -> list[dict]:
    rows = []
    for method, output in candidates.items():
        row = metric_row(input_rgb, output, mask, gt)
        row.update({"sample_id": sample_id, "split": split, "method": method, "seed": seed})
        rows.append(row)
    return rows


def choose_ours(input_rgb: np.ndarray, candidates: dict[str, np.ndarray], mask: np.ndarray, model: dict, seed: int) -> tuple[str, np.ndarray, dict]:
    options = []
    for method, output in candidates.items():
        metrics = metric_row(input_rgb, output, mask, None)
        options.append((guard_score(metrics, model), method, output, metrics))
    options.sort(key=lambda item: (item[0], item[1]))
    score, method, output, metrics = options[0]
    metrics = dict(metrics)
    metrics["guard_score"] = float(score)
    metrics["selected_from"] = method
    return method, output, metrics


def proxy_experiment(root: Path) -> tuple[list[dict], dict, dict]:
    rows = read_csv(root / "research" / "data" / "open3dhk_proxy_pairs.csv")
    raw_metrics: list[dict] = []
    per_sample: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        clean = load_rgb(root / row["clean_path"])
        distorted = load_rgb(root / row["distorted_path"])
        mask = mask_from_row(root, row, "mask_path")
        candidates = candidate_images(distorted, mask, int(row["seed"]))
        per_sample[row["sample_id"]] = candidates
        raw_metrics.extend(metrics_for_candidates(distorted, candidates, mask, clean, row["sample_id"], row["split"], int(row["seed"])))

    train_rows = [r for r in raw_metrics if r["split"] == "train"]
    model = fit_guard_model(train_rows)
    json_dump(root / "research" / "outputs" / "restoration" / "guard_selection_model.json", model)

    selected_rows = []
    for row in rows:
        distorted = load_rgb(root / row["distorted_path"])
        clean = load_rgb(root / row["clean_path"])
        mask = mask_from_row(root, row, "mask_path")
        method, output, selection_metrics = choose_ours(distorted, per_sample[row["sample_id"]], mask, model, int(row["seed"]))
        ours = metric_row(distorted, output, mask, clean)
        ours.update({"sample_id": row["sample_id"], "split": row["split"], "method": "Ours_guard_selected", "seed": row["seed"], "selected_from": method, "guard_score": selection_metrics["guard_score"]})
        selected_rows.append(ours)
    all_rows = raw_metrics + selected_rows
    fields = [
        "sample_id", "split", "method", "seed", "mask_ratio", "masked_psnr", "masked_ssim", "masked_lpips",
        "masked_l1", "mask_outside_max_abs", "mask_outside_mean_abs", "original_mask_outside_max_abs", "original_mask_outside_mean_abs",
        "boundary_color_l1", "boundary_edge_l1",
        "inner_color_l1", "color_mean_delta", "color_std_delta", "sharpness_gain", "input_sharpness", "output_sharpness",
        "selected_from", "guard_score",
    ]
    write_csv(root / "research" / "outputs" / "restoration" / "proxy_metrics.csv", all_rows, fields)

    # Small proxy overview grid (GT is shown only for the domain proxy).
    tiles: list[tuple[str, np.ndarray]] = []
    for row in rows[:10]:
        clean = load_rgb(root / row["clean_path"])
        distorted = load_rgb(root / row["distorted_path"])
        mask = mask_from_row(root, row, "mask_path")
        candidates = per_sample[row["sample_id"]]
        _, ours, _ = choose_ours(distorted, candidates, mask, model, int(row["seed"]))
        tiles.extend([
            (f"{row['sample_id'][:12]} input", distorted),
            ("mask", np.repeat(mask[..., None] * 255, 3, axis=2).astype(np.uint8)),
            ("B1", candidates["B1_deterministic"]),
            ("G-lite", candidates["G_lite_stochastic_proxy"]),
            ("Ours", ours),
            ("clean proxy GT", clean),
        ])
    grid(tiles, root / "research" / "outputs" / "restoration" / "proxy_grid.png", columns=6, tile_size=(190, 170))
    summary = summarize_proxy(all_rows)
    json_dump(root / "research" / "outputs" / "restoration" / "proxy_summary.json", summary)
    return all_rows, model, summary


def summarize_proxy(rows: list[dict]) -> dict:
    summary = {}
    for split in ("train", "val"):
        split_rows = [row for row in rows if row.get("split") == split]
        summary[split] = {}
        for method in sorted({row["method"] for row in split_rows}):
            method_rows = [row for row in split_rows if row["method"] == method]
            summary[split][method] = {
                "count": len(method_rows),
                "masked_psnr_mean": float(np.nanmean([float(r["masked_psnr"]) for r in method_rows])) if method_rows else None,
                "masked_ssim_mean": float(np.nanmean([float(r["masked_ssim"]) for r in method_rows])) if method_rows else None,
                "masked_l1_mean": float(np.nanmean([float(r.get("masked_l1", "nan")) for r in method_rows])) if method_rows else None,
                "outside_max_abs_max": int(max(int(r["mask_outside_max_abs"]) for r in method_rows)) if method_rows else None,
            }
    return summary


def robustness_experiment(root: Path, model: dict) -> list[dict]:
    proxy_rows = [row for row in read_csv(root / "research" / "data" / "open3dhk_proxy_pairs.csv") if row["split"] == "val"]
    output_rows = []
    mask_deltas = (-7, -3, 0, 3, 7)
    for row in proxy_rows:
        clean = load_rgb(root / row["clean_path"])
        input_rgb = load_rgb(root / row["distorted_path"])
        original_mask = mask_from_row(root, row, "mask_path")
        for mask_delta in mask_deltas:
            mask = binary_erode(original_mask, -mask_delta) if mask_delta < 0 else binary_dilate(original_mask, mask_delta)
            for jpeg_quality in (60, 80):
                for scale in (0.75, 1.25):
                    transformed = scale_roundtrip(jpeg_roundtrip(input_rgb, jpeg_quality), scale)
                    for seed in (0, 1, 2):
                        candidates = candidate_images(transformed, mask, seed)
                        selected, output, select_metrics = choose_ours(transformed, candidates, mask, model, seed)
                        metrics = metric_row(transformed, output, mask, clean)
                        metrics.update({
                            "sample_id": row["sample_id"], "split": "val", "method": "Ours_guard_selected",
                            "mask_delta_px": mask_delta, "jpeg_quality": jpeg_quality, "scale": scale, "seed": seed,
                            "selected_from": selected, "guard_score": select_metrics["guard_score"],
                        })
                        output_rows.append(metrics)
    fields = [
        "sample_id", "split", "method", "mask_delta_px", "jpeg_quality", "scale", "seed", "selected_from", "guard_score",
        "mask_ratio", "masked_psnr", "masked_ssim", "masked_lpips", "masked_l1", "mask_outside_max_abs", "mask_outside_mean_abs",
        "original_mask_outside_max_abs", "original_mask_outside_mean_abs",
        "boundary_color_l1", "boundary_edge_l1", "inner_color_l1", "color_mean_delta", "color_std_delta", "sharpness_gain",
    ]
    write_csv(root / "research" / "outputs" / "restoration" / "robustness_metrics.csv", output_rows, fields)
    return output_rows


def load_showcase_mask(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray, str]:
    manual = mask_from_row(root, row, "annotation_mask_path")
    sam_path = row.get("sam_mask_path", "")
    if sam_path and (root / sam_path).exists():
        sam = mask_from_row(root, row, "sam_mask_path")
        return manual, sam, "SAM_LoRA_cpu"
    return manual, manual, "manual_annotation_fallback"


def run_showcase(root: Path, model: dict) -> dict:
    rows = read_csv(root / "research" / "data" / "open3dhk_showcase.csv")
    output_root = root / "research" / "outputs" / "restoration" / "showcase_results"
    grid_tiles: list[tuple[str, np.ndarray]] = []
    zoom_tiles: list[tuple[str, np.ndarray]] = []
    failure_records = []
    metric_rows = []
    for row in rows:
        rgb = load_rgb(root / row["image_path"])
        manual, used_mask, mask_source = load_showcase_mask(root, row)
        candidates = candidate_images(rgb, used_mask, int(row["showcase_index"]))
        selected, ours, select_metrics = choose_ours(rgb, candidates, used_mask, model, int(row["showcase_index"]))
        seed_candidates = [candidate_images(rgb, used_mask, seed) for seed in (0, 1, 2)]
        seed_outputs = {method: [candidate[method] for candidate in seed_candidates] for method in METHODS}
        seed_variance = {method: masked_output_std(outputs, used_mask) for method, outputs in seed_outputs.items()}
        seed_ours = [choose_ours(rgb, candidate, used_mask, model, seed)[1] for seed, candidate in zip((0, 1, 2), seed_candidates)]
        seed_variance["Ours_guard_selected"] = masked_output_std(seed_ours, used_mask)
        sample_dir = output_root / f"{int(row['showcase_index']):02d}_{Path(row['image_path']).stem}"
        save_rgb(sample_dir / "input.png", rgb)
        save_rgb(sample_dir / "manual_mask_overlay.png", overlay_mask(rgb, manual, color=(50, 180, 255)))
        save_rgb(sample_dir / "sam_mask_overlay.png", overlay_mask(rgb, used_mask, color=(255, 50, 50)))
        for method, output in candidates.items():
            save_rgb(sample_dir / f"{method}.png", output)
        save_rgb(sample_dir / "Ours_guard_selected.png", ours)
        manual_iou = float(np.logical_and(manual, used_mask).sum() / np.logical_or(manual, used_mask).sum()) if np.logical_or(manual, used_mask).any() else 1.0
        manual_dice = float(2.0 * np.logical_and(manual, used_mask).sum() / (manual.sum() + used_mask.sum())) if (manual.sum() + used_mask.sum()) else 1.0
        row_common = {"sample_id": row["sample_id"], "showcase_index": row["showcase_index"], "severity": row["severity"], "category": row["category"], "mask_source": mask_source, "manual_sam_iou": manual_iou, "manual_sam_dice": manual_dice, "sam_ratio": float(used_mask.mean()), "selected_from": selected}
        for method, output in {**candidates, "Ours_guard_selected": ours}.items():
            metrics = metric_row(rgb, output, used_mask, None)
            metrics.update(row_common)
            metrics["method"] = method
            metrics["guard_score"] = select_metrics["guard_score"] if method == "Ours_guard_selected" else guard_score(metric_row(rgb, output, used_mask, None), model)
            metrics["seed_output_variance"] = seed_variance[method]
            metric_rows.append(metrics)

        grid_tiles.extend([
            (f"{row['showcase_index']} input", rgb),
            ("manual mask", np.repeat(manual[..., None] * 255, 3, axis=2).astype(np.uint8)),
            ("SAM mask", np.repeat(used_mask[..., None] * 255, 3, axis=2).astype(np.uint8)),
            ("B1 deterministic", candidates["B1_deterministic"]),
            ("G-lite proxy", candidates["G_lite_stochastic_proxy"]),
            ("Ours", ours),
        ])
        zoom_tiles.extend([
            (f"{row['showcase_index']} input", crop_zoom(rgb, used_mask)),
            ("B1", crop_zoom(candidates["B1_deterministic"], used_mask)),
            ("G-lite", crop_zoom(candidates["G_lite_stochastic_proxy"], used_mask)),
            ("Ours", crop_zoom(ours, used_mask)),
        ])
        failure_records.append(((1.0 - manual_iou) + 0.1 * float(select_metrics["guard_score"]), row, rgb, manual, used_mask, candidates, ours))

    grid(grid_tiles, root / "research" / "outputs" / "restoration" / "open3dhk_showcase.png", columns=6, tile_size=(190, 170))
    grid(zoom_tiles, root / "research" / "outputs" / "restoration" / "zoom_comparison.png", columns=4, tile_size=(220, 220))
    failure_records.sort(key=lambda item: item[0], reverse=True)
    failure_tiles: list[tuple[str, np.ndarray]] = []
    failure_rows = []
    for score, row, rgb, manual, used_mask, candidates, ours in failure_records[:6]:
        failure_tiles.extend([
            (f"{row['showcase_index']} input", rgb),
            ("manual", np.repeat(manual[..., None] * 255, 3, axis=2).astype(np.uint8)),
            ("SAM", np.repeat(used_mask[..., None] * 255, 3, axis=2).astype(np.uint8)),
            ("B1", candidates["B1_deterministic"]),
            ("G-lite", candidates["G_lite_stochastic_proxy"]),
            ("Ours", ours),
        ])
        failure_rows.append({"sample_id": row["sample_id"], "failure_rank_score": score, "manual_sam_iou": float(np.logical_and(manual, used_mask).sum() / np.logical_or(manual, used_mask).sum()) if np.logical_or(manual, used_mask).any() else 1.0, "reason": "low mask agreement and/or high guard score; qualitative inspection required"})
    grid(failure_tiles, root / "research" / "outputs" / "restoration" / "failure_cases.png", columns=6, tile_size=(190, 170))
    write_csv(root / "research" / "outputs" / "restoration" / "failure_cases.csv", failure_rows, ["sample_id", "failure_rank_score", "manual_sam_iou", "reason"])

    fields = [
        "sample_id", "showcase_index", "severity", "category", "mask_source", "method", "manual_sam_iou", "manual_sam_dice", "sam_ratio", "selected_from", "guard_score", "seed_output_variance",
        "mask_ratio", "mask_outside_max_abs", "mask_outside_mean_abs", "original_mask_outside_max_abs", "original_mask_outside_mean_abs",
        "boundary_color_l1", "boundary_edge_l1", "inner_color_l1",
        "color_mean_delta", "color_std_delta", "sharpness_gain", "input_sharpness", "output_sharpness",
    ]
    write_csv(root / "research" / "outputs" / "restoration" / "open3dhk_metrics.csv", metric_rows, fields)
    summary = {
        "count": len(rows),
        "manual_sam_iou_mean": float(np.mean([x["manual_sam_iou"] for x in metric_rows if x["method"] == "Ours_guard_selected"])) if metric_rows else None,
        "manual_sam_dice_mean": float(np.mean([x["manual_sam_dice"] for x in metric_rows if x["method"] == "Ours_guard_selected"])) if metric_rows else None,
        "multi_seed_output_variance_mean_rgb": {
            method: float(np.mean([x["seed_output_variance"] for x in metric_rows if x["method"] == method]))
            for method in (*METHODS, "Ours_guard_selected")
        },
        "multi_seed_output_variance_max_rgb": {
            method: float(np.max([x["seed_output_variance"] for x in metric_rows if x["method"] == method]))
            for method in (*METHODS, "Ours_guard_selected")
        },
        "mask_source": sorted({x["mask_source"] for x in metric_rows}),
        "generation_backend": "G-lite stochastic proxy; no diffusion checkpoint/backend was available in this short cycle",
        "b3_previous_method": "not run on this fixed annotation showcase because there is no same-view paired previous output",
    }
    json_dump(root / "research" / "outputs" / "restoration" / "open3dhk_summary.json", summary)
    return summary


def write_run_commands(root: Path, model: dict) -> None:
    text = "# Short restoration cycle\n\n"
    text += "All commands run from the repository root with the isolated runtime.\n\n"
    text += "```powershell\n"
    text += "$CODEX_PYTHON = 'C:\\Users\\LENOVO\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe'\n"
    text += '& $CODEX_PYTHON research/restoration/prepare_showcase.py --count 24\n'
    text += '& $CODEX_PYTHON research/restoration/build_open3dhk_proxy_pairs.py --count 50 --seed 20260820\n'
    text += '& $CODEX_PYTHON research/restoration/run_sam_check.py --mode all --p0-limit 12 --runtime-dir .research_runtime\n'
    text += '& $CODEX_PYTHON research/restoration/run_restoration.py\n'
    text += "```\n\n"
    text += "## Runtime notes\n\n"
    text += "- Checkpoints load on CPU with explicit `map_location=cpu`; the supplied LoRA wrapper hard-codes CUDA deserialization, so the research script reproduces its state-dict loading logic independently.\n"
    text += "- B1 is a deterministic unsharp/detail-recovery proxy. B2 is a low-intensity stochastic proxy (`G_lite_stochastic_proxy`), not a diffusion model. A real LaMa/BrushNet/Fill backend was not available in this short cycle.\n"
    text += "- Every candidate uses the same soft alpha and exact input pixels outside the mask/dilated boundary.\n"
    text += "- Showcase masks include both manual annotation and SAM prediction; proxy PSNR/SSIM are not Open3DHK clean-GT claims.\n"
    (root / "research" / "outputs" / "restoration" / "run_commands.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    proxy_rows, model, proxy_summary = proxy_experiment(root)
    robustness_rows = [] if args.skip_robustness else robustness_experiment(root, model)
    showcase_summary = run_showcase(root, model)
    write_run_commands(root, model)
    summary = {"proxy": proxy_summary, "robustness_rows": len(robustness_rows), "showcase": showcase_summary, "guard_model": model}
    json_dump(root / "research" / "outputs" / "restoration" / "short_cycle_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
