#!/usr/bin/env python3
"""Audit the available window-view pairs and build a mask-constrained baseline.

The script intentionally uses only Pillow and NumPy so the first experiment can
run in the lightweight workspace without downloading a diffusion checkpoint.
Original data are read-only; all manifests, masks, metrics and visualizations are
written below ``research/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PROMPT = (
    "Hong Kong, buildings, clear, no restructured facade layouts, "
    "generated contents for distortion"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-scenes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-grid", type=int, default=12)
    return parser.parse_args()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def find_data_root(repo_root: Path) -> Path:
    candidates = [
        repo_root
        / "To_Yukun_3DModelImageryEnhancement"
        / "To_Yukun_3DModelImageryEnhancement"
        / "s7_make_datasets",
        repo_root / "s7_make_datasets",
    ]
    for candidate in candidates:
        if (candidate / "s1_trainingset").exists():
            return candidate
    raise FileNotFoundError("Could not locate s7_make_datasets/s1_trainingset")


def source_group_from_stem(stem: str, parent_name: str) -> str:
    """Remove augmentation and patch indices from the existing filename scheme."""
    tokens = stem.split("_")
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    group = "_".join(tokens).strip("_")
    return group or parent_name


def image_category(path: Path, data_root: Path) -> str:
    rel_parts = path.resolve().relative_to(data_root.resolve()).parts
    for bucket in ("image_patches_small", "image_pitches_large"):
        if bucket in rel_parts:
            index = rel_parts.index(bucket)
            if bucket == "image_patches_small" and index + 1 < len(rel_parts):
                return f"{bucket}/{rel_parts[index + 1]}"
            return bucket
    return rel_parts[0] if rel_parts else "unknown"


def scan_pairs(data_root: Path, repo_root: Path) -> tuple[list[dict], dict]:
    train_root = data_root / "s1_trainingset"
    pairs: list[dict] = []
    seen_generated: set[Path] = set()
    missing_generated: list[str] = []
    buckets = ["image_patches_small", "image_pitches_large"]

    for bucket in buckets:
        bucket_root = train_root / bucket
        if not bucket_root.exists():
            continue
        for original in sorted(bucket_root.rglob("*_ori.png")):
            generated = original.with_name(original.name.replace("_ori.png", "_generated.png"))
            seen_generated.add(generated.resolve())
            stem = original.stem[: -len("_ori")]
            group = source_group_from_stem(stem, original.parent.name)
            record = {
                "sample_id": f"{bucket}/{stem}",
                "bucket": bucket,
                "category": image_category(original, data_root),
                "source_group": group,
                "input_path": relative(original, repo_root),
                "output_path": relative(generated, repo_root),
                "input_exists": True,
                "output_exists": generated.exists(),
            }
            if not generated.exists():
                missing_generated.append(record["sample_id"])
            else:
                pairs.append(record)

        for generated in bucket_root.rglob("*_generated.png"):
            if generated.resolve() not in seen_generated:
                missing_generated.append(f"orphan_generated:{relative(generated, repo_root)}")

    return pairs, {
        "candidate_original_files": len(pairs) + len(missing_generated),
        "paired_files": len(pairs),
        "missing_or_orphan_files": len(missing_generated),
        "missing_or_orphan_examples": missing_generated[:20],
    }


def resolve_declared_path(raw: str, repo_root: Path, basename_index: dict[str, list[Path]]) -> tuple[bool, str]:
    path = Path(raw)
    if path.exists():
        return True, "declared_path"
    candidates = basename_index.get(path.name, [])
    if len(candidates) == 1:
        return True, "unique_basename"
    if len(candidates) > 1:
        return False, "ambiguous_basename"
    return False, "missing"


def audit_json_manifests(data_root: Path, repo_root: Path) -> list[dict]:
    image_files = list((data_root / "s1_trainingset").rglob("*.png"))
    image_files += list((data_root / "s2_large_scale_dataset").rglob("*.png"))
    basename_index: dict[str, list[Path]] = defaultdict(list)
    for image in image_files:
        basename_index[image.name].append(image)

    summaries: list[dict] = []
    json_root = data_root / "s2_large_scale_dataset"
    for manifest in sorted(json_root.rglob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            summaries.append({"manifest": relative(manifest, repo_root), "parse_error": str(exc)})
            continue
        rows = payload if isinstance(payload, list) else []
        input_found = 0
        output_found = 0
        input_modes = Counter()
        output_modes = Counter()
        for row in rows:
            if not isinstance(row, dict):
                continue
            input_raw = str(row.get("input_image", ""))
            output_raw = str(row.get("output_image", ""))
            input_ok, input_mode = resolve_declared_path(input_raw, repo_root, basename_index)
            output_ok, output_mode = resolve_declared_path(output_raw, repo_root, basename_index)
            input_found += int(input_ok)
            output_found += int(output_ok)
            input_modes[input_mode] += 1
            output_modes[output_mode] += 1
        summaries.append(
            {
                "manifest": relative(manifest, repo_root),
                "entries": len(rows),
                "input_resolved": input_found,
                "output_resolved": output_found,
                "input_resolution_modes": dict(input_modes),
                "output_resolution_modes": dict(output_modes),
            }
        )
    return summaries


def canonical_split(records: list[dict], test_scenes: int, seed: int) -> dict[str, str]:
    groups = sorted({record["source_group"] for record in records})

    def rank(group: str) -> str:
        return hashlib.sha1(f"{seed}:{group}".encode("utf-8")).hexdigest()

    ranked = sorted(groups, key=rank)
    if not ranked:
        return {}
    n_test = min(test_scenes, max(1, int(round(len(ranked) * 0.2))))
    n_test = min(n_test, len(ranked) - 1) if len(ranked) > 1 else 1
    test = set(ranked[:n_test])
    remaining = ranked[n_test:]
    n_val = max(1, int(round(len(remaining) * 0.2))) if remaining else 0
    val = set(remaining[:n_val])
    return {
        group: ("test" if group in test else "val" if group in val else "train")
        for group in groups
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_to(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(array, mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """A lightweight global SSIM sanity check, not a replacement for benchmark SSIM."""
    x = a.astype(np.float64).mean(axis=2)
    y = b.astype(np.float64).mean(axis=2)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))


def infer_mask(input_rgb: np.ndarray, pseudo_rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
    diff = np.abs(pseudo_rgb.astype(np.float32) - input_rgb.astype(np.float32)).mean(axis=2)
    nonzero = diff[diff > 0]
    if nonzero.size == 0:
        return np.zeros(diff.shape, dtype=np.uint8), 0.0, 0.0
    threshold = max(18.0, float(np.percentile(nonzero, 80)))
    binary = (diff >= threshold).astype(np.uint8) * 255
    # Close small holes and connect thin facade changes using Pillow filters.
    mask_image = Image.fromarray(binary, mode="L").filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
    mask = np.asarray(mask_image, dtype=np.uint8)
    return mask, float(diff.mean()), threshold


def classify_severity(diff_mean: float, q1: float, q2: float) -> str:
    if diff_mean <= q1:
        return "light"
    if diff_mean <= q2:
        return "medium"
    return "severe"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fit_tile(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (width, height), "white")
    fitted = ImageOps.contain(image, (width - 8, height - 8))
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def make_grid(rows: list[dict], output_path: Path, repo_root: Path, sample_dir: Path) -> None:
    columns = ["Input", "Pseudo-GT / FLUX", "Inferred mask", "E1 strict mask composite", "Change map"]
    tile_w, tile_h, label_h = 256, 220, 28
    top_h = 44
    grid = Image.new("RGB", (tile_w * len(columns), top_h + len(rows) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for col, label in enumerate(columns):
        x = col * tile_w
        draw.text((x + 6, 8), label, fill="black", font=font)
    for row_index, record in enumerate(rows):
        y = top_h + row_index * (tile_h + label_h)
        sample_input = load_rgb(repo_root / record["input_path"])
        sample_pseudo = load_rgb(repo_root / record["output_path"])
        if sample_pseudo.shape[:2] != sample_input.shape[:2]:
            sample_pseudo = resize_to(sample_pseudo, sample_input.shape[1], sample_input.shape[0])
        mask, _, _ = infer_mask(sample_input, sample_pseudo)
        composite = np.where(mask[..., None] > 0, sample_pseudo, sample_input).astype(np.uint8)
        diff = np.abs(sample_pseudo.astype(np.float32) - sample_input.astype(np.float32)).mean(axis=2)
        diff = np.clip(diff / max(float(diff.max()), 1.0) * 255.0, 0, 255).astype(np.uint8)
        diff_rgb = np.repeat(diff[..., None], 3, axis=2)
        tiles = [
            Image.fromarray(sample_input),
            Image.fromarray(sample_pseudo),
            Image.fromarray(mask).convert("RGB"),
            Image.fromarray(composite),
            Image.fromarray(diff_rgb),
        ]
        for col, tile in enumerate(tiles):
            x = col * tile_w
            grid.paste(fit_tile(tile, tile_w, tile_h), (x, y))
        row_label = f"{record['sample_id']} | {record['severity']} | mask={record['mask_ratio']:.1%}"
        draw.text((6, y + tile_h + 5), row_label[:180], fill="black", font=font)
        sample_output = sample_dir / record["sample_id"].replace("/", "__")
        sample_output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sample_input).save(sample_output / "input.png")
        Image.fromarray(sample_pseudo).save(sample_output / "pseudo_gt_flux.png")
        Image.fromarray(mask).save(sample_output / "inferred_mask.png")
        Image.fromarray(composite).save(sample_output / "e1_strict_mask_composite.png")
        Image.fromarray(diff_rgb).save(sample_output / "change_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = find_data_root(repo_root)
    research_root = repo_root / "research"
    data_output = research_root / "data"
    output_root = research_root / "outputs"
    sample_output = output_root / "baseline_samples"
    data_output.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs, pair_audit = scan_pairs(data_root, repo_root)
    split_by_group = canonical_split(pairs, args.test_scenes, args.seed)
    json_audit = audit_json_manifests(data_root, repo_root)
    for record in pairs:
        record["split"] = split_by_group.get(record["source_group"], "unknown")

    valid_records: list[dict] = []
    invalid_records: list[dict] = []
    diff_values: list[float] = []
    for record in pairs:
        input_path = repo_root / record["input_path"]
        output_path = repo_root / record["output_path"]
        try:
            input_rgb = load_rgb(input_path)
            pseudo_rgb = load_rgb(output_path)
            record["input_width"], record["input_height"] = input_rgb.shape[1], input_rgb.shape[0]
            record["output_width"], record["output_height"] = pseudo_rgb.shape[1], pseudo_rgb.shape[0]
            record["shape_match"] = input_rgb.shape[:2] == pseudo_rgb.shape[:2]
            if not record["shape_match"]:
                pseudo_rgb = resize_to(pseudo_rgb, input_rgb.shape[1], input_rgb.shape[0])
            mask, diff_mean, mask_threshold = infer_mask(input_rgb, pseudo_rgb)
            record["diff_mean"] = diff_mean
            record["mask_threshold"] = mask_threshold
            record["mask_ratio"] = float(np.mean(mask > 0))
            record["input_pseudo_psnr"] = psnr(input_rgb, pseudo_rgb)
            record["input_pseudo_ssim_global"] = ssim_global(input_rgb, pseudo_rgb)
            composite = np.where(mask[..., None] > 0, pseudo_rgb, input_rgb).astype(np.uint8)
            outside = mask == 0
            outside_delta = np.abs(composite.astype(np.int16) - input_rgb.astype(np.int16))[outside]
            record["e1_pseudo_psnr"] = psnr(composite, pseudo_rgb)
            record["e1_pseudo_ssim_global"] = ssim_global(composite, pseudo_rgb)
            record["e1_mask_outside_max_abs"] = int(outside_delta.max()) if outside_delta.size else 0
            record["e1_mask_outside_mean_abs"] = float(outside_delta.mean()) if outside_delta.size else 0.0
            record["identity_candidate"] = False
            valid_records.append(record)
            diff_values.append(diff_mean)
        except Exception as exc:  # keep the audit running on corrupt files
            record["error"] = repr(exc)
            invalid_records.append(record)

    if diff_values:
        q1, q2 = np.percentile(np.asarray(diff_values), [33, 66]).tolist()
        q20 = float(np.percentile(np.asarray(diff_values), 20))
    else:
        q1 = q2 = q20 = 0.0
    for record in valid_records:
        record["severity"] = classify_severity(record["diff_mean"], q1, q2)
        category = record["category"].lower()
        record["identity_candidate"] = bool(
            "clear" in category and "noimagine" in category and record["diff_mean"] <= q1
        )

    manifest_fields = [
        "sample_id", "bucket", "category", "source_group", "split", "input_path", "output_path",
        "input_exists", "output_exists", "shape_match", "input_width", "input_height",
        "output_width", "output_height", "severity", "identity_candidate", "diff_mean",
        "mask_ratio", "mask_threshold", "input_pseudo_psnr", "input_pseudo_ssim_global",
        "e1_pseudo_psnr", "e1_pseudo_ssim_global", "e1_mask_outside_max_abs",
        "e1_mask_outside_mean_abs",
    ]
    write_csv(data_output / "canonical_manifest.csv", valid_records, manifest_fields)
    write_csv(data_output / "invalid_pairs.csv", invalid_records, ["sample_id", "input_path", "output_path", "error"])

    split_payload = {
        "version": 1,
        "seed": args.seed,
        "grouping_rule": "remove leading augmentation index and trailing patch index from paired filename",
        "group_counts": dict(Counter(split_by_group.values())),
        "sample_counts": dict(Counter(record["split"] for record in valid_records)),
        "groups": split_by_group,
    }
    json_dump(data_output / "canonical_split.json", split_payload)

    audit_payload = {
        "data_root": relative(data_root, repo_root),
        "pair_audit": pair_audit,
        "valid_pairs": len(valid_records),
        "invalid_pairs": len(invalid_records),
        "source_groups": len(split_by_group),
        "split": split_payload,
        "json_manifests": json_audit,
        "severity_thresholds_diff_mean": {"light_max": q1, "medium_max": q2, "identity_q20": q20},
        "e0_status": "unavailable: no InstructPix2Pix checkpoint is present in the repository",
        "e1_status": "implemented: output = mask * pseudo_gt + (1 - mask) * input",
        "pseudo_gt_warning": "Existing generated images are FLUX-like pseudo-GT and can change structure; metrics are sanity checks only.",
    }
    json_dump(data_output / "data_audit.json", audit_payload)
    json_dump(
        output_root / "baseline_summary.json",
        {
            "E0": "unavailable; checkpoint absent",
            "E1": "strict mask composition using an inferred difference mask",
            "mask_outside_max_abs": max((record["e1_mask_outside_max_abs"] for record in valid_records), default=None),
            "mask_outside_mean_abs": float(np.mean([record["e1_mask_outside_mean_abs"] for record in valid_records])) if valid_records else None,
            "paired_metrics_are_not_GT_metrics": True,
            "sample_count": len(valid_records),
        },
    )
    metric_fields = [
        "sample_id", "source_group", "split", "category", "severity", "identity_candidate",
        "mask_ratio", "input_pseudo_psnr", "input_pseudo_ssim_global", "e1_pseudo_psnr",
        "e1_pseudo_ssim_global", "e1_mask_outside_max_abs", "e1_mask_outside_mean_abs",
    ]
    write_csv(output_root / "metrics.csv", valid_records, metric_fields)

    grid_records = [record for record in valid_records if record["split"] == "test"]
    grid_records.sort(key=lambda record: (record["severity"], not record["identity_candidate"], record["sample_id"]))
    grid_records = grid_records[: args.max_grid]
    make_grid(grid_records, output_root / "e0_e1_comparison_grid.png", repo_root, sample_output)

    print(json.dumps({
        "data_root": relative(data_root, repo_root),
        "paired_files": pair_audit["paired_files"],
        "valid_pairs": len(valid_records),
        "invalid_pairs": len(invalid_records),
        "source_groups": len(split_by_group),
        "split_groups": split_payload["group_counts"],
        "split_samples": split_payload["sample_counts"],
        "json_manifests": json_audit,
        "grid_samples": len(grid_records),
        "grid": relative(output_root / "e0_e1_comparison_grid.png", repo_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
