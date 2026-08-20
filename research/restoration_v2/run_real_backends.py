#!/usr/bin/env python3
"""Run real pretrained restoration backends for restoration-v2.

This script deliberately keeps model inference separate from routing and
evaluation.  The two backends are the official NAFNet GoPro model and the
official LaMa ``big-lama`` checkpoint.  Neither backend is a hand-written
filter or a toy proxy.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "restoration_v2"))

from common import (  # noqa: E402
    dilate,
    json_dump,
    load_mask,
    load_rgb,
    read_csv,
    save_mask,
    save_rgb,
    soft_composite,
    write_csv,
)


NAF_REPO = ROOT / ".research_cache" / "NAFNet"
NAF_CKPT = ROOT / ".model_cache" / "NAFNet-GoPro-width32.pth"
LAMA_REPO = ROOT / ".research_cache" / "lama"
LAMA_CONFIG = ROOT / ".model_cache" / "big-lama" / "big-lama" / "config.yaml"
LAMA_CKPT = ROOT / ".model_cache" / "big-lama" / "big-lama" / "models" / "best.ckpt"
OUT_ROOT = ROOT / "research" / "outputs" / "restoration_v2"


def _configure_imports() -> None:
    sys.path.insert(0, str(NAF_REPO))
    sys.path.insert(0, str(LAMA_REPO))


def _to_tensor(rgb: np.ndarray):
    import torch

    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def _to_rgb(tensor) -> np.ndarray:
    value = tensor.detach().float().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return np.rint(value * 255.0).astype(np.uint8)


class NAFNetBackend:
    name = "R1_NAFNet"

    def __init__(self, device: str = "cuda") -> None:
        _configure_imports()
        import torch
        from basicsr.models.archs.NAFNet_arch import NAFNet

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = NAFNet(width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
        state = torch.load(NAF_CKPT, map_location="cpu")
        params = state.get("params", state)
        self.model.load_state_dict(params, strict=True)
        self.model.eval().to(self.device)

    def run(self, rgb: np.ndarray) -> np.ndarray:
        tensor = _to_tensor(rgb).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
        return _to_rgb(output)


class LaMaBackend:
    name = "R2_LaMa"

    def __init__(self, device: str = "cuda", context: int = 96, max_side: int = 512) -> None:
        _configure_imports()
        import torch
        from omegaconf import OmegaConf
        from saicinpainting.training.trainers import load_checkpoint

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.context = int(context)
        self.max_side = int(max_side)
        config = OmegaConf.load(str(LAMA_CONFIG))
        config.training_model.predict_only = True
        config.visualizer.kind = "noop"
        self.model = load_checkpoint(config, str(LAMA_CKPT), strict=False, map_location="cpu")
        self.model.freeze()
        self.model.to(self.device).eval()

    @staticmethod
    def _bbox(mask: np.ndarray, context: int) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        return (
            max(0, int(xs.min()) - context),
            max(0, int(ys.min()) - context),
            min(mask.shape[1], int(xs.max()) + 1 + context),
            min(mask.shape[0], int(ys.max()) + 1 + context),
        )

    def run(self, rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
        if not np.any(mask):
            return rgb.copy(), {"crop": [0, 0, rgb.shape[1], rgb.shape[0]], "resized": False}
        x0, y0, x1, y1 = self._bbox(mask, self.context)
        crop = rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        original_hw = crop.shape[:2]
        scale = min(1.0, self.max_side / max(crop.shape[:2]))
        if scale < 1.0:
            new_w = max(32, int(round(crop.shape[1] * scale)))
            new_h = max(32, int(round(crop.shape[0] * scale)))
            crop = np.asarray(Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
            crop_mask = np.asarray(Image.fromarray((crop_mask.astype(np.uint8) * 255)).resize((new_w, new_h), Image.Resampling.NEAREST), dtype=np.uint8) > 127
        h, w = crop.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), mode="constant")
        batch = {
            "image": _to_tensor(crop).to(self.device),
            "mask": self.torch.from_numpy(crop_mask.astype(np.float32))[None, None].to(self.device),
        }
        with self.torch.no_grad():
            result = self.model(batch)["inpainted"][:, :, :h, :w]
        repaired = _to_rgb(result)
        if scale < 1.0:
            repaired = np.asarray(Image.fromarray(repaired).resize((original_hw[1], original_hw[0]), Image.Resampling.LANCZOS), dtype=np.uint8)
        full_candidate = rgb.copy()
        full_candidate[y0:y1, x0:x1] = repaired
        return full_candidate, {
            "crop": [x0, y0, x1, y1],
            "resized": bool(scale < 1.0),
            "scale": float(scale),
            "max_side": self.max_side,
        }


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _manual_mask(row: dict[str, str]) -> np.ndarray:
    return load_mask(_resolve(row["annotation_mask_path"]))


def _sam_mask(row: dict[str, str]) -> np.ndarray:
    path = _resolve(row["sam_mask_path"])
    if not path.exists():
        return _manual_mask(row)
    return load_mask(path)


def _run_one(
    row: dict[str, str],
    mask_source: str,
    input_path: Path,
    output_dir: Path,
    r1: NAFNetBackend,
    r2: LaMaBackend,
) -> dict[str, str]:
    rgb = load_rgb(input_path)
    mask = _manual_mask(row) if mask_source == "manual" else _sam_mask(row)
    if mask.shape != rgb.shape[:2]:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 127
    sample_dir = output_dir / f"{row['showcase_index']}_{mask_source}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(sample_dir / "input.png", rgb)
    save_mask(sample_dir / "mask.png", mask)

    start = time.perf_counter()
    r1_candidate = r1.run(rgb)
    r1_raw_time = time.perf_counter() - start
    r1_output, allowed = soft_composite(rgb, r1_candidate, mask, dilate_px=3, feather=1.5)
    save_rgb(sample_dir / "R1_NAFNet.png", r1_output)

    start = time.perf_counter()
    r2_candidate, crop_info = r2.run(rgb, mask)
    r2_raw_time = time.perf_counter() - start
    r2_output, _ = soft_composite(rgb, r2_candidate, mask, dilate_px=3, feather=1.5)
    save_rgb(sample_dir / "R2_LaMa.png", r2_output)

    outside = ~allowed
    return {
        "sample_id": row["sample_id"],
        "showcase_index": row["showcase_index"],
        "mask_source": mask_source,
        "input_path": str(input_path.relative_to(ROOT)),
        "mask_path": str((sample_dir / "mask.png").relative_to(ROOT)),
        "R1_NAFNet_path": str((sample_dir / "R1_NAFNet.png").relative_to(ROOT)),
        "R2_LaMa_path": str((sample_dir / "R2_LaMa.png").relative_to(ROOT)),
        "r1_runtime_sec": f"{r1_raw_time:.4f}",
        "r2_runtime_sec": f"{r2_raw_time:.4f}",
        "r2_crop": str(crop_info.get("crop", [])),
        "r2_resized": str(crop_info.get("resized", False)),
        "mask_ratio": f"{mask.mean():.8f}",
        "outside_allowed_max": str(int(np.abs(r1_output.astype(np.int16) - rgb.astype(np.int16))[outside].max()) if np.any(outside) else 0),
    }


def run_showcase(args, r1: NAFNetBackend, r2: LaMaBackend) -> list[dict[str, str]]:
    rows = read_csv(ROOT / "research" / "data" / "open3dhk_showcase.csv")
    candidates = [r for r in rows if 0.03 < float(r["manual_mask_ratio"]) < 0.95]
    if args.showcase_all:
        selected = rows
    else:
        selected = candidates[: args.showcase_limit]
    result_rows = []
    out_dir = OUT_ROOT / "real_results" / "showcase"
    for row in selected:
        input_path = _resolve(row["image_path"])
        for source in ("manual", "sam"):
            result_rows.append(_run_one(row, source, input_path, out_dir, r1, r2))
    return result_rows


def run_proxy(args, r1: NAFNetBackend, r2: LaMaBackend) -> list[dict[str, str]]:
    rows = read_csv(ROOT / "research" / "data" / "restoration_v2_proxy_pairs.csv")
    selected = rows[: args.proxy_limit] if args.proxy_limit > 0 else rows
    result_rows = []
    out_dir = OUT_ROOT / "real_results" / "proxy"
    for row in selected:
        rgb = load_rgb(_resolve(row["distorted_path"]))
        mask = load_mask(_resolve(row["mask_path"]))
        sample_dir = out_dir / row["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(sample_dir / "input.png", rgb)
        save_mask(sample_dir / "mask.png", mask)
        start = time.perf_counter()
        r1_candidate = r1.run(rgb)
        r1_time = time.perf_counter() - start
        r1_output, allowed = soft_composite(rgb, r1_candidate, mask, dilate_px=3, feather=1.5)
        save_rgb(sample_dir / "R1_NAFNet.png", r1_output)
        start = time.perf_counter()
        r2_candidate, crop_info = r2.run(rgb, mask)
        r2_time = time.perf_counter() - start
        r2_output, _ = soft_composite(rgb, r2_candidate, mask, dilate_px=3, feather=1.5)
        save_rgb(sample_dir / "R2_LaMa.png", r2_output)
        result_rows.append({
            "sample_id": row["sample_id"],
            "split": row["split"],
            "degradation": row["degradation_type"],
            "input_path": row["distorted_path"],
            "target_path": row["clean_path"],
            "mask_path": row["mask_path"],
            "R1_NAFNet_path": str((sample_dir / "R1_NAFNet.png").relative_to(ROOT)),
            "R2_LaMa_path": str((sample_dir / "R2_LaMa.png").relative_to(ROOT)),
            "r1_runtime_sec": f"{r1_time:.4f}",
            "r2_runtime_sec": f"{r2_time:.4f}",
            "r2_crop": str(crop_info.get("crop", [])),
            "r2_resized": str(crop_info.get("resized", False)),
            "outside_allowed_max": str(int(np.abs(r1_output.astype(np.int16) - rgb.astype(np.int16))[~allowed].max()) if np.any(~allowed) else 0),
        })
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--showcase-limit", type=int, default=8, help="number of partial-mask showcase samples for the P1 smoke run")
    parser.add_argument("--showcase-all", action="store_true")
    parser.add_argument("--proxy-limit", type=int, default=0, help="0 means all proxy pairs")
    parser.add_argument("--skip-showcase", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    r1 = NAFNetBackend(args.device)
    r2 = LaMaBackend(args.device)
    rows: list[dict[str, str]] = []
    if not args.skip_showcase:
        rows.extend(run_showcase(args, r1, r2))
    if not args.skip_proxy:
        rows.extend(run_proxy(args, r1, r2))
    fields = sorted({key for row in rows for key in row})
    if args.skip_proxy and not args.skip_showcase:
        manifest_name = "real_backend_manifest_showcase.csv"
    elif args.skip_showcase and not args.skip_proxy:
        manifest_name = "real_backend_manifest_proxy.csv"
    else:
        manifest_name = "real_backend_manifest.csv"
    write_csv(OUT_ROOT / manifest_name, rows, fields)
    json_dump(OUT_ROOT / "real_backend_status.json", {
        "status": "completed",
        "device": str(r1.device),
        "backends": {
            "R1_NAFNet": {
                "official_repo": "https://github.com/megvii-research/NAFNet",
                "repo_commit": "2b4af71ebe098a92a75910c233a3965a3e93ede4",
                "checkpoint": str(NAF_CKPT.relative_to(ROOT)),
                "checkpoint_bytes": NAF_CKPT.stat().st_size,
                "configuration": "width32 GoPro NAFNet",
            },
            "R2_LaMa": {
                "official_repo": "https://github.com/advimman/lama",
                "repo_commit": "786f5936b27fb3dacd2b1ad799e4de968ea697e7",
                "checkpoint": str(LAMA_CKPT.relative_to(ROOT)),
                "checkpoint_bytes": LAMA_CKPT.stat().st_size,
                "configuration": "big-lama, context crop, max_side=512",
                "stochastic": False,
            },
        },
        "brushnet": {
            "status": "not_used",
            "reason": "official BrushNet requires an additional SD base and large checkpoint; LaMa was used as the allowed real inpainting fallback on the 6GB GPU.",
            "official_repo": "https://github.com/TencentARC/BrushNet",
        },
        "result_rows": len(rows),
    })
    print(f"wrote {len(rows)} result rows to {OUT_ROOT / manifest_name}")


if __name__ == "__main__":
    main()
