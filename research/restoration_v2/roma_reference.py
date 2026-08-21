#!/usr/bin/env python3
"""Optional official RoMa v2 dense matching probe.

The current CUDA environment is Python 3.9 while the upstream package is
published for newer Python versions.  The loader below injects postponed
annotation evaluation in memory only; it does not modify the cached upstream
checkout.  Model weights are downloaded by the official RoMa code into the
user torch cache and are never committed.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROMA_SRC = ROOT / ".research_cache" / "RoMaV2" / "src"


class _FutureLoader(importlib.machinery.SourceFileLoader):
    def get_data(self, path):
        data = super().get_data(path)
        return b"from __future__ import annotations\n" + data


class _FutureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("romav2"):
            return None
        parts = fullname.split(".")
        base = ROMA_SRC / "romav2" / Path(*parts[1:])
        module_file = base.with_suffix(".py")
        package_file = base / "__init__.py"
        if module_file.exists():
            return importlib.util.spec_from_file_location(fullname, module_file, loader=_FutureLoader(fullname, str(module_file)))
        if package_file.exists():
            return importlib.util.spec_from_file_location(fullname, package_file, loader=_FutureLoader(fullname, str(package_file)), submodule_search_locations=[str(base)])
        return None


def _load_model():
    sys.meta_path.insert(0, _FutureFinder())
    import torch

    # RoMa v2 explicitly asserts the highest setting at inference time.
    torch.set_float32_matmul_precision("highest")
    from romav2 import RoMaV2

    cfg = RoMaV2.Cfg(setting="fast")
    return RoMaV2(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch

    model = _load_model()
    with torch.inference_mode():
        preds = model.match(args.target, args.reference)
    result = {
        "status": "completed",
        "setting": "fast (512x512, batch=1)",
        "device": str(next(model.parameters()).device),
        "target": args.target,
        "reference": args.reference,
        "warp_shape": list(preds["warp_AB"].shape),
        "overlap_shape": list(preds["overlap_AB"].shape),
        "overlap_mean": float(preds["overlap_AB"].mean().item()),
        "overlap_median": float(preds["overlap_AB"].median().item()),
        "bidirectional": bool(preds.get("warp_BA") is not None),
        "official_repo": "https://github.com/Parskatt/RoMaV2",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
