# Restoration-v2 reproducible commands

Commands below were run from the repository root with `G:\fishnet\.venv_cuda\Scripts\python.exe` (PyTorch 2.5.1+cu121, RTX 3060 6GB).

```powershell
# Build 30 proxy pairs for the manual inspection gate, then expand to 120.
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/build_proxy_pairs.py --count 30
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/build_proxy_pairs.py --count 120

# P1 real-model smoke on the first eight fixed partial-mask showcase rows.
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/run_real_backends.py --skip-proxy --showcase-limit 8

# Real-model proxy evaluation outputs (NAFNet + LaMa).
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/run_real_backends.py --skip-showcase --proxy-limit 120

# Same-building retrieval and SIFT/RANSAC alignment.
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/reference_retrieval.py --grid-limit 8

# Metrics, empty human-preference form, and figures.
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/evaluate.py
& 'G:\fishnet\.venv_cuda\Scripts\python.exe' research/restoration_v2/make_figures.py
```

The official NAFNet GoPro-width32 and LaMa big-lama checkpoints are local ignored caches and are intentionally not committed. BrushNet was not used: its official route requires an additional Stable Diffusion base plus BrushNet checkpoint; LaMa is the real pretrained fallback used in this 6GB environment. Generated per-sample images remain local; the committed summary CSVs and showcase grids are the review artifacts.
