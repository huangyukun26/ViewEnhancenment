# Short restoration cycle

All commands run from the repository root with the isolated runtime.

```powershell
$CODEX_PYTHON = 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $CODEX_PYTHON research/restoration/prepare_showcase.py --count 24
& $CODEX_PYTHON research/restoration/build_open3dhk_proxy_pairs.py --count 50 --seed 20260820
& $CODEX_PYTHON research/restoration/run_sam_check.py --mode all --p0-limit 12 --runtime-dir .research_runtime
& $CODEX_PYTHON research/restoration/run_restoration.py
```

## Runtime notes

- Checkpoints load on CPU with explicit `map_location=cpu`; the supplied LoRA wrapper hard-codes CUDA deserialization, so the research script reproduces its state-dict loading logic independently.
- B1 is a deterministic unsharp/detail-recovery proxy. B2 is a low-intensity stochastic proxy (`G_lite_stochastic_proxy`), not a diffusion model. A real LaMa/BrushNet/Fill backend was not available in this short cycle.
- Every candidate uses the same soft alpha and exact input pixels outside the mask/dilated boundary.
- Showcase masks include both manual annotation and SAM prediction; proxy PSNR/SSIM are not Open3DHK clean-GT claims.
