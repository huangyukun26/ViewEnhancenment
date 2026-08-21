# Constrained FLUX quick6 commands

The current committed run is an honest blocker run because no local FLUX.2-klein-4B weights, HF token, or BFL API key were available. It does not label identity passthrough as FLUX output.

```powershell
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' research/restoration_v2/flux_constrained.py --quick6 research/data/flux_quick6.csv
```

With a locally provisioned official checkpoint (do not commit weights):

```powershell
$env:KLEIN_4B_MODEL_PATH='D:\path\to\FLUX.2-klein-4B'
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' research/restoration_v2/flux_constrained.py --quick6 research/data/flux_quick6.csv --model-path $env:KLEIN_4B_MODEL_PATH
```

Official references: [FLUX.2 repository](https://github.com/black-forest-labs/flux2), [FLUX.2-klein-4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
