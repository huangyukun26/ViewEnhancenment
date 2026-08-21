# Constrained FLUX quick6 commands

The original local run in `research/outputs/restoration_v2/flux_constrained/` is an honest blocker run because no local FLUX.2-klein-4B weights, HF token, or BFL API key were available. It does not label identity passthrough as FLUX output. The cloud run below is the completed real-model quick6 run.

```powershell
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' research/restoration_v2/flux_constrained.py --quick6 research/data/flux_quick6.csv
```

For a remote run while preserving the blocker artifacts, add `--output-dir research/outputs/restoration_v2/flux_constrained_cloud_v100`.

Cloud V100 run used for the committed quick6 results (official weights are not committed):

```bash
FLUX_FULL_GPU=1 KLEIN_4B_MODEL_PATH=/home/vipuser/models/FLUX.2-klein-4B \
  /home/vipuser/fluxenv/bin/python research/restoration_v2/flux_constrained.py \
  --quick6 research/data/flux_quick6.csv \
  --model-path /home/vipuser/models/FLUX.2-klein-4B \
  --steps 4 \
  --output-dir research/outputs/restoration_v2/flux_constrained_cloud_v100
```

With a locally provisioned official checkpoint (do not commit weights):

```powershell
$env:KLEIN_4B_MODEL_PATH='D:\path\to\FLUX.2-klein-4B'
& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' research/restoration_v2/flux_constrained.py --quick6 research/data/flux_quick6.csv --model-path $env:KLEIN_4B_MODEL_PATH
```

Official references: [FLUX.2 repository](https://github.com/black-forest-labs/flux2), [FLUX.2-klein-4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
