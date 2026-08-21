# FLUX fidelity quick6 commands

P0 post-processing from the committed FLUX.2 cloud output:

```bash
python research/restoration_v2/flux_fidelity.py --no-fill
```

Official FLUX.1-Fill-dev smoke/full run on the V100 (weights are not committed):

```bash
FLUX_FILL_MODEL_PATH=/home/vipuser/models/FLUX.1-Fill-dev \
+  /home/vipuser/fluxenv/bin/python research/restoration_v2/flux_fidelity.py \
+  --fill-model-path /home/vipuser/models/FLUX.1-Fill-dev --fill-steps 50
```

Source baseline: `/home/vipuser/ViewEnhancenment/research/outputs/restoration_v2/flux_constrained_cloud_v100`
Official implementation: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
