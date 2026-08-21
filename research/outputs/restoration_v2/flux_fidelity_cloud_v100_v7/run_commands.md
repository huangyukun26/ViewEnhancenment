# FLUX fidelity quick6 commands

P0 post-processing from the committed FLUX.2 cloud output:

```bash
python research/restoration_v2/flux_fidelity.py --no-fill
```

Official FLUX.1-Fill-dev smoke/full run on the V100 (weights are not committed):

```bash
FLUX_FILL_MODEL_PATH=/home/vipuser/models/FLUX.1-Fill-dev \
  /home/vipuser/fluxenv/bin/python research/restoration_v2/flux_fidelity.py \
  --fill-model-path /home/vipuser/models/FLUX.1-Fill-dev --fill-steps 50 --cpu-offload
```

The formal cloud run used balanced Accelerate offload because the V100 host had
about 15 GiB system RAM. The model weights stay on the cloud host and are not
part of this repository. The completed run used 6 images, four seeds per image,
and 24 official Fill calls; the smoke run used `--fill-steps 8 --smoke-only`.

Source baseline: `/home/vipuser/ViewEnhancenment/research/outputs/restoration_v2/flux_constrained_cloud_v100`
Official implementation: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
