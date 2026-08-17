# First research experiment

Run from the repository root with the bundled Python runtime:

```powershell
& "$env:CODEX_PYTHON" research/audit_and_baseline.py
```

The script reads the existing s1/s2 data, writes a deterministic group-level
split and manifest under `research/data/`, and creates the first sanity-check
comparison under `research/outputs/`. It does not treat the existing generated
images as real ground truth. The strict E1 composition is:

```text
output = mask * pseudo_gt + (1 - mask) * input
```

The inferred mask is only a first proxy derived from input/pseudo-GT differences;
it is not a substitute for SAM/3D masks. E0 remains unavailable until a matching
InstructPix2Pix checkpoint is supplied.

## Supplied distortion-segmentation asset

The extracted annotation asset is under
`research/assets/distortion_segmentation_annotation_dataset/`. Its repository-
relative configuration is
`research/configs/distortion_segmentation_local.json`. It contains 220 image/
mask pairs, with the fixed 187/33 train/validation split. `checkpoint_best.pth`
is kept at the repository root and is referenced as the LoRA/SAM checkpoint;
the SAM base checkpoint is an external prerequisite.
