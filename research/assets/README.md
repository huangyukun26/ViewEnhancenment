# Distortion segmentation annotation asset

This directory is the extracted, audit-friendly form of the local annotation
archive. The original archive is intentionally not tracked by Git.

- `distortion_segmentation_annotation_dataset/for_segmentation/images/`: 220
  900x900 RGBA window-view images (alpha is constant 255).
- `distortion_segmentation_annotation_dataset/for_segmentation/mask_for_sam/`:
  220 same-name binary masks with values 0 and 1.
- `distortion_segmentation_annotation_dataset/for_segmentation/ImageSet/`:
  fixed `train.txt` (187 names) and `val.txt` (33 names); there is no test list.
- `../configs/distortion_segmentation_local.json`: repository-relative paths and
  the checkpoint/model settings used by the supplied scripts.

The supplied `checkpoint_best.pth` is a project checkpoint intended for the
LoRA/SAM path (`pred_view_distortion.py --lora_ckpt`). The matching SAM ViT-B
base checkpoint is now available at the repository root as
`sam_vit_b_01ec64.pth` and should be supplied with `--ckpt`.
