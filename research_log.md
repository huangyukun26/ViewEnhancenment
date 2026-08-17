# Research log

## 2026-08-17 - E0/E1 data audit, canonical split, and mask-composition sanity check

- 实验假设：显式的局部 mask 合成可以让未编辑区域保持完全不变；轻量的确定性约束应先于无 mask 的全图生成式编辑验证。
- 修改内容：新增 `research/audit_and_baseline.py`；扫描现有 s1 配对图和 s2 JSON；按源场景/建筑标识建立 deterministic canonical split；从现有 `*_ori`/`*_generated` 差异估计首版 proxy mask；生成 E1 严格融合、metrics、manifest 和对比图。
- 失败或成功现象：E0 不可复现，仓库中没有可用的 InstructPix2Pix checkpoint；s2 large-scale JSON 中的 `/mnt/e/...` 路径无法直接访问，需依赖本地 basename 复核。
- 原因判断：当前训练脚本只读取 input/output/prompt 三列并以全图 latent MSE 训练，没有显式区域 mask、mask 外保真或结构损失；这与 InstructPix2Pix 的全图编辑范式一致，但不满足当前首要约束。
- 结果变化：实际找到 382 个有效 ori/generated 配对、0 个损坏配对、80 个源场景组；canonical split 为 train/val/test = 51/13/16 组、262/53/67 样本，且 0 个源场景跨 split 泄漏。平均 mask 占比为 7.59%，E1 mask 外最大绝对像素变化为 `0`、平均为 `0.0`。输入与 pseudo-GT 的平均 PSNR sanity check 为 30.84 dB，E1 与 pseudo-GT 为 38.23 dB；该数值只验证融合行为，不代表真实 GT 提升。
- 数据 blocker：large-scale `training.json` 声称 645 条，但原始 `/mnt/e/...` 声明路径中没有可直接访问的路径；按本地 basename 仅能唯一解析 50 个 input、49 个 output，另有歧义/缺失项。当前可用测试组只有 16 个，无法凑满目标的 80–100 个独立测试源场景。
- 是否保留：保留。现有 FLUX-like 生成图仅作为 pseudo-GT/候选参考，所有 paired 指标只作 sanity check，不能解释为真实 GT 提升。
- 下一步：根据 canonical test split 统计退化强度；优先补充严格对齐的 synthetic paired data，并实现一个可运行的确定性 restoration baseline；严重缺失再接入 BrushNet 或 FLUX.1 Fill。

### 参考论文与官方实现

- InstructPix2Pix: https://arxiv.org/abs/2211.09800
- Hugging Face Diffusers InstructPix2Pix training guide: https://huggingface.co/docs/diffusers/v0.34.0/training/instructpix2pix
- NAFNet / Simple Baselines for Image Restoration: https://arxiv.org/abs/2204.04676 and https://github.com/megvii-research/NAFNet
- BrushNet official implementation: https://github.com/TencentARC/BrushNet
- FLUX.1 Fill official documentation: https://docs.bfl.ml/flux_1_fill
