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

## 2026-08-20 - P0-P3 Fidelity-Guarded Masked Restoration 短周期

- 实验假设：SAM 定位后，确定性修复与低强度生成候选通过边界/颜色/结构保真筛选，并只在允许区域内合成，可以减少 Open3DHK 的结构漂移。
- 修改内容：确认 `sam_vit_b_01ec64.pth`（375,042,383 bytes，314 个 state keys）和 `checkpoint_best.pth`（15,784,103 bytes，173 个 state keys）可在 CPU 加载；新增独立的 SAM CPU 加载与推理脚本、固定 24 张 showcase、50 对域内 proxy pairs、B0 identity、B1 确定性锐化 proxy、G-lite 低强度随机 proxy、guard candidate selection 和鲁棒性脚本。SAM 原 wrapper 的 CUDA 反序列化问题通过研究脚本显式 `map_location=cpu` 绕开，未修改前人源码。
- 失败或成功现象：P0 的 12 张样本 SAM 平均 IoU=`0.81896`、Dice=`0.82573`，24 张固定 showcase 的人工标注-SAM 平均 IoU=`0.90313`，因此进入修复实验。早先从旧 s1 `ori` 图像选 showcase 时，白底导致 SAM 平均前景比例约 `0.944`，不适合作为本轮展示入口；已记录该失败并改用标注数据集中的真实 Open3DHK 图像，按小/中/大 mask 各 8 张固定选择，未依据模型结果挑图。
- 原因判断：当前环境没有可用的 LaMa/BrushNet/FLUX Fill 推理后端和对应生成 checkpoint，因此 G-lite 仅是保守锐化加 seed-controlled 微扰的可运行 proxy，不能称为 diffusion 结果；旧 enhancement 输出也没有与固定 showcase 的同视角配对，B3 本轮未纳入。
- 结果变化：proxy 共 50 对、按 50 个源组划分 train/val=`40/10`。val 上 B0 为 PSNR/SSIM/L1=`33.5874/0.7166/16.0617`，B1 为 `33.6462/0.7099/17.3965`，G-lite 为 `26.6235/0.6955/17.3164`，Ours guard-selected 为 `33.5874/0.7166/16.0617`。B1 只有极小 PSNR 上升且 SSIM/L1 变差；G-lite 明显变差；guard 基本保持 identity，暂不支持把随机候选作为主方法。
- 结果变化：完成 600 条鲁棒性记录，覆盖 mask erosion/dilation `±3/±7` px、JPEG `60/80`、缩放 `0.75/1.25` 和 3 个 seed。重新定义后，所有 proxy/showcase/robustness CSV 的允许区域外最大像素误差均为 `0`；原始 mask 外统计仍会包含窄边界 feather，最大值分别为 proxy `41`、robustness `17`、showcase `68`，因此不能把原始 mask 外数值误报为严格保真失败。
- 结果变化：24 张 showcase 的人工/SAM Dice=`0.92645`。三 seed 的 mask 内 RGB 输出标准差均值为 B0=`0`、B1=`0`、G-lite=`1.145`、Ours=`0`（G-lite 最大=`1.316`）；guard 在本轮保守地选择了不随 seed 变化的候选。已生成完整 showcase、局部 zoom 和 failure-case 图。未安装 MUSIQ/CLIP-IQA，未用无参考分数替代视觉结论；人工失败图用于逐张复核，当前未将“修复不足”伪装成有 GT 的数值率。
- 是否保留：保留严格 soft-mask 合成、固定 showcase、SAM 与人工 mask 对照、proxy GT 仅作域内 sanity check 和 guard selection；放弃 G-lite 作为当前主修复器，不把 proxy PSNR 当作真实 Open3DHK GT 结论。
- 下一步：优先接入一个真实可运行的 NAFNet/去模糊 checkpoint 和一个真实 mask-aware inpainting backend；在同一 24 张 showcase 上比较局部质量与失败率。若没有真实 Open3DHK clean GT，只报告边界色差、结构断裂、多 seed 方差和人工失败统计，不声称真实像素质量已被 PSNR 证明提升。

## 2026-08-20 - restoration_v2 P0 严格 proxy 30 对检查

- 实验假设：只有在退化严格限制于精确 mask、clean 区域本身有建筑纹理且 input/clean 差异足够明显时，proxy 指标才可用于筛选真实恢复模型。
- 修改内容：新建 `research/restoration_v2/` 和独立 `research/data/restoration_v2_proxy_pairs.csv`；生成 `blur_downsample`、`smear_warp`、`repeat_missing` 各 10 对，退化按 `distorted=M*degraded+(1-M)*clean` 构造，并按 source group 划分 train/val。
- 失败或成功现象：首次 source-group 解析把 annotation 文件名中的数字误识别为组名，导致 30 对只有 3 个组且 val=0；已修正为识别 10 位大写组标识并重新生成。当前 30 对为 28 train / 2 val、3 类各 10 对。
- 原因判断：人工网格检查显示 clean patch 主要包含楼体、窗户或立面纹理；mask 内 blur、局部 warp、重复/缺失块均肉眼可见，mask 外未引入退化。仍需在 120 对上做正式统计，30 对只作为数据质量 gate。
- 结果变化：仅定性观察；未将这 30 对当作最终模型指标，也未使用上一轮 toy proxy 作为本轮恢复模型。
- 是否保留：保留 v2 数据生成路线；上一轮 `research/outputs/restoration/` 结果保持不变。
- 下一步：扩展到 120 对，加载官方 NAFNet-GoPro-width32 和官方 LaMa big-lama checkpoint，先在固定 showcase 和 proxy val 上跑真实模型。

## 2026-08-20 - restoration_v2 P0 扩展与真实后端 P1 烟测

- 实验假设：严格 mask 内的 paired proxy 可以先判断恢复模型是否真的修复局部退化；真实 Open3DHK 则必须同时检查视觉保真和 mask 外逐像素不变。
- 修改内容：将 proxy 扩展为 120 对（`blur_downsample`、`smear_warp`、`repeat_missing` 各 40），加入官方 NAFNet GoPro-width32 和官方 LaMa big-lama 推理脚本；LaMa 使用 mask bounding-box 加上下文 crop，两个模型输出都经过 soft composite。
- 失败或成功现象：NAFNet 和 LaMa 均在 RTX 3060 6GB 上成功加载并推理；BrushNet 本轮未接入，因为官方路径还需要额外的 Stable Diffusion base 和较大 BrushNet checkpoint，选择真实 LaMa 作为允许的 fallback，未用滤波器冒充生成模型。
- 原因判断：8 张 partial-mask showcase 的人工烟测中，NAFNet 输出大多接近输入，未形成稳定肉眼正向修复；LaMa 能真实改变 mask 内内容，但抽查中出现平坦纹理、纹理重画和材质/结构不一致，当前不能作为保真主方法。因此按预设 Go/No-Go 停止扩大生成式分支，保留其真实失败输出供审核。
- 结果变化：proxy 120 对的初步汇总已生成。Identity 的 masked PSNR/SSIM/LPIPS 均值为 `18.9908/0.4459/0.3986`；R1 NAFNet 为 `16.3373/0.3750/0.4916`；R2 LaMa 为 `20.9517/0.6017/0.1726`。LaMa 在严格配对 proxy 上有较好的数值表现，但不能抵消其在真实 Open3DHK 上的结构/材质漂移；因此 proxy 结果只作域内参考，不替代真实 showcase 结论。所有真实后端的允许区域外最大像素误差保持 `0`。
- 结果变化：同建筑候选检索使用当前本地可用的 220 张 annotation images，以 SIFT descriptor + RANSAC homography 做透明 fallback；24 张 showcase 中 13 张达到 `inliers>=8、inlier_ratio>=0.2、coverage>=0.03` 的可靠对齐，超过继续验证所需的 6 张。其余样本记录为无候选或弱 homography，未强行融合。
- 是否保留：保留严格 proxy、真实 NAFNet/LaMa 代码、SIFT/RANSAC 参考分支、固定 24 张 showcase、blind grid、failure cases 和空 `human_preference.csv`；不把 LaMa 在 proxy 上的优势写成 Open3DHK 真实提升，也不把 R1 的接近输入输出称作已验证 enhancement。
- 下一步：以 reference warp 的 13 个可靠样本作为下一轮可延伸方向；本轮最终报告以 `open3dhk_full_grid.png`、`open3dhk_zoom_grid.png` 和失败案例为主。若要得到真正的 Open3DHK 像素质量提升，下一轮需要更可靠的同建筑参考筛选/融合或具备真实建筑数据训练的恢复模型，而不是继续调低强度掩盖 LaMa 漂移。
