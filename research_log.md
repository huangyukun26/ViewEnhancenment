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

## 2026-08-21 - reference-consistency P0 与 MatrixCity 小规模 GT 验证

- 实验假设：修正缩放坐标后的 homography、只用 mask 外可靠重叠区域做颜色调整，并把参考纹理限制在高置信度 mask 交集内，可以避免全局参考 warp 的结构漂移；若参考视图确实有用，MatrixCity 的真实 clean target/reference 对应应能相对 identity 改善。
- 修改内容：在新分支 `codex/reference-consistency-20260821` 增加 `reference_consistency.py`。使用 `H_native=inv(S_target)@H_small@S_reference`；候选池去重到 220 张 RGB，每个目标保留 top-5；增加 `mask_support_ratio`、mask 内高置信度/平均 confidence、forward-backward consistency、local reprojection error、mask 外 photometric residual，并按 24 张 unique image 分开汇总 manual/SAM。生成 manual/SAM 24 张 contact sheet、局部 zoom、confidence/difference 图和空白 `human_preference.csv`。
- 失败或成功现象：P0 严格 mask 外最大误差为 `0`。manual 仅 `2/24` 达到当前保守可靠条件，SAM 为 `1/24`；median mask support 为 manual=`0.6498`、SAM=`0.8883`，但 median mask confidence 仅约 `0.0025`。许多样本虽有投影覆盖，SIFT inlier 在 mask 内稀疏或被单个全局 homography 拉成错误平面，未强行融合。
- 原因判断：当前 220 张候选仍不足以覆盖同一建筑的可靠邻近视角；SIFT+homography 对非平面立面和大 mask 不稳。该结果支持继续试真实稠密匹配，但不支持调阈值来人为扩大覆盖。
- 结果变化：MatrixCity 官方 small-city street train_dense 的 `road_down` 仅按 HTTP Range 读取 RGB tar 前 512MB，配合官方 pose 索引和 Open3DHK mask 形状，构造了 100 个 target/reference/clean-GT 对；退化包括 `uv_stretch`、`repeat_seam`、`resolution_blur`，严格满足 `distorted=M*degraded+(1-M)*clean`。未将 MatrixCity 原始数据或 tar 提交仓库。
- 结果变化：MatrixCity 100 对的 PSNR/SSIM/LPIPS 均值分别为 identity=`18.7909/0.5359/0.2805`，SIFT homography=`17.3935/0.4627/0.1515`；median LPIPS 为 `0.2563/0.1262`。LPIPS 下降但 PSNR/SSIM 同时下降，说明该 baseline 可能改善感知特征却产生像素/结构错位，不能满足“像素质量确实提高”的 go 条件；mask 外最大误差为 `0`。
- 结果变化：RoMa v2 官方实现使用 Python 3.12、RTX 3060、fast/512、batch=1 成功初始化并完成烟测；8 张 Open3DHK manual showcase 的 mean mask support=`0.1603`，其中只有 2 张约 `0.53/0.60`，其余多为低重叠，mask 外最大误差仍为 `0`。fast 模式不提供双向 warp（`bidirectional=false`），因此本轮不宣称已完成 forward-backward dense consistency。
- 是否保留：保留 P0 修正、完整图和失败案例、严格 outside invariant、MatrixCity 小规模有 GT 代码与缓存外数据说明；不保留全局 homography 作为默认修复方法。
- 下一步：完成 RoMa 官方实现烟测；若官方稠密匹配能在 RTX 3060 6GB 的 fast/512 设置下运行，再只在小子集比较 dense overlap/confidence。若 RoMa 不能稳定运行或仍无有效 mask 内覆盖，则按 Go/No-Go 停止扩展修复器，结论转为需要更密集的同建筑邻近视图。

## 2026-08-21 - FLUX constrained restoration quick6 Go/No-Go

- 实验假设：FLUX.2-klein-4B 在包含上下文的 Open3DHK 局部 crop 上，配合正向保真 prompt、固定 seed、可选同建筑参考和严格 mask 合成，可能比 identity 产生可见修复，同时不改变 mask 外像素。
- 修改内容：新建 `research/data/flux_quick6.csv`，固定 6 张代表性图（2 张 UV stretch、2 张 repeat/seam、2 张 blur/low-resolution），并新增 `research/restoration_v2/flux_constrained.py`。runner 实现 target-only / target-plus-reference 两类候选、seed 0/1、96--160 像素上下文 crop、4 像素 mask 内 feather、raw generation 保存、边界/低频/变化量指标、contact sheet、盲评图和空白 `human_preference_flux.csv`。
- 失败或成功现象：本机没有 FLUX.2-klein-4B 权重，`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` 和 `BFL_API_KEY` 均未配置；检测到 RTX 3060 仅 6GB 显存，而官方 FLUX.2-klein-4B 模型卡标注约 13GB 显存。为避免把 identity 冒充 FLUX，本轮没有发起伪造生成，也没有扩展到 24 张。
- 原因判断：当前是外部模型权重、凭据和硬件资源 blocker，不是模型质量负结果。runner 的 Diffusers 真实入口已保留，给定本地官方 checkpoint 后可按 `run_commands.md` 复现；当前输出中的 `blocked_identity_passthrough` 仅用于检查数据入口、图像组织和严格合成约束。
- 结果变化：6 张、14 个候选记录均为 blocker identity passthrough；真实 FLUX candidate 数为 `0`，selected identity/abstain 为 `6/6`。所有候选 `outside_max_abs=0`，满足像素外保真 sanity check；`changed_fraction_mask` 和 `mean_abs_delta_mask` 为 `0`，因此没有任何可据此声称的视觉提升。
- 是否保留：保留 quick6 manifest、真实 FLUX.2 Diffusers runner、raw/candidate/selected 目录、contact sheet、blind key、空白人工评价表和准确 blocker 报告；不把身份透传结果列为 FLUX enhancement，不填写主观 better/worse。
- 下一步：只有在提供官方 FLUX.2-klein-4B 本地权重且具备可用显存/远程推理资源后，重新运行这 6 张并由人工填写盲评；在真实 FLUX 候选产生前，本轮判定 `NO-GO_BLOCKED`，不扩展到 24 张，也不继续调 SIFT/RoMa/NAFNet/LaMa。

## 2026-08-21 - FLUX.2-klein-4B 云端 V100 quick6 实推

- 实验假设：官方 FLUX.2-klein-4B 在 Open3DHK 局部 crop 上能产生真实候选；32GB V100 使用纯 GPU 推理应比 6GB 本地显卡的 CPU offload 更适合短周期验证，同时严格 mask 合成应保持 mask 外像素不变。
- 修改内容：在云服务器按 Diffusers 目录只下载必需权重，使用官方 Hugging Face URL、aria2 多连接和断点续传；补装 runner 所需的 `scikit-image`；新增按显存自动选择纯 GPU/CPU offload 的路径，32GB V100 用 fp16 纯 GPU，6GB 显卡保留 offload fallback。保留一份慢路径的 1 张部分结果到 `flux_constrained_cloud_v100_offload_partial`，完整结果写入独立的 `flux_constrained_cloud_v100`。
- 失败或成功现象：模型权重完整下载并成功加载；第一次 CPU offload 候选约 1 分钟级，确认不是模型错误后停止该慢路径并重跑。纯 GPU runner 成功完成 6 张 quick6、14 个真实 FLUX 候选（target-only 12 个，含参考图的 2 个），没有 blocker。参考图仅存在于其中 1 张样本，不能据此得出 multi-reference 优势结论。
- 原因判断：32GB 显存足以避免本轮 CPU offload；纯 GPU 使 4 步候选推理约为 `2.6806--8.9972 s`，均值 `5.4565 s`，按退化类型均值为 uv_stretch=`6.207 s`、repeat_seam=`5.330 s`、blur_lowres=`4.457 s`。下载慢的主要瓶颈是 Hugging Face 上游/CDN，未采用实测超时的镜像。
- 结果变化：`flux_candidates.csv` 共 14 行，所有候选 `outside_max_abs=0`，独立复核也是 `0`；mask 内确实发生了生成变化，`changed_fraction_mask` 约为 `0.9934--0.9999`。summary 标记 `REVIEW_QUICK6`，生成候选数为 `14`，selected identity 数为 `0`。这些数值只证明真实生成和严格合成已运行，不能证明视觉质量提升。
- 是否保留：保留官方 FLUX.2 runner、纯 GPU 显存分流、完整 6 张 contact/blind/zoom/difference/raw 输出和空白 `human_preference_flux.csv`；不由 Codex 填写人工 better/same/worse，也不把自动 delta 分数解释成真实 Open3DHK 质量提升。
- 下一步：先由人工查看盲评图并填写 `human_preference_flux.csv`，同时检查结构改变、材质漂移和接缝；只有满足原定 quick6 门槛才扩展到 24 张，否则保留失败图并停止换模型。当前 target-only 与 multi-reference 的稳定性、哪类退化最适合生成修复，仍需人工盲评后回答。
