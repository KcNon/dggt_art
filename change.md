# ArtVGGT 渲染迁移与架构文档

> 本文档记录 **GS → SDF 体渲染** 的改造、配套的数据兼容性修复，以及改造后
> **当前模型的架构与前向传播流程**。代码引用形如 `文件:行号`。

---

## 1. 变更日志

### 2026-06-07　数据兼容性修复（PartNet-Mobility data_processed）

**动机**：新数据 `/data2/lza/partnet-Mobility/data_processed` 由 `process.py`(SAPIEN)
重新生成，与旧 loader 不兼容：① 多了一层「类别」目录；② part_mask 文件名用的是
SAPIEN 全局递增的 `per_scene_id`（绝对 link id，如 82–86），而非旧约定 `joint_k→k+2`，
且 seg_id↔joint 映射未保存。

- **`scripts/build_partnet_part_index.py`（新增）**：精确复刻 `process.py` 的全局
  加载顺序（灯光占 id 1,2；每模型消耗 `link 数 + 4 相机`），**无重渲染**复原
  `seg_id↔joint` 映射，为每个场景写 `part_index.json`。全部 1063 场景校验 0 错配。
- **`datasets/articulated_dataset.py`**：
  - 扫描场景时若顶层找不到，自动下探一层支持 `data_root/{类别}/{object_id}/`。
  - 存在 `part_index.json` 时按映射读 mask（动态关节 k→slot k+1，base/静态件/杂散 id→slot 0，
    slot0 = 1−动态并集，保证逐像素精确划分）；无该文件回退旧 `k+2` 逻辑。

### 2026-06-07　GS → SDF 体渲染迁移（阶段 1：rest 帧静态多部件合成）

**动机**：将外观/几何表征从 Gaussian Splatting 换成论文的
**SDF 体渲染（VolSDF + 每部件 hexa-plane）**。本阶段先打通最小闭环：利用数据集
frame 0 = 静止初始态，所有部件处于 canonical rest pose，**无需关节变换**，做纯静态
多部件 SDF 合成渲染。决策：装 nerfacc 加速、完全替换 GS、分阶段推进、不预测相机
（全程用 GT 外参）。

- **`dggt/heads/articulation_head.py`**：关节分支重构为**单关节向量** `Â_p∈R¹⁴`
  （论文 Eq.3-6）再按通道切分重映射：`bbox_center=2rσ−r`、`bbox_size=2rσ`、
  `axis=归一化`、`pivot=2rσ−r`、`scalar=2σ−1`、`type=2 logits`；**Slot 0 固定 static**
  （movable 2 分类只作用于 Slot 1..P-1）。
- **`dggt/heads/hexaplane_sdf_head.py`（新增）**：每部件解码 6 张带符号特征平面
  `{xy±,yz±,xz±}`（R=64, Cf=32），共享 SDF/RGB MLP；**球初始化**（SDF MLP 末层零初始化
  → 初始 SDF≈`‖x̂‖−0.5`）。查询 `query(planes, x̂)`：hexa bilinear → 3Cf 特征
  → `s=MLP_sdf+s_bias`、`c=σ(MLP_rgb)`。
- **`dggt/render/sdf_volume.py`（新增）**：nerfacc 体渲染器。
  `generate_rays`（**OpenGL 相机约定**）→ 逐部件 `ray_aabb_intersect` + 分层采样
  → `query` 取 SDF/RGB → `volsdf_density` 转密度 → 跨部件合并按 ray 距离排序
  → `render_weight_from_density` + `accumulate_along_rays` 合成 RGB/opacity/depth/逐部件 opacity。
- **`dggt/models/art_vggt.py`**：`use_camera_head=False`（默认，见 train_art 构造）；
  `gaussian_head → sdf_head`；forward 输出 `preds["planes"]`，去掉所有 `gs_*`；
  `set_phase` 的解冻列表 `gaussian_head→sdf_head`。
- **`train_art.py`**：
  - 新增 `sdf_render_loss()`：仅渲染并监督**运动部件**（GT slot0=背景∪静态，无法分离
    静态 silhouette）——逐部件 opacity BCE + 并集 silhouette BCE + 前景 RGB L1。
  - `compute_loss` 用 `sdf_render_loss` 替换 `per_part_alpha_render_loss` + gsplat 全局渲染；
    `dead_opacity`（依赖 gs_opacity）置 0；`pose_enc` 因无 camera_head 自动归零；
    调用处传入 `head=raw_model.sdf_head`。
  - 模型构造 `use_camera_head=False`。
- **验证脚本**：`scripts/smoke_sdf_phase1.py`（整管线过拟合：loss 下降、能渲图、无 NaN）；
  `scripts/smoke_sdf_render_overfit.py`（隔离渲染器拟合单 GT mask → **IoU 0.95**）。

### 2026-06-09　slot0 = 纯静态主体 + 背景 sink slot（对齐论文）

**动机**：论文 "one slot is reserved for the base part"。原 slot0 = 背景∪静态主体，
导致最大、最实心的 base 部件从不被渲染监督。重定义：**slot0 = 纯 base 部件，背景不属于
任何 slot**，由 router 里一个可学习的「背景 sink」吸收（softmax 在 P+1 上做）。

- **`datasets/articulated_dataset.py`**：`part_masks[s,0]` = 纯静态主体
  （`part_index.json` 的 `static_ids` 并集 / legacy `root_mask`），不再含背景；
  新增 `static_body_mask`/`has_static_mask` 字段。
- **`dggt/heads/part_slot_router.py`**：可学习 `bg_token`；attention 在 P+1 个 slot 上；
  返回 `(slot_features[B,P], assign_maps[B,P,·,·], bg_map[B,1,·,·])`，下游 P-slot 接口不变
  （assign_maps 求和 <1，背景质量被移除）。
- **`dggt/models/art_vggt.py`**：透传 `preds["bg_map"]`。
- **`train_art.py`**：`mask_loss(..., bg_map)` 构造 P+1 类 NLL（背景=类 P，默认标签=P）；
  phase-1a eval 的 argmax 拼上 bg 通道。`sdf_render_loss` 改为**渲染 base（pred slot0↔gt0
  固定配对、motion 强制 static）+ 运动部件**，union = 完整物体前景；新增 GT depth 监督
  （`gt_depth`/`w_depth` 参数，相机 z 深度→光线距离）。
- **验证**：smoke 通过；40453 geom-only overfit mIoU 0.32（仅运动件）→ **0.63**（含 base）。

### 2026-06-09~10　overfit 验证阶梯 + 损失设计 A/B（`scripts/overfit_scene_sdf.py`）

按「mask → mask+GT关节 → mask+关节+RGB」三步验证表示能力上限（每场景直接优化
latent+bbox+共享 MLP，无 transformer）。新 flags：`--only_part/--no_base/--no_part_mask/
--single_stage/--rgb_mode/--seed`。

- **Step1**（单 stage、全静态、mask-only）：mIoU **0.96** —— 分解能力没问题。
- **Step2**（+GT 关节、8 stage）：开抽屉 stage IoU 0.8–0.95，base ~0.9；闭抽屉低 IoU
  是可见性 artifact（GT 仅 ~100px），非表示缺陷。
- **Step3**（+RGB）：分解保持（base 0.92），PSNR 31.5。**整条管线在给定条件下可 overfit**。
- **A/B per-part vs composite 渲染损失**（同 seed 对照，40453）：composite 胜
  （base IoU 0.94 vs 0.84，快 ~4×，PSNR 打平）。论文 per-part 的收益依赖 **amodal mask**；
  我们是 **modal mask + GT depth**，composite depth 在每个前景像素钉住最前表面，
  per-part 反而拆散该耦合。**保留 composite**，`--rgb_mode perpart` 留作日后实验。
- **A/B depth on/off**：几乎零差异（mask loss 已钉住轮廓）→ **1b 用 `--w_depth 0`**（贴论文）。

### 2026-06-10　phase 1a 修复：bg-sink 吞掉 base 部件（val IoU 0.40 → 0.69）

**症状**：fresh 1a 训练 val IoU 先崩到 ~0 再爬到 **0.40 封顶**，永达不到 0.7 warmup gate。
**诊断**（新增 `scripts/diag_assign_maps.py`，dump ckpt 的 assign/bg argmax 可视化 + 逐 slot
质量统计）：所有场景 slot0 mass=0.000（死亡）、bg sink 占 78–91%——**bg sink 把 base 当背景
吞掉了**。根因：`hungarian_match_masks` 只匹配 slot/gt 1..P-1，从不返回 (slot0,gt0)
（注释称 "paired by design" 但无人真正配对）；`mask_loss` 里 base 像素落入背景默认标签
→ 等于训练 sink 吃掉 base、slot0 零前景监督。

- **修复**：`mask_loss` 与 `eval_mean_iou` 在 bg_map 存在时显式补固定对 `(0,0)`
  （label_map + Dice / IoU 计分均含 base）。micro check（新增
  `scripts/micro_check_phase1a.py`）：slot0 mass 0→0.39，base IoU 10 步 0→**0.85**。
- **lr warmup**：CosineAnnealingLR → LambdaLR（线性 500 步 warmup → cosine，
  `--lr_warmup_steps`）。无 warmup 时 fresh 模型开局即被 1e-4 峰值 lr 冲崩（前 6k 步 IoU≈0）。
- **NCCL 稳定性**：`eval_mean_iou` 限 `--val_max_batches`（默认 64；rank0-only 全量 val
  超 10 分钟看门狗 → 其他 rank 在 broadcast 处 abort）；`init_process_group` timeout 30min。
- **结果**：fresh 重训（GPU 0,1,3）20k 步 val IoU 稳定爬升至 **0.69 平台**（无早期崩溃；
  距 0.7 gate 差 0.01 且 lr 已耗尽，判定 1a 完成）。ckpt：
  `/data5/lza/checkpoint/Art/phase1a_bgsink_warmup/ckpt_020000.pth`。旧 buggy run 保留在
  `.../phase1a_bgsink` 供对照。

### 2026-06-11　phase 1b 多卡启动准备：DDP「marked ready twice」修复链

**症状**：1b（`w_render=1`）多卡启动即报 DDP `sdf_head.rgb_mlp ... marked as ready twice`，
且对检查点开关、static_graph、find_unused、分辨率（518/252）均时好时坏。
**真根因**：`sdf_mlp`/`rgb_mlp` 只在**损失侧**渲染时使用（`compute_loss` 拿
`raw_model.sdf_head`，绕过 DDP wrapper），不在 DDP 的 forward 图里——find_unused 在
forward 结束遍历输出图时把它们误标记为「未用→ready」，backward 真梯度一来 hook 再触发
→ 二次标记。检查点与分辨率都不是根因。

- **`dggt/render/sdf_volume.py`**：`render_rays_static` 重构——逐部件循环只收集 hexa
  特征（`query_features`，用的是各部件 planes 激活，无共享参数），循环后把所有部件采样点
  拼一起，共享 SDF/RGB MLP **单次批量调用**（更快，参数使用次数从「逐部件」降为「每帧一次」）。
- **`train_art.py`** DDP 策略（分相位）：1a 维持 `find_unused_parameters=True`（warmup
  冻结头）；**1b 把 `sdf_head.sdf_mlp/rgb_mlp` 从 DDP reducer 排除**
  （`_set_params_and_buffers_to_ignore_for_model`）+ **backward 后手动 all_reduce 其梯度**
  （`_manual_sync_params`，各 rank 统一参与、无梯度时补零）；其余参数每步必用且图静态
  → `static_graph=True` 正确处理梯度检查点的延迟梯度（此前单独用 static_graph 失败是因
  逐部件 MLP 调用次数随场景变化；批量化后调用次数固定）。
- **`train_art.py`** 其他：`--reset_step`（仅载全部权重，step/优化器/调度器/warmup 清零，
  用于 1a→1b 跨相 resume）；`--w_depth` 提升为 CLI 参数；**pos_embed 跨分辨率插值**
  （resume 时 ckpt 与模型 patch 网格不一致则 bicubic 插值，支持 518↔252）。
- **`scripts/launch_phase1b.sh`**：重写为本机配置（原为他人机器路径）：GPU 0-3、
  518 + 8 帧 + 梯度检查点、`--w_depth 0`、`--reset_step` resume 1a `ckpt_020000`、
  30k 步、lr 2e-5。
- **状态（✅ 已根治并启动正式 1b，2026-06-13）**：见下方 option B；4 卡 100 步 smoke 通过
　（"Training complete"），正式 30k 跑通前 250 步无报错。早期 60 步 4 卡 smoke 曾通过（峰值 34GB/40GB），但 **120 步多场景长测
  （smokeF）第 15 步即崩**，且换配置错误形态在变（`find_unused=True`→rgb_mlp marked twice；
  `find_unused=False`→`aggregator.global_blocks.11` did not receive grad；ignore+static_graph→
  smokeF 崩）。说明「损失侧共享 MLP + DDP reducer + 梯度检查点 + 每场景可变图」这组叠加
  **尚未被现有方案稳定解决**，对场景/卡数敏感。**正式 1b 未启动**。
- **根治（option B，✅ 已做并跑通）**：`sdf_render_loss` 两趟——趟 A 用新拆出的
　`collect_samples`（仅采样 + 逐部件 `query_features`，**不碰共享 MLP**）跨**所有 (batch,
　frame) 收集全部采样点**；趟尾 `torch.cat` 后 `sdf_mlp`/`rgb_mlp` **全局只调一次**；趟 B
　用 `composite_samples` 逐帧 nerfacc 合成、算 sil/part/rgb/depth。配合「从 DDP reducer
　排除 sdf_mlp/rgb_mlp + backward 后手动 all_reduce + `find_unused_parameters=True`」，
　保住 518 + 8 帧 + 梯度检查点。`render_rays_static` 改为 collect→单次 MLP→composite 薄封装。
  （旧表述，保留为对照）把 `sdf_render_loss` 重构为「跨**所有帧 + 部件**收集特征 →
  `sdf_mlp`/`rgb_mlp` 全局**只调一次** → 再逐帧合成」，使共享参数真正单次使用，
  从根上消除多次 hook 触发；届时 DDP 可回到朴素 `find_unused`。当前的逐部件→单次批量化
  （已做）是这条路的一半，还差跨帧批量化。
- **已放弃 option A**（1b 降 252 + 关检查点）：测出根因不是分辨率/检查点而是 DDP-reducer，
  故保留 518 + 8 帧 + 检查点；pos_embed 跨分辨率插值代码留作后用。

---

## 2. 当前模型架构（ArtVGGT）

前馈式关节物体重建 Transformer。组件（`dggt/models/art_vggt.py`）：

| 模块 | 文件 | 作用 |
|---|---|---|
| Aggregator | `dggt/models/aggregator.py` | VGGT 编码器：DINOv2 patch_embed(冻结) + frame/global attention + P 个 slot token |
| ~~CameraHead~~ | `dggt/heads/camera_head.py` | **本配置禁用**（`use_camera_head=False`），全程用 GT 外参 |
| PartSlotRouter | `dggt/heads/part_slot_router.py` | 输出 `slot_features[B,P,D]`、`assign_maps[B,P,H_p,W_p]`（逐 slot 软分割） |
| ArticulationHead | `dggt/heads/articulation_head.py` | 关节向量 Â_p：motion_type / axis / pivot / scalar / bbox |
| **HexaPlaneSDFHead** | `dggt/heads/hexaplane_sdf_head.py` | 每部件 hexa-plane 特征 + 共享 SDF/RGB MLP（**替换 ArtGaussianHead**） |
| SDF 渲染器 | `dggt/render/sdf_volume.py` | VolSDF 体渲染 + 多部件合成（训练时按需采样光线） |

约定：`P=8` slots，Slot 0 = 静态 base，Slot 1..P-1 = 各运动关节。`scene_radius r=1.0`。

---

## 3. 当前前向传播流程

输入：`images[B,S,3,H,W]`、`extrinsics[B,S,4,4]`(GT, cam-to-world)、`intrinsics[B,3,3]`、`timestamps[B,S]`。

```
① Aggregator(images)
     → image_tokens[B,S,N+5,2C], dino_tokens[B,S,N+5,C], slot_states[B,P,C]
② (CameraHead 跳过) → 直接用 GT extrinsics
③ Plücker rays  ← 由 GT extrinsics + intrinsics 计算 [B,S,N_patch,6]，喂给 router
④ PartSlotRouter(image_tokens, dino_tokens, plucker, slot_states)
     → slot_features[B,P,D], assign_maps[B,P,H_p,W_p]
⑤ ArticulationHead(slot_features, timestamps)
     → motion_type_logits[B,P,2], axis[B,P,3], pivot[B,P,3],
       scalars[B,P,S], bbox_center[B,P,3], bbox_size[B,P,3]
⑥ patch_feats_frame0 = concat(agg_frame0[2C], dino_frame0[C])  [B,N_p,3C]
   HexaPlaneSDFHead.decode_planes(slot_features, patch_feats_frame0, assign_maps)
     → preds["planes"][B,P,6,Cf,R,R]
返回 preds（含 planes / bbox / axis / pivot / scalars / motion_type_logits /
            assign_maps / slot_features / plucker_rays）
```

> 渲染**不在 forward 内**进行；planes 等随 preds 返回，渲染发生在 loss / 推理阶段
> （按需对采样光线体渲染），避免每步全图渲染的开销。

---

## 4. SDF 体渲染流程（`render_rays_static`，阶段 1 静态）

对一张图的一批光线 `(o,d)`（世界系，**OpenGL 约定**：`d_cam=[(u-cx)/fx, -(v-cy)/fy, -1]`）：

```
for 每个 alive 部件 p:
    AABB_p = [center_p - size_p, center_p + size_p]
    t_min,t_max,hit = nerfacc.ray_aabb_intersect(o,d,AABB_p)
    命中光线段内分层采样 n_samples 点  → t_mid
    x_world = o + t_mid·d ;  x̂ = (x_world - center_p)/size_p ∈ [-1,1]³
    sdf, rgb = head.query(planes_p, x̂)
    σ = volsdf_density(sdf, β)               # α=1/β, 数值稳定 exp(-|s|/β)
合并所有部件采样点 → 按 (ray, t) 排序
weights = render_weight_from_density(t_starts,t_ends,σ, ray_indices)
rgb合成/opacity/depth/逐部件opacity = accumulate_along_rays(weights, …)
comp_rgb = comp_rgb + (1-opacity)·bg
```

**VolSDF 密度**（`sdf_volume.volsdf_density`，论文 Eq.S3 / [69]）：
`α=1/β`，`half=½·exp(-|s|/β)`，`σ = α·(s≥0 ? half : 1-half)`。
（统一用 `exp(-|s|/β)` 规避 torch.where 两分支同时求值导致的 inf/NaN。）

---

## 5. 损失（`compute_loss`，非 warmup 阶段）

| 损失 | 说明 | 是否依赖 SDF |
|---|---|---|
| mask | assign_maps(+bg_map) vs 帧0 GT：匈牙利匹配 1..P-1 + **固定 (slot0,gt0) 对**，P+1 类 NLL + Dice | 否（router 监督） |
| sparsity | slot 稀疏 | 否 |
| kinematic | type/axis/pivot/scalar 对 GT | 否 |
| bbox | bbox 投影对 GT mask（GT-slot 序） | 否 |
| **render** | `sdf_render_loss`：**base + 运动部件**合成渲染；逐部件 opacity BCE + 并集 silhouette BCE + 前景 RGB L1 + depth L1（`--w_depth`，1b 默认 0） | **是** |
| dead_opacity / render_global / pose_enc / pseudo | 均置 0（GS/相机相关，已弃用） | — |

`sdf_render_loss` 渲染 **base（slot0↔gt0 固定配对，motion 强制 static）+ 匹配上的运动
部件**，运动部件按各帧 GT 关节状态做光线逆变换；每帧采样 `sdf_rays`(默认1024) 条光线
（一半前景一半随机），β 默认 0.1。RGB/depth 用 composite 而非论文的 per-part
（A/B 实证 modal mask + depth 下 composite 更优，见变更日志 06-09~10）。

---

## 6. 关键约定与注意事项

- **相机 = OpenGL 约定**（forward −z、+y up、行 v 向下）。由 GT depth 反投影实测确认
  （`process.py`: `depth=-position_z`、`get_model_matrix`=cam-to-world），物体落在
  ~[-0.85,0.85]³，故 `scene_radius=1.0` 合适。⚠️ 旧 GS 渲染代码假设 OpenCV，对此数据是错的。
- **nerfacc**：CUDA 算子在 phase 1b（渲染）首次调用时 JIT 编译；首编需
  `CUDA_HOME=/usr/local/cuda-12.1`、`CC=gcc-11`、`CXX=g++-11`、`ninja`。phase 1a warmup 不触发。
- 渲染 loss 仅在 warmup 之后（phase 1b）生效，与原 GS 一致。

---

## 7. 后续阶段 TODO

- ~~**阶段 2（动态帧）**~~ ✅ 已完成：`sdf_volume.inverse_transform_rays` 光线逆变换
  （论文 Eq.S4-S8），`sdf_render_loss` 渲染多帧（`sdf_frames`，默认 4 帧均匀采样）；
  motion_type 训练期软混合（slot0 强制 static）。
- **阶段 3（部分完成）**：depth 监督已实现（`--w_depth`，1b 默认关，A/B 显示作用有限）；
  1/β 退火与 eikonal 已在 overfit 脚本验证，**主训练 1b 仍用固定 β=0.1、无 eikonal**，
  待 1b 跑通后视渲染质量决定是否引入；法线未做。
- **阶段 4（清理）**：删除 `ArtGaussianHead` 与 gsplat 路径；改 `eval_gs.py` /
  `inference_art.py` / `eval_phase1b.py`（仍读 `gs_*`）；`train_art.py` 的 phase-1b
  eval 分支仍用 `gs_mu`，跑 1b eval 前需替换为 SDF 渲染 IoU。
