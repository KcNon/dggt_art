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
| mask | assign_maps vs 帧0 GT（匈牙利匹配 + Dice/NLL） | 否（router 监督） |
| sparsity | slot 稀疏 | 否 |
| kinematic | type/axis/pivot/scalar 对 GT | 否 |
| bbox | bbox 投影对 GT mask（GT-slot 序） | 否 |
| **render** | `sdf_render_loss`：逐部件 opacity BCE + silhouette BCE + 前景 RGB L1 | **是** |
| dead_opacity / render_global / pose_enc / pseudo | 均置 0（GS/相机相关，已弃用） | — |

`sdf_render_loss` 只渲染并监督**匹配上的运动部件**（GT slot0=背景∪静态，静态件无干净
silhouette）；每帧采样 `sdf_rays`(默认1024) 条光线（一半前景一半随机），β 默认 0.1。

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

- **阶段 2（动态帧）**：新增 `dggt/utils/ray_transform.py` 做光线逆变换（论文 Eq.S4-S8），
  渲染所有帧而非仅 frame 0；motion_type 训练期软混合、推理期 argmax。
- **阶段 3**：eikonal 正则 + depth 监督 + 1/β 退火 + 法线。
- **阶段 4（清理）**：删除 `ArtGaussianHead` 与 gsplat 路径；改 `eval_gs.py` /
  `inference_art.py` / `eval_phase1b.py`（仍读 `gs_*`）；`train_art.py` 的 phase-1b
  eval 分支仍用 `gs_mu`，跑 1b eval 前需替换为 SDF 渲染 IoU。
