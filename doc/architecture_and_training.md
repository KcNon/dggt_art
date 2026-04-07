# DGGT-ART: 铰链物体四维高斯变换器 — 架构与训练详解

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [模块详解](#3-模块详解)
   - 3.1 [编码器：Aggregator（VGGT）](#31-编码器aggregatorvggt)
   - 3.2 [姿态估计头：CameraHead](#32-姿态估计头camerahead)
   - 3.3 [Plücker 射线计算](#33-plücker-射线计算)
   - 3.4 [零件槽路由器：PartSlotRouter](#34-零件槽路由器partslotrouterpart_slot_routerpy)
   - 3.5 [铰链运动头：ArticulationHead](#35-铰链运动头articulationhead)
   - 3.6 [三维高斯头：ArtGaussianHead](#36-三维高斯头artgaussianhead)
   - 3.7 [刚体变换工具](#37-刚体变换工具)
4. [数据集](#4-数据集)
5. [损失函数](#5-损失函数)
6. [训练流程](#6-训练流程)
7. [推理与评估](#7-推理与评估)
8. [张量形状速查表](#8-张量形状速查表)
9. [数据流图](#9-数据流图)

---

## 1. 项目概述

DGGT-ART 是一个基于 Transformer 的多视角铰链物体四维重建框架。输入多帧多视角图像，输出：

- **零件分割**：每个像素属于哪个零件（槽）
- **关节参数**：每个关节的类型（平移/旋转）、轴向量、枢轴点、运动标量
- **三维高斯表示**：每个零件的规范空间高斯分布，可渲染到任意帧/视角

### 数据来源
- **Phase 1**：PartNet-Mobility（全监督，含 GT 关节参数与分割掩码）
- **Phase 2**：真实数据 + SAM2 生成伪掩码（弱监督）

### 支持的关节类型
| 类型 | 编号 | 描述 |
|------|------|------|
| 平移关节 (prismatic) | 0 | 沿轴线平移，无旋转 |
| 旋转关节 (revolute)  | 1 | 绕枢轴旋转 |

> **注意**：静态底座（Slot 0）固定不动，不参与关节类型分类损失。

---

## 2. 整体架构

```
输入: images [B, S, 3, H, W]  (B=batch, S=帧数)
      extrinsics [B, S, 4, 4]  (相机外参，cam-to-world)
      intrinsics [B, 3, 3]     (相机内参)
      timestamps [B, S]        (时间戳，归一化到 [0,1])

                    │
                    ▼
         ┌─────────────────────┐
         │  Aggregator (VGGT)  │  ← 冻结的 DINOv2 + 聚合层
         │  (图像编码器)         │
         └─────────┬───────────┘
                   │ agg_tokens [B,S,P_total,2D]
                   │ dino_tokens [B,S,P_total,D]
                   │
          (可选) ┌──▼──────────┐
                 │ CameraHead  │  → pose_enc [B,S,9] → extrinsics
                 └─────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   Plücker 射线计算   │  → plucker_rays [B,S,N_patches,6]
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │  PartSlotRouter     │  ← 核心零件分解模块
         │  (分组 Transformer) │
         └──────┬──────┬───────┘
                │      │
      slot_features   assign_maps
      [B,P,D]         [B,P,H_p,W_p]
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ArticulationH │  │ArtGaussianH  │
│(关节参数头)   │  │(三维高斯头)   │
└──────┬───────┘  └──────┬───────┘
       │                 │
  type/axis/pivot/    gs_mu/rot/
  scalars/bbox        scale/color/opacity
```

**核心类**：`ArtVGGT`（`dggt/models/art_vggt.py`）

```python
ArtVGGT(
    img_size=518,          # 输入图像大小
    patch_size=14,         # DINOv2 patch 大小
    embed_dim=1024,        # 特征维度
    num_slots=8,           # 最大零件槽数量（含 Slot 0 静态底座）
    n_gaussians=256,       # 每个槽的高斯数量
    scene_radius=1.0,      # 场景尺度半径
    use_camera_head=True,  # 是否启用相机姿态估计
    stop_gradient_plucker=False,
)
```

---

## 3. 模块详解

### 3.1 编码器：Aggregator（VGGT）

- 架构基础：冻结的 **DINOv2 ViT-L**（patch_size=14，embed_dim=1024）
- 在 DINOv2 之上加入跨帧聚合层，形成 `agg_tokens`（维度 = 2×embed_dim = 2048）
- 输出：
  - `dino_tokens [B, S, P_total, 1024]`：原始 patch 特征（含 cls/reg 等特殊 token）
  - `agg_tokens [B, S, P_total, 2048]`：聚合后的跨帧特征

训练阶段的冻结策略：
| 阶段 | DINOv2 | Aggregator |
|------|--------|------------|
| Phase 1a (Warmup) | 冻结 | 训练（部分层） |
| Phase 1b (全监督) | 冻结 | 训练 |
| Phase 2 (弱监督) | 冻结 | 仅最后 4 个 block 解冻 |

---

### 3.2 姿态估计头：CameraHead

- 输入：`agg_tokens [B, S, P_total, 2D]`
- 输出：`pose_enc [B, S, 9]`（可解码为 4×4 外参矩阵）
- Phase 1 使用 GT extrinsics；Phase 2 由 CameraHead 预测，Plücker 射线 **stop-gradient** 防止梯度回传破坏编码器

---

### 3.3 Plücker 射线计算

**文件**：`dggt/utils/plucker.py`

每个 patch 用一条 **Plücker 线（6D）** 表示几何先验：

```
d = 单位射线方向（世界坐标系）
m = o × d（矩向量，o 为相机原点）
Plücker = [d, m]  ∈ R^6
```

计算流程：
1. 构造像素/patch 中心坐标网格
2. 用内参矩阵 K_inv 投影到相机空间射线方向 `d_cam`
3. 用外参旋转矩阵 R 转换到世界坐标系：`d_world = R @ d_cam`，L2 归一化
4. 相机原点：`o = extrinsics[:, :, :3, 3]`
5. 矩向量：`m = o × d_world`

输出形状：`[B, S, N_patches, 6]`，其中 `N_patches = (H/14) × (W/14)`

---

### 3.4 零件槽路由器：PartSlotRouter（`part_slot_router.py`）

这是模型的核心分解模块，将图像特征路由到 P 个"零件槽"。

#### 输入 Token 拼接

```
agg_token    [B, S, P_total, 2048]  (VGGT 编码)
dino_token   [B, S, P_total, 1024]  (DINOv2 原始特征)
plucker_ray  [B, S, N_patches, 6]   (几何射线先验)
stage_emb    [B, S, N_patches, D_s] (时间戳嵌入)

→ 线性投影 → image_tokens [B×S×N_patches, dim_slot=1024]
```

时间戳嵌入：标量时间戳 → 64 维向量（MLP），拼接到 patch token 上，让模型感知时序。

#### 可学习槽 Token

```python
self.slot_tokens = nn.Parameter(torch.zeros(1, P, dim_slot))
# 初始化为零，训练后分化为不同零件表示
```

#### Transformer 层结构（共 8 层）

层顺序（75% cross / 25% self）：

```
[Cross, Cross, Cross, Self, Cross, Cross, Cross, Self]
```

**交叉注意力层（CrossAttentionLayer）**：
- Q：image_tokens，K/V：slot_tokens
- 每个 image token 对所有 slot 做注意力 → 软分配
- 返回**注意力权重** `[B, N_image, P]`（经 softmax 归一化）→ 累计为 assign_maps

**自注意力层（SelfAttentionLayer）**：
- 将 `[image_tokens, slot_tokens]` 拼接后做 joint self-attention
- image 和 slot 部分分别经过独立的 FFN
- 两个流都被更新，slot_tokens 吸收全局上下文

#### 输出

```
slot_features  [B, P, 1024]      # 每个槽的聚合特征
assign_maps    [B, P, H_p, W_p]  # 软分配图（跨帧平均的注意力权重）
```

assign_maps 的物理含义：每个 patch 属于哪个零件槽的概率分布（行求和为 1）。

---

### 3.5 铰链运动头：ArticulationHead（`articulation_head.py`）

#### 架构

```
slot_features [B, P, D]
       ↓
    backbone MLP（共享，每个 slot 独立）
       ├──→ bbox_head (Linear → 6)  → bbox_center [B,P,3] + bbox_size [B,P,3]
       ├──→ type_head (Linear → 2)  → motion_type_logits [B,P,2]
       ├──→ axis_head (Linear → 3)  → axis [B,P,3]（L2 归一化）
       └──→ pivot_head(Linear → 3)  → pivot [B,P,3]（sigmoid 缩放到 [-r,r]³）

timestamps [B, S]
       ↓ 与 slot_features 拼接后输入 scalar_mlp
    scalar_mlp（时间戳条件化 MLP）
       └──→ scalars [B, P, S]（tanh → [-1, 1]）
```

#### 各输出语义

| 输出 | 形状 | 语义 |
|------|------|------|
| `motion_type_logits` | [B, P, 2] | 关节类型 logits（0=平移, 1=旋转） |
| `axis` | [B, P, 3] | 归一化关节轴向量 |
| `pivot` | [B, P, 3] | 关节枢轴点（旋转关节有效） |
| `scalars` | [B, P, S] | 每帧运动量（∈ [-1,1]，乘以 max 得到真实量） |
| `bbox_center` | [B, P, 3] | 各零件包围盒中心（世界坐标） |
| `bbox_size` | [B, P, 3] | 包围盒半尺寸（> 0） |

#### Slot 0（静态底座）特殊处理
- 类型 logits **参与预测但排除**于 CE 损失
- 包围盒 **仍然监督**（用于约束高斯中心）
- Scalars **强制为 0**（不移动）
- 死槽检测时不施加稀疏性惩罚

---

### 3.6 三维高斯头：ArtGaussianHead（`art_gaussian_head.py`）

#### 架构

每个槽独立一个 `SlotGaussianMLP`：

```
slot_feature [B, D]
       ↓ MLP（Linear → LayerNorm → GELU → Linear）
       ↓
  raw_output [B, N_g, 14]
       ↓
  解码为 Gaussian 参数
```

每个高斯的 14 维参数：

| 分量 | 维度 | 激活 | 含义 |
|------|------|------|------|
| mu | 3 | sigmoid → bbox 约束 | 三维位置（规范空间） |
| rot | 4 | normalize | 单位四元数 (w,x,y,z) |
| scale | 3 | exp(raw + init) | 正尺度（初始化偏小） |
| color | 3 | sigmoid | RGB ∈ [0,1] |
| opacity | 1 | sigmoid | 不透明度（初始化偏小） |

初始化偏置：
- `opacity_bias = -2.0`（初始近透明）
- `scale_init_log = -4.0`（初始极小，稳定训练）

包围盒约束（若提供 bbox_center/size）：
```
mu = bbox_center + bbox_size * (2 * sigmoid(raw_mu) - 1)
```
高斯中心被约束在预测的包围盒内，防止 Gaussian 飘散。

输出：
```
gs_mu      [B, P, N_g, 3]
gs_rot     [B, P, N_g, 4]
gs_scale   [B, P, N_g, 3]
gs_color   [B, P, N_g, 3]
gs_opacity [B, P, N_g, 1]
```

---

### 3.7 刚体变换工具（`dggt/utils/rigid_transform.py`）

用于将规范空间高斯按关节参数变换到世界空间（各帧）。

#### Rodrigues 旋转公式（可微分）

```
R = I + sin(θ)[D]× + (1 - cos(θ))[D]×²

其中 [D]× 是轴向量 axis 的反对称矩阵（skew-symmetric）
```

#### 旋转关节变换

```python
angle = scalar × max_angle   # max_angle = 2π
R = rodrigues(axis, angle)
pts_world = pivot + R @ (pts - pivot)
quats_world = R_quat ⊗ quats
```

#### 平移关节变换

```python
translation = scalar × max_translation × axis  # max_translation = 1.0
pts_world = pts + translation
quats_world = quats  # 不变
```

#### 训练时软混合（保持可微性）

```
p_s, p_p, p_r = softmax(motion_type_logits)  # 各类型概率

pts_world = p_s × pts_static + p_p × pts_prismatic + p_r × pts_revolute
quats_world = normalize(p_s × q_static + p_p × q_prismatic + p_r × q_revolute)
```

推理时使用 hard argmax（`apply_rigid_transform_hard`）。

---

## 4. 数据集

**文件**：`datasets/articulated_dataset.py`

### 支持的数据格式

**Format A（单相机）**：
```
data_root/object_id/
├── images/              *.jpg
├── intrinsics.txt       3×3 矩阵
├── extrinsics/          {frame_id}.txt，4×4 矩阵
├── joint_params.json    {"joint_0": {...}, ...}
├── joint_angles.json    {"000": {"joint_0": angle, ...}, ...}
└── part_masks/          {frame_id}_{part_id}.png
```

**Format B（多相机）**：
```
data_root/object_id/
├── cam_00/ cam_01/ cam_02/ cam_03/  各含 images/intrinsics/extrinsics/part_masks
├── joint_params.json
└── joint_angles.json
```

默认排除 `cam_00`（可配置）。

### 掩码 ID 与槽分配约定

```
Part ID (掩码文件编号) → 槽索引
  part_id = k + 2  (joint_{k-1} 的运动零件, k=1..n_joints)
  part_id = n_joints + 2  (静态根部, 合并到 Slot 0)
  背景像素  → Slot 0

Slot 0 = 静态背景 + 静态根部
Slot k (k=1..n_joints) = joint_{k-1} 的运动零件
```

### 数据增强与预处理

- 图像缩放到 `target_size=518`（DINOv2 标准输入）
- 内参随图像尺寸等比例缩放
- 时间戳归一化到 `[0, 1]`
- 运动标量归一化：每个关节在当前场景内归一化到 `[-1, 1]`
- 数据集切分：在**场景级别**按 `val_ratio=0.15` 随机划分（防止泄漏），种子固定

### `__getitem__` 返回字典

```python
{
    "images":          [S, 3, H, W],     # 图像帧序列
    "extrinsics":      [S, 4, 4],        # cam-to-world 外参
    "intrinsics":      [3, 3],           # 相机内参
    "timestamps":      [S],              # 归一化时间戳
    "part_masks":      [S, P, H, W],     # GT 分割掩码（Phase 1）
    "pseudo_masks":    [S, P, H, W],     # 伪掩码（Phase 2，否则全零）
    "has_pseudo_masks": bool,
    "gt_motion_type":  [P],              # {0=static,1=prismatic,2=revolute}
    "gt_axis":         [P, 3],           # GT 关节轴
    "gt_pivot":        [P, 3],           # GT 枢轴点
    "gt_scalars":      [P, S],           # GT 运动标量（归一化）
    "has_pose":        bool,
    "scene_id":        str,
    "n_active_parts":  int,
}
```

---

## 5. 损失函数

### 5.1 掩码损失（Mask Loss）

**位置**：`train_art.py::mask_loss()`

```
前景加权像素级交叉熵损失

Label map：将每个 GT 掩码像素标记为其匹配的槽编号
FG 权重 = fg_weight（默认 10.0），BG 权重 = 1.0

Loss = Σ_pixels [weight × NLL(pred_softmax, label)] / Σ_pixels weight
```

前景加权的意义：模型初期容易把所有像素路由到 Slot 0（背景），10× 加权迫使模型关注前景零件。

### 5.2 稀疏性损失（Sparsity Loss）

**位置**：`dggt/utils/dead_slot_gating.py::slot_sparsity_loss()`

```
L1 正则化，仅施加于动态槽（Slot 1..P-1），Slot 0 豁免

Loss = l1_weight × mean(assign_maps[1:].sum(patch dims))
```

权重调度：Warmup 阶段 = 0.001，Post-warmup = 0.1

### 5.3 运动学损失（Kinematic Loss）

**位置**：`train_art.py::kinematic_loss()`

| 子损失 | 公式 | 备注 |
|--------|------|------|
| 类型 CE | CrossEntropy(logits_2class, gt_2class) | GT 映射：1→0, 2→1；Slot 0 排除；死槽掩码 |
| 轴向损失 | 1 - \|cos(pred_axis, gt_axis)\| | 对称，符号修正后计算 |
| 枢轴 MSE | \|\|pred_pivot - gt_pivot\|\|² | 死槽掩码 |
| 标量 MSE | \|\|pred_scalar - gt_scalar\|\|² | Slot 0 标量强制为 0 |

**符号歧义修正**（`hungarian_matching.py::fix_axis_sign_ambiguity()`）：
轴向量存在 180° 方向歧义（+axis 和 -axis 几何等价），通过：
```python
dot = (pred_axis · gt_axis)
if dot < 0:
    pred_axis = -pred_axis
    pred_scalar = -pred_scalar  # 同时翻转，保持方向一致性
```

### 5.4 死槽不透明度损失

**位置**：`dggt/utils/dead_slot_gating.py::dead_slot_opacity_loss()`

```
对分配质量 < threshold_fraction × N_patches 的槽，
强制其所有高斯不透明度趋近于零：

Loss = l1_weight × mean(opacity × dead_mask)
```

### 5.5 Alpha 渲染损失（Per-part Alpha Render Loss）

**位置**：`train_art.py::per_part_alpha_render_loss()`

无需 gsplat，纯可微渲染：

1. 对每个（槽, 帧）：
   - 将规范空间高斯中心按关节参数变换到世界坐标
   - 投影到相机平面（patch 分辨率）
   - 用各向同性高斯核累积 alpha 图：
     ```
     alpha[u,v] = Σ_g opacity_g × exp(-||[u,v] - [u_g,v_g]||² / (2σ²))
     ```
2. 损失 = Dice Loss + BCE Loss（与 GT 掩码对比）

### 5.6 包围盒投影损失（BBox Loss）

**位置**：`train_art.py::bbox_loss()`

通过 2D 投影间接监督 3D 包围盒位置：
1. 计算 GT 掩码的 2D 质心（跨帧平均）
2. 将预测的 `bbox_center` 投影到图像平面
3. MSE（归一化坐标）× GT 掩码面积权重

### 5.7 伪掩码损失（Phase 2）

同掩码损失（`mask_loss`），但使用 SAM2 生成的伪掩码，权重 `w_pseudo_mask = 0.05`。

### 5.8 损失权重配置

```python
cfg.w_type           = 1.0    # 关节类型 CE
cfg.w_axis           = 1.0    # 轴向损失
cfg.w_pivot          = 1.0    # 枢轴 L2
cfg.w_scalar         = 1.0    # 运动标量 MSE
cfg.w_dead_opacity   = 1.0    # 死槽不透明度
cfg.w_render         = 1.0    # Alpha 渲染（Dice+BCE）
cfg.w_bbox           = 0.1    # 包围盒投影
cfg.w_pseudo_mask    = 0.05   # Phase 2 伪掩码
cfg.l1_sparsity_warmup = 0.001
cfg.l1_sparsity        = 0.1
```

---

## 6. 训练流程

### 三阶段训练策略

#### Phase 1a：热身（Warmup）

**目标**：先学会把图像分解成合理的零件槽，再引入关节监督。

| 配置项 | 值 |
|--------|-----|
| 激活损失 | 掩码 CE + 稀疏性 |
| 冻结模块 | ArticulationHead、GaussianHead |
| 训练模块 | Aggregator、PartSlotRouter |
| 稀疏权重 | 0.001 |
| 终止条件 | val IoU ≥ `warmup_iou_threshold` 连续 3 次 |

#### Phase 1b：全监督

**目标**：在 PartNet-Mobility 上完整监督所有关节参数。

| 配置项 | 值 |
|--------|-----|
| 激活损失 | 全部（掩码+稀疏+运动学+渲染+包围盒） |
| 冻结模块 | DINOv2（Aggregator 内部） |
| 训练模块 | 所有 heads |
| 稀疏权重 | 0.1 |

每步都进行 **匈牙利匹配** + **轴符号修正**，然后计算损失。

#### Phase 2：弱监督域自适应

**目标**：迁移到真实数据，利用 SAM2 伪掩码。

| 配置项 | 值 |
|--------|-----|
| 数据 | 真实视频 + SAM2 伪掩码 |
| CameraHead | 激活（预测相机姿态） |
| Plücker 射线 | stop-gradient（防止伪梯度） |
| Aggregator | 仅解冻最后 4 个 block |
| 伪掩码权重 | 0.05 |

### 损失计算流水线

```python
def compute_loss(preds, batch, step, cfg, is_warmup):

    # 1. 匈牙利匹配：将预测槽分配给 GT 零件
    matches = batch_hungarian_match(assign_maps, gt_masks)

    if is_warmup:
        # 热身：只优化分割
        loss = mask_loss(...) + sparsity_loss(weight=0.001)
        return loss

    # Post-warmup：完整损失
    loss = (
        mask_loss(...)                      # 分割掩码
      + sparsity_loss(weight=0.1)           # 槽稀疏性
      + kinematic_loss(...)                 # 类型+轴+枢轴+标量
      + dead_slot_opacity_loss(...)         # 死槽透明
      + per_part_alpha_render_loss(...)     # 可微渲染
      + bbox_loss(...)                      # 包围盒投影
      + (pseudo_mask_loss(...) if phase2)   # 伪掩码
    )
    return loss
```

### 槽的生命周期管理（Dead Slot Gating）

`dggt/utils/dead_slot_gating.py` 负责识别和管理"死槽"（空槽）：

```
检测：分配质量 < 0.005 × N_patches → 该槽标记为死槽

死槽的处理：
  - 运动学损失 → 掩码为 0（不反向传播）
  - 渲染损失   → 跳过
  - 不透明度   → 额外 L1 惩罚（强制透明）
  - 稀疏损失   → 始终施加（L1 正则化）
```

Slot 0（静态底座）**永远不标记**为死槽，且豁免稀疏性惩罚。

---

## 7. 推理与评估

### 推理流程（`inference_art.py`）

1. 加载 checkpoint，设置 `model.set_phase("1b")`
2. 前向传播，获取所有预测
3. 运动类型：`argmax(motion_type_logits)` → 0 或 1
4. 高斯变换：对每帧调用 `apply_rigid_transform_hard`，使用硬选择
5. 可视化：8 色槽覆盖、每零件详细图（图像/预测掩码/GT 掩码/热力图）
6. 输出每场景 JSON 报告

### 评估指标（`eval_phase1b.py`）

| 指标 | 公式 | 说明 |
|------|------|------|
| Mask IoU | inter / union | 基于 argmax 二值化（非阈值） |
| Motion Type Acc | 2-class accuracy | 仅评估前景活动零件（GT type ≥ 1） |
| Axis Cosine Sim | \|dot(pred_axis, gt_axis)\| | 绝对值（对称） |
| Axis Error (deg) | arccos(\|cos_sim\|) × 180/π | 越小越好 |
| Scalar MAE | mean \|pred_scalar - gt_scalar\| | 每帧运动量误差 |
| Pivot L2 | \|\|pred_pivot - gt_pivot\|\|₂ | 枢轴点距离 |

**重要实现细节**：
- 匹配使用 **argmax 二值化**（非 0.5 阈值）：softmax 输出均值约 1/P，阈值 0.5 会导致大部分 patch 不被任何槽认领
- 运动类型评估只针对 `gt_motion_type ≥ 1` 且 `mask_sum > 0` 的活动零件
- 使用与训练完全相同的 val_ratio 和 split_seed，保证无数据泄漏

---

## 8. 张量形状速查表

| 变量 | 形状 | 说明 |
|------|------|------|
| `images` | [B, S, 3, H, W] | 输入图像序列 |
| `extrinsics` | [B, S, 4, 4] | cam-to-world |
| `intrinsics` | [B, 3, 3] | 相机内参 |
| `timestamps` | [B, S] | 归一化时间戳 |
| `agg_tokens` | [B, S, P_total, 2048] | 聚合编码 |
| `dino_tokens` | [B, S, P_total, 1024] | DINOv2 原始特征 |
| `plucker_rays` | [B, S, N_patches, 6] | Plücker 射线 |
| `slot_features` | [B, P, 1024] | 槽聚合特征 |
| `assign_maps` | [B, P, H_p, W_p] | 软分配图 |
| `motion_type_logits` | [B, P, 2] | 关节类型 logits |
| `axis` | [B, P, 3] | 关节轴（归一化） |
| `pivot` | [B, P, 3] | 枢轴点 |
| `scalars` | [B, P, S] | 每帧运动量 |
| `bbox_center` | [B, P, 3] | 包围盒中心 |
| `bbox_size` | [B, P, 3] | 包围盒半尺寸 |
| `gs_mu` | [B, P, N_g, 3] | 高斯中心 |
| `gs_rot` | [B, P, N_g, 4] | 高斯旋转（四元数） |
| `gs_scale` | [B, P, N_g, 3] | 高斯尺度 |
| `gs_color` | [B, P, N_g, 3] | 高斯颜色 |
| `gs_opacity` | [B, P, N_g, 1] | 高斯不透明度 |

> 符号约定：B=batch size, S=帧数, P=槽数(默认8), N_g=高斯数(默认256),
> H/W=图像分辨率, H_p/W_p=patch 分辨率 (H/14, W/14), P_total=所有 patch + special token 数

---

## 9. 数据流图

```
images [B,S,3,518,518]
timestamps [B,S]
extrinsics [B,S,4,4]  (Phase1: GT, Phase2: predicted)
intrinsics [B,3,3]
         │
         ▼
  ┌─────────────────────────────────┐
  │         Aggregator              │
  │  DINOv2(frozen) + Aggregation   │
  └────────┬────────────────────────┘
           │ agg_tokens[B,S,Pt,2048]
           │ dino_tokens[B,S,Pt,1024]
           │
    ┌──────▼──────┐
    │ CameraHead  │(Phase2)→ pose_enc[B,S,9]→extrinsics
    └─────────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │    Plücker Ray Computation      │
  │  plucker_rays [B,S,N_p,6]      │
  └────────┬────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │         PartSlotRouter          │
  │  8层 Transformer (6×cross+2×self)│
  │                                  │
  │  image_tokens → cross-attn → slot_tokens
  │             ↓ self-attn ↑        │
  │  assign_maps (注意力权重)         │
  └──────┬───────────┬───────────────┘
         │           │
  slot_features   assign_maps
  [B,P,1024]      [B,P,H_p,W_p]
         │
    ┌────┴───────────────────┐
    │                        │
    ▼                        ▼
┌───────────────┐   ┌────────────────┐
│ArticulationH  │   │ ArtGaussianH   │
│               │   │                │
│backbone MLP   │   │ Per-slot MLP   │
│  ├ type[B,P,2]│   │ [B,D]→[B,N_g,14]
│  ├ axis[B,P,3]│   │                │
│  ├ pivot[B,P,3│   │ gs_mu/rot/     │
│  ├ scalar[B,P,│   │ scale/color/   │
│  └ bbox[B,P,6]│   │ opacity        │
└───────┬───────┘   └───────┬────────┘
        │                   │
        └──────┬────────────┘
               ▼
    ┌─────────────────────┐
    │  Rigid Transform    │  (训练时软混合，推理时硬选择)
    │  canonical → world  │
    │  for each frame     │
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │   Loss Computation  │
    │  1. Hungarian Match │
    │  2. mask_loss       │
    │  3. kinematic_loss  │
    │  4. render_loss     │
    │  5. bbox_loss       │
    │  6. sparsity_loss   │
    └─────────────────────┘
```

---

*文档生成时间：2026-03-29*
*代码分支：art*
