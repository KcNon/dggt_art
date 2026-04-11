# ArtVGGT 改动记录

---

## 2026-04-11：Slot Tokens 移入 Aggregator（跨帧全局注意力）

### 背景与动机

原始 `PartSlotRouter` 在 Aggregator 输出之后单独维护可学习的 `slot_tokens`，
通过独立的 cross/self-attention 层与 image tokens 交互。这导致 slot 只能看到
Aggregator 已处理完毕的静态特征，无法在 Aggregator 的多帧全局注意力中动态更新。

论文原意是：slot tokens 应与 image tokens 一起参与 transformer 层，通过
self-attention 共享全局上下文，通过 cross-attention 显式路由视觉信息。

### 改动内容

#### 1. `dggt/models/aggregator.py`

- `__init__` 新增参数 `num_slots: int = 0`
- 当 `num_slots > 0` 时，初始化 `self.slot_tokens [1, P, C]` 可学习参数（std=0.02）
- `_process_global_attention` 新增 `slot_tokens` 参数：
  - 将 slot tokens 拼接到图像 token 序列末尾：`[B, S*P_img + P_slot, C]`
  - Slot tokens 赋零 RoPE 位置（无空间先验）
  - 经过 global block 处理后分离出更新后的 slot states
  - Intermediates 只包含图像 tokens，保持 `output_list` 形状不变
- `forward` 方法：
  - 初始化 `slot_states = self.slot_tokens.expand(B, -1, -1)`
  - 每次 global attention 调用时传入并更新 `slot_states`
  - 返回值从 5 个扩展为 6 个，末尾新增 `slot_states [B, P, C]`

**数据流：**
```
frame_blocks:   [B*S, P_img, C]                  (slot 不参与帧内注意力)
global_blocks:  [B, S*P_img + P_slot, C]  ──►  slot 在跨帧注意力中更新
                                                    ↓
                                           slot_states [B, P_slot, C]
```

#### 2. `dggt/heads/part_slot_router.py`

- 移除 `self.slot_tokens` 可学习参数（已移入 Aggregator）
- `forward` 新增参数 `slot_init: torch.Tensor | None`
- 使用 `slot_init`（来自 Aggregator 的 slot states）作为 slot 初始状态，
  PSR 继续负责用 plucker rays、timestamps、image tokens 进一步精化
- 保留全部 cross/self-attention 层和 `assign_maps` 生成逻辑不变

#### 3. `dggt/models/art_vggt.py`

- `Aggregator` 构造时传入 `num_slots=num_slots`
- `forward` 解包 Aggregator 第 6 个返回值 `slot_states`
- `PartSlotRouter` 调用时传入 `slot_init=slot_states`

### 效果

| 方面 | 旧设计 | 新设计 |
|------|--------|--------|
| Slot 获取多帧信息的时机 | Aggregator 输出之后（静态） | Aggregator 内部每个 global block（动态更新） |
| Slot 间相对位置感知 | PSR 的 self-attn 层 | Aggregator global_blocks 中与 image tokens 联合 |
| PSR 的职责 | 生成 slot_features + assign_maps | 精化 slot_features + 生成 assign_maps |
| 参数量变化 | PSR 有 slot_tokens | Aggregator 有 slot_tokens（移动，不增加） |

---

## 2026-04-11：训练配置更新（art_v20）

### 数据集
- 切换到优化后的数据集 `/data2/cyt/data_root_refine`（更好的视角覆盖）
- 数据已做视角优化，**移除 `--exclude_cams cam_00`**，所有相机视角均参与训练

### Checkpoint 目录
- Phase 1a：`/data2/cyt/checkpoints/art_v20_phase1a`
- Phase 1b：`/data2/cyt/checkpoints/art_v20_phase1b`

### Phase 1a 配置变化
- IoU 阈值：0.55 → **0.75**（确保 mask 分配充分收敛再解冻运动头）
- 去除 `--exclude_cams`

### 启动脚本
- `scripts/launch_phase1a.sh`：Phase 1a 一键启动
- `scripts/launch_phase1b.sh <phase1a_pid>`：Phase 1b watcher，传入 Phase 1a PID
  以避免 GPU OOM（等 Phase 1a 完全退出后再启动）

---

## 2026-04-10：ArtGaussianHead 分支重构：几何 / 外观解耦

## 背景与动机

原始 `ArtGaussianHead` 使用单个 MLP（`SlotGaussianMLP`）同时预测所有 Gaussian 属性：
位置（mu）、旋转（rot）、尺度（scale）、颜色（color）、透明度（opacity）。

引入第一帧图像特征融合后，该 MLP 的输入变为 `cat([slot_features, patch_proj_feat])`，
导致图像特征同时影响几何属性和外观属性，存在两个问题：

1. **mu 本身已由 bbox 约束**（`mu = bbox_center + tanh(raw) * bbox_size`），
   位置预测任务简单，不需要图像特征，图像梯度反而引入噪声。
2. **颜色和透明度直接对应第一帧像素**，图像特征对外观学习最有价值，
   应集中用在外观预测上而非分散给几何。

---

## 改动内容（`dggt/heads/art_gaussian_head.py`）

### 1. 常量拆分

```python
# 旧
GS_DIM = 14  # mu(3)+rot(4)+scale(3)+color(3)+opacity(1)

# 新
GS_DIM_GEO = 10   # mu(3) + rot(4) + scale(3)
GS_DIM_APP = 4    # color(3) + opacity(1)
```

### 2. MLP 类拆分：`SlotGaussianMLP` → `SlotGeometryMLP` + `SlotAppearanceMLP`

| 类 | 输入 | 输出 | 图像特征 |
|----|------|------|----------|
| `SlotGeometryMLP` | `slot_features [B, D]` | `[B, N_g, 10]` | **无** |
| `SlotAppearanceMLP` | `cat([slot_features, patch_proj]) [B, D+256]` | `[B, N_g, 4]` | **有** |

`SlotAppearanceMLP` 使用两层隐藏层（`hidden → hidden`），略小于几何 MLP（`hidden → 2×hidden`），
因为外观任务相对简单。

### 3. `ArtGaussianHead` 参数变化

新增两个 `ModuleList`：

```python
self.slot_geo_mlps = nn.ModuleList([SlotGeometryMLP(dim_in, hidden_dim, n_gaussians) ...])
self.slot_app_mlps = nn.ModuleList([SlotAppearanceMLP(app_dim_in, hidden_dim, n_gaussians) ...])
```

`patch_proj`（共享投影层）保持不变，但其输出**只拼接到** `slot_app_mlps` 的输入。

### 4. Bias 初始化分离

```python
# 几何 MLP：初始化 scale bias → exp(scale_init_log) 近似小 Gaussian
for mlp in self.slot_geo_mlps:
    mlp.mlp[-1].bias[_IDX_SCALE].fill_(scale_init_log)   # 默认 -4.0

# 外观 MLP：初始化 opacity bias → sigmoid(-2) ≈ 0.12（略透明）
for mlp in self.slot_app_mlps:
    mlp.mlp[-1].bias[_IDX_OPACITY].fill_(-2.0)
```

### 5. Forward 逻辑

```python
for p in range(P):
    sf = slot_features[:, p]                         # [B, D]

    # 几何分支：仅用 slot 特征
    geo_list.append(slot_geo_mlps[p](sf))            # [B, N_g, 10]

    # 外观分支：slot 特征 + 第一帧图像特征
    app_in = cat([sf, pooled_proj[:, p]], dim=-1)    # [B, D+256]
    app_list.append(slot_app_mlps[p](app_in))        # [B, N_g, 4]

geo_raw = stack(geo_list, dim=1)   # [B, P, N_g, 10]
app_raw = stack(app_list, dim=1)   # [B, P, N_g, 4]

# 激活函数
mu      = bbox_center + tanh(geo_raw[..., :3]) * bbox_size
rot     = normalize(geo_raw[..., 3:7])
scale   = exp(geo_raw[..., 7:10]).clamp(min=1e-6)
color   = sigmoid(app_raw[..., :3])
opacity = sigmoid(app_raw[..., 3:4])
```

---

## 数据流示意图

```
slot_features [B, P, D]
       │
       ├─────────────────────────────────────────► SlotGeometryMLP × P
       │                                                   │
       │                                            [B, P, N_g, 10]
       │                                           mu / rot / scale
       │
       │          patch_feats_frame0 [B, N_p, 3C]
       │                   │
       │          assign_maps.detach()
       │                   │  weighted pool per slot
       │          pooled [B, P, 3C]
       │                   │  shared patch_proj
       │          pooled_proj [B, P, 256]
       │                   │
       └──── cat ──────────┘
             [B, P, D+256]
                   │
            SlotAppearanceMLP × P
                   │
            [B, P, N_g, 4]
           color / opacity
```

---

## 对训练的影响

| 方面 | 旧设计 | 新设计 |
|------|--------|--------|
| 几何梯度来源 | 渲染损失 + 外观梯度（间接） | **仅**渲染/bbox 损失 |
| 外观梯度来源 | 渲染损失（稀释于几何） | 渲染损失（集中） |
| 图像特征利用率 | 全属性共享，稀释 | 集中在 color/opacity |
| 参数量变化 | 8 × 1 MLP | 8 × 2 MLP（约增加 ~30%） |

---

## 与 `art_vggt.py` 的接口兼容性

`ArtGaussianHead.forward` 签名未变，`art_vggt.py` 无需修改。
Checkpoint 恢复时 `strict=False` 会自动跳过形状不匹配的权重（原 `slot_mlps` → 新 `slot_geo_mlps` / `slot_app_mlps`），新参数从随机初始化开始训练。

---

*修改日期：2026-04-10*
