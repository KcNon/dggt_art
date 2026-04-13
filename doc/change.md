# ArtVGGT 改动记录

---

## 2026-04-12：BF16 混合精度 + 训练加速优化

### 动机

Phase 1a 训练时 GPU 显存有剩余（~24GB/40GB），希望加速训练。

### 改动

1. **BF16 autocast**（`train_art.py`）：
   - 在 forward+loss 外包裹 `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
   - 新增 `--use_bf16` 命令行参数
   - BF16 与 FP16 不同，指数位宽与 FP32 一致（8 bit），不会溢出，不需要 GradScaler
   - A100 Tensor Core 上矩阵乘法 bf16 比 fp32 快 ~2x

2. **num_workers 4→8**（`scripts/launch_phase1a.sh`, `scripts/launch_phase1b.sh`）

3. **保留 gradient_checkpointing**：
   - 尝试去掉后 OOM（40GB 不够存完整激活），必须保留

4. **num_frames 保持 6**：
   - 尝试 6→8 后 OOM，全局注意力内存按 (S×P)² 缩放，8 帧约增加 1.78x 注意力内存

### 效果

GPU 显存 24GB → 29GB，训练速度因 BF16 提升。

---

## 2026-04-12：引入 canonical rest state，scalar 归一化以静止状态为零点

### 动机

原先 `_normalize_scalars` 用 min-max 归一化到 [-1, 1]，scalar=0 对应运动范围中点，
没有物理意义。数据集中 frame 0 始终为静止状态（抽屉关闭、门关上等），但模型不知道。

这导致：
- canonical GS 空间无明确物理对应（既非第一帧也非静止状态）
- scalar=0 代表运动中点而非静止，跨实例不一致
- 推理时无法通过 scalar=0 可靠地输出静止状态

### 改动

`datasets/articulated_dataset.py` 中的 `_normalize_scalars`：
- 旧：min-max → [-1, 1]，scalar=0 = 运动中点
- 新：以 `values[0]`（rest state，即 frame 0 的角度值）为零点，最大偏移量映射到 ±1

效果：scalar=0 明确对应静止状态，canonical GS 空间与 rest pose 对齐。

### 对训练的影响

仅改变 `gt_scalars` 监督信号。Phase 1a warmup 阶段 ArticulationHead 冻结，
scalar loss 未激活，因此可以从现有 checkpoint（ckpt_006000）无缝恢复。

---

## 2026-04-12：launch_phase1a.sh 同步实际运行参数

将脚本参数与当前实际训练配置对齐：
- `num_frames`: 8 → 6（8 帧 OOM）
- `total_steps`: 15000 → 20000
- 添加 `--gradient_checkpointing`（去掉会 OOM）
- 添加 `--resume ckpt_006000.pth`

---

## 2026-04-13：warmup IoU 平台期诊断与修复

### 问题现象

Phase 1a warmup IoU 从 step 8000 起长期卡在 0.61-0.62，持续 12000 步无改善，
LR 重置（`--reset_scheduler`）也无效。

### 根本原因分析

**原因一（主因）：assign_maps 与 GT mask 的时间聚合方式不一致**

| | 实现 |
|---|---|
| 预测 assign_maps | `attn_weights.mean(dim=1)`：跨所有 S 帧平均 |
| GT mask（训练 & 评估）| `part_masks.max(dim=1)`：跨所有 S 帧取 union |

对于一个移动中的零件（如抽屉从关闭到打开）：
- GT mask（max）= 零件在所有帧出现位置的**并集**（大区域）
- 预测（mean）= 各帧 attention 的平均值，在零件只在 frame 0 出现的像素处，
  5/6 帧的 background slot attention 拉高了 mean，导致该像素被判为背景

结果：即使模型对每帧单独预测正确，IoU 也因跨帧平均而系统性地被压低到 ~0.62。

**原因二：NLL loss 与 IoU 指标不完全对齐**

NLL 优化每像素的 `P(correct_slot | pixel)`，允许预测概率分散（0.40 vs 0.35 也满足）。
IoU 需要 `argmax` 在整个 mask 区域一致正确，对覆盖率更敏感。模型找到 NLL 的局部最优
（mask loss ≈ 0.02）但 IoU 仍停在 0.67，两者之间存在本质差距。

### 修复

**修复 1：assign_maps 改用 frame 0（`dggt/heads/part_slot_router.py`）**

```python
# 旧：跨帧平均
assign = last_attn_weights.reshape(B, S, N_patches, self.num_slots)
assign = assign.mean(dim=1)

# 新：仅取 frame 0（canonical rest state）
assign = last_attn_weights.reshape(B, S, N_patches, self.num_slots)
assign = assign[:, 0, :, :]
```

Frame 0 是静止状态，零件在固定位置，GT mask 是单一位置的干净二值 mask，
预测与监督完全一致，消除了时间聚合的歧义。

这不影响下游（ArticulationHead/GaussianHead）的语义：assign_maps 定义
canonical GS 布局，静止状态下初始化 GS 点位置是正确的。

**修复 2：GT mask 训练与评估同步改为 frame 0（`train_art.py`）**

```python
# 旧
gt_masks = (batch["part_masks"].max(dim=1).values > 0.5).float()
# 新（compute_loss 和 eval_mean_iou 均修改）
gt_masks = (batch["part_masks"][:, 0] > 0.5).float()
```

**修复 3：增加 Dice loss（`train_art.py` `mask_loss`）**

在原有 foreground-weighted NLL 基础上，对每个匹配的 (slot, part) 对增加 Dice 项：

```python
dice = (2 * (pred_mask * gt_mask).sum()) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
loss += dice_weight * (1.0 - dice)   # dice_weight = 2.0
```

Dice loss 直接优化 soft mask 与 GT 的重叠比例，数学形式近似 IoU，
弥补 NLL 对覆盖率不敏感的缺陷。

### 效果

| 阶段 | 改动 | warmup IoU |
|------|------|-----------|
| step 0–8000 | 原始 | 0.51 |
| step 8000–20000 | 原始（LR 衰减后平台） | 0.61–0.62 |
| step 20000–30000 | frame 0 修复 | 0.65–0.68（峰值 0.676） |
| step 30000–（进行中）| frame 0 + Dice loss | 待观测 |

### 训练配置变化

- `total_steps`: 20000 → 30000 → 40000（分阶段延长）
- 每次 resume 均使用 `--reset_scheduler` 重置 LR 调度器
- 当前 resume 起点：`ckpt_030000.pth`
