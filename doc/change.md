### train_art.py — GT 排序 reorder 修复（关键 Bug Fix）

**问题**: `per_part_alpha_render_loss` 和 `bbox_loss` 直接用 slot 索引 `p` 对应 GT 通道 `p`，但 Hungarian 匹配允许 assign_maps 自由排列（slot 1 可能匹配 GT 零件 3）。这导致监督信号冲突：GT mask 通道 p 在监督 pred slot p 的 GS，但 assign_maps 却把这个 slot 分配到其他零件位置。**后果**: 非背景 slot 全部退化为 dead（ckpt_074000 仅 slot 0 有 opacity > 0）。

**修复位置**: `train_art.py:670-721`

**修复方法**: 在计算 per-part 损失前，将预测张量按 Hungarian 匹配结果重排到 GT 顺序：

```python
def _to_gt_order(t):
    out = reorder_by_match(t, matches, P_gt)  # 按匹配重排
    out[:, 0] = t[:, 0]                        # Slot 0 (静态基座) 直接透传
    return out

gs_mu_gt       = _to_gt_order(preds["gs_mu"])
gs_opacity_gt  = _to_gt_order(preds["gs_opacity"])
pivot_gt       = _to_gt_order(preds["pivot"])
mtl_gt         = _to_gt_order(preds["motion_type_logits"])
bbox_center_gt = _to_gt_order(preds["bbox_center"])
bbox_size_gt   = _to_gt_order(preds["bbox_size"])
# axis_fixed / scalar_fixed 已由 match_and_fix 输出 GT 排序
```

未匹配的 GT 通道（填充槽，零件数少于 max_parts 的场景）被标记为 dead，损失跳过。

**注意**: `global_render_loss`（gsplat 合成渲染）合成整幅图像，slot 顺序无关紧要，仍使用 pred-slot 顺序。

---

### 5. train_art.py — mask_loss 改进

**改动**: Mask 损失从单纯 NLL（交叉熵）改为 **NLL + Dice** 联合损失。

```python
# NLL: 前景像素权重 ×10，推动 P(correct_slot|pixel) 上升
nll = F.nll_loss(log_pred[b], label_map, reduction="none")
total += (nll * pixel_weight).sum() / pixel_weight.sum()

# Dice: 直接优化 soft-mask overlap ≈ IoU
dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
total += dice_weight * (1 - dice)
```

**意义**: NLL 允许预测扩散（diffuse），Dice 直接约束 IoU，联合使用突破了约 0.67 的 IoU 瓶颈。

---

### 6. train_art.py — Frame 0 GT mask 固定

**改动**: mask_loss 只用第 0 帧的 GT mask（`batch["part_masks"][:, 0]`）。

**位置**: `train_art.py:587-589`

**意义**: 数据集第 0 帧强制为静止初始状态（rest pose），零件遮罩位置明确，无运动模糊；assign_maps 也使用第 0 帧 cross-attention 权重。这消除了多帧 GT 间不一致导致的梯度冲突。

---

### 7. train_art.py — Hungarian 匹配 + 轴符号修复

**位置**: `dggt/utils/hungarian_matching.py`

**Hungarian 匹配**:
- 代价矩阵: `C[p, g] = 1 - IoU(pred_bin_p, gt_mask_g)`
- 仅匹配 Slot 1..P-1（前景）→ GT 1..P_gt-1（跳过 Slot 0 静态基座）
- 跳过全零 GT mask（填充零件）

**轴符号修复** (`fix_axis_sign_ambiguity`):
- 若 `dot(pred_axis, gt_axis) < -threshold`（默认 0.3），翻转轴和 scalar
- 置信度阈值防止 dot≈0 时（两轴近似垂直，匹配不可靠）随机翻转——这是训练初期 scalar MSE 暴涨到 4.0 的根本原因

---

### 8. train_art.py — 训练阶段设计

#### Phase 1a（Warmup）
- 仅激活 mask 损失（NLL+Dice）+ slot sparsity 损失
- KinematicHead、GaussianHead 冻结，不参与梯度更新
- **Warmup gate**: 验证集 mean-IoU ≥ `warmup_iou_threshold`（默认 0.65）连续 3 次检查通过后，解冻所有头部，保存 warmup checkpoint，重建优化器

#### Phase 1b（全监督）
- 全部头部解冻
- 损失: mask + sparsity + kinematic(type/axis/pivot/scalar) + dead_opacity + **per_part_alpha_render** + **bbox** + pose_enc
- 以 warmup checkpoint 为起点，`--reset_scheduler` 重置学习率

#### Phase 2（弱监督域适应）
- 使用真实数据，CameraHead 预测相机位姿（Plücker rays stop-grad 防污染 slot router）
- Aggregator 早期层冻结（仅保留最后 4 层可训练）
- 伪 mask（SAM2 生成）权重 0.05

---

### 9. train_art.py — 训练工程改进

#### BF16 训练
- `--use_bf16` 启用 `autocast(dtype=torch.bfloat16)`
- BF16 指数位宽与 FP32 相同（无溢出风险），Ampere+ GPU 约 1.5-2× 加速

#### DDP NaN 同步
- NaN/Inf 检查通过 `dist.all_reduce(loss_ok, ReduceOp.MIN)` 跨所有 rank 同步
- 所有 rank 同时跳过该 batch，防止部分 rank 跳过时 DDP all_reduce 死锁

#### Gradient Checkpointing
- `--gradient_checkpointing` 对 Aggregator attention blocks 启用
- 以约 33% 额外算力换取显著显存节省（40GB A100 上 batch_size=1 必需项）
- 使用 `checkpoint(..., use_reentrant=False)` + 批大小限制为 1（batch_size>1 与 gradient_checkpointing 在此模型存在 PyTorch 兼容性问题）

#### CosineAnnealingLR
- `T_max=cfg.total_steps`，确保 LR 曲线与训练步数对齐（避免 `last_epoch > T_max` 溢出）

#### 其他
- `--exclude_cams`: 排除特定相机（默认排除 `cam_00` 背面视角）；传入空列表包含所有相机
- `--partial_resume`: 仅恢复 aggregator + camera_head 权重（用于架构变更后的迁移）
- `--reset_slot_tokens`: 跨 phase 恢复后重新初始化 slot 令牌，打破单 slot 主导局部极值

---

### 10. eval_gs.py — BF16 推理修复

**改动**: 推理时同样使用 `autocast(dtype=torch.bfloat16)`，与训练一致，消除因精度不匹配导致的 NaN。

**位置**: `eval_gs.py:209`

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
    preds = model(images, extrinsics, intrinsics, timestamps)
```

---

## 训练命令参考

### Phase 1a（从零开始）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 train_art.py \
    --phase 1a \
    --total_steps 50000 \
    --warmup_iou_threshold 0.65 \
    --use_bf16 \
    --gradient_checkpointing \
    --batch_size 1 \
    --num_workers 8 \
    --output_dir /data2/cyt/checkpoints/art_v21_phase1a \
    --data_root /data2/cyt/data_root_refine
```
> `--exclude_cams` 省略表示包含所有相机（cam_00 ～ cam_03）

### Phase 1b（从 warmup checkpoint 开始）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 train_art.py \
    --phase 1b \
    --resume /data2/cyt/checkpoints/art_v21_phase1a/ckpt_warmup_XXXXXX.pth \
    --reset_scheduler \
    --total_steps 100000 \
    --use_bf16 \
    --gradient_checkpointing \
    --num_workers 8 \
    --output_dir /data2/cyt/checkpoints/art_v21_phase1b \
    --data_root /data2/cyt/data_root_refine
```

---

## 已知问题与说明

| 问题 | 根因 | 状态 |
|------|------|------|
| batch_size=2 + gradient_checkpointing → `grad_input must be contiguous` | PyTorch checkpoint(use_reentrant=False) 与非连续内存的已知兼容性问题 | 固定 batch_size=1 规避 |
| slot collapse（仅 slot 0 有 opacity）| per_part 损失未 reorder 到 GT 顺序 | 已修复（§4）|
| scalar MSE 暴涨至 4.0 | 轴符号 flip 在 dot≈0 时随机翻转 | 已修复（置信度阈值 §7）|
| Phase 1a IoU 停在 ~0.67 | 纯 NLL 无法直接优化 IoU | 已修复（NLL+Dice §5）|
