# v13 训练分析：Bug 定位与优化方向

**分析对象**：`art_v13_phase1b/ckpt_050000.pth`（step 36000→50000）
**对比基线**：`art_v12_phase1b/ckpt_036000.pth`

---

## 一、训练结果对比

| 指标 | v12 @ step36k | v13 @ step50k | 变化 |
|------|--------------|--------------|------|
| Mask IoU | **0.816** | **0.000** | 崩溃 |
| Motion Type Acc | 0.992 | 0.989 | ≈ |
| \|cos(axis)\| | 0.959 | **0.967** | 微升 |
| Axis Angle Error | 5.56° | **4.55°** | 改善 |
| Scalar MAE | 0.081 | **0.067** | 改善 |
| Pivot L2 | 0.048 | **0.040** | 改善 |

**结论**：运动学参数（轴/标量/枢轴）有所改善，但分割图（assign_maps）完全崩溃。实测 assign_maps：
- Slot 0: argmax 覆盖 **100%** 像素，mass=1321
- Slot 1: mass=47（微量），Slot 2-7: mass=0

---

## 二、损失曲线分析

| 步骤区间 | mask | sparsity | axis | scalar | render | bbox | total |
|---------|------|---------|------|--------|--------|------|-------|
| 36k-38k | 2.04 | 0.75 | 0.775 | 0.055 | 1.98 | **0.000** | 5.90 |
| 38k-40k | 1.89 | 0.78 | 0.753 | 0.039 | 1.96 | **0.000** | 5.68 |
| 40k-42k | 1.85 | 0.81 | 0.722 | 0.018 | 1.91 | **0.000** | 5.56 |
| 44k-46k | 1.87 | 0.83 | 0.698 | 0.019 | 1.83 | **0.000** | 5.49 |
| 48k-50k | 1.86 | 0.83 | 0.649 | 0.005 | 1.80 | **0.000** | 5.37 |

**关键异常**：
1. `bbox_loss = 0.000` 贯穿全程（共 280 步，100%）
2. `render_loss` 始终在 1.8-2.4，未收敛
3. `axis_loss` 量化为离散值（0.857, 0.714, 0.500...）

---

## 三、Bug 定位

### Bug 1（根本原因）：坐标系深度符号错误 ★★★

**位置**：[train_art.py:260](../train_art.py#L260)（`bbox_loss`）和 [train_art.py:362](../train_art.py#L362)（`per_part_alpha_render_loss`）

**现象**：
```
Scene 0: bbox depths = ['-2.778', '-2.783', '-2.755', '-2.761'] → behind=8/8
（共 10 个场景，每个场景 8/8 个 slot 均被判定在相机后方）
```

**根因**：PartNet-Mobility 数据集使用 **Blender/OpenGL cam-to-world 约定**，相机前方的点在相机坐标系下 **z < 0**（相机看向 -z 方向）。但代码两处均假设 OpenCV 约定（z > 0 为相机前方）：

```python
# bbox_loss (train_art.py:260)
depth = uv_h[..., 2].clamp(min=0.5)    # ← 实际 z = -4.12（物体在前方）
behind = (uv_h[..., 2] < 0.0)          # ← 全部判定为"在后方" → weight=0 → loss=0

# per_part_alpha_render_loss (train_art.py:362)
depth = mu_cam[..., 2].clamp(min=0.01) # ← depth=-4.12，投影 u=fx*x/(-4.12) 方向翻转
```

**验证**：
```python
# GT pivot 在世界原点 [0,0,0]
P_cam_z = R.T @ ([0,0,0] - [-2,-2,3]) = -4.12   ← 负数，OpenGL 前方正确
# 正确投影（OpenGL）：depth = -(-4.12) = 4.12 → u=259.0, v=259.0（图像中心）✓
# 错误投影（OpenCV）：depth = -4.12 → u=259.0（对称原点凑巧相同，但其他位置翻转）
```

**影响链**：
```
z_cam 全负
  ├─→ bbox_loss: behind 掩码=1 → weight=0 → loss=0（bbox_center 永远不被监督）
  ├─→ bbox_center 预测飘移 → gaussian 位置错误
  ├─→ render_loss: depth=-4.12 → u/v 方向翻转 → alpha 图错位 → loss≈2.0 无法收敛
  └─→ render_loss 向 slot_features 反传错误梯度 → assign_maps 在 sparsity 压力下崩溃
```

**修复**（两处均需修改）：
```python
# bbox_loss
depth = (-uv_h[..., 2]).clamp(min=0.5)          # 取负 z_cam 作为深度
behind = (uv_h[..., 2] > 0.0)                   # 正 z_cam 才是在相机后方（OpenGL）

# render_loss
depth = (-mu_cam[..., 2]).clamp(min=0.01)        # 同样取负
u = fx * mu_cam[..., 0] / depth + cx
v = fy * mu_cam[..., 1] / depth + cy
```

---

### Bug 2：`kinematic_loss` alive 掩码被未匹配 GT 槽污染 ★★

**位置**：[train_art.py:148](../train_art.py#L148)

**现象**：axis_loss 量化为 `6/7≈0.857`、`5/7≈0.714` 等离散值（占 27%、9%...），从未收敛到 0。

**根因**：`reorder_by_match` 对未匹配的 GT 位置用零填充（`tensor.new_zeros`），填充值 `0.0 → bool → False`，即"不是死槽"。这导致没有对应预测的 GT 槽被错误标记为 alive：

```python
is_dead_r = reorder_by_match(is_dead.float().unsqueeze(-1), matches, P_gt)
is_dead_r = is_dead_r.squeeze(-1).bool()   # 未匹配位置填0 → False → alive!
alive = ~is_dead_r                          # 包含了所有零填充的 GT 位置
```

**量化公式验证**：
```
物体有 k 个关节，max_parts=8，alive 槽数=7（排除 slot0）
  k 个已匹配位置：cos_sim=1（完美）→ loss=0
  (7-k) 个未匹配位置：axis_fixed=0, gt_axis=0 → cos(0,0)=0 → loss=1.0

axis_loss = [(7-k)×1.0 + k×0] / 7 = (7-k)/7

k=1: 6/7 = 0.857  ← 出现频率 27% ✓
k=2: 5/7 = 0.714  ← 出现频率 11% ✓
k=3: 4/7 = 0.571  ← 出现频率  4% ✓
```

**实际含义**：轴向预测质量远好于表观指标。`axis_loss=0.857` 实际意味着那个唯一的真实关节**已经完美预测**（|cos|=1）。

**影响**：
- 指标虚高：报告的 axis_loss 主要反映"有多少空 GT 槽"而非"轴向预测有多准"
- 梯度噪声：对 `[0,0,0]` 轴做余弦梯度（值为 0，但计算资源浪费）
- 不影响 scalar/pivot：它们对零填充的梯度也是 0（MSE 的零向量梯度为零）

**修复**：在 `reorder_by_match` 后，对未匹配 GT 位置显式置为"死槽"：

```python
# 记录哪些 GT 位置被匹配到
is_matched = torch.zeros(B, P_gt, dtype=torch.bool, device=device)
for b, (pred_idx, gt_idx) in enumerate(matches):
    is_matched[b, gt_idx] = True

# 未匹配位置强制标记为死（不参与损失）
is_dead_r = is_dead_r | (~is_matched)
```

---

### Bug 3：Plücker 射线方向与数据集约定相反 ★

**位置**：[dggt/utils/plucker.py:61](../dggt/utils/plucker.py#L61)

**现象**：
```
相机位置：[-2, -2, 3]（世界坐标）
相机看向：+[0.485, 0.485, -0.727]（指向世界原点）
d_world = R @ d_cam = R[:,2] = [-0.485, -0.485, 0.727]  ← 反向！
```

**根因**：Blender 导出的 cam-to-world 矩阵的第 3 列（R[:,2]）是相机 **后方**向量（camera backward），而不是前方（camera forward）。代码直接用 `d_world = R @ K_inv @ [u,v,1]` 得到的是反向射线。

**影响**：
- Plücker 射线方向翻转，但矩向量 `m = o × d` 也同步翻转
- 相对几何关系在不同相机之间仍然一致（旋转等效），Transformer 可以部分补偿
- 这解释了为什么 v12 在没有正确 Plücker 的情况下 IoU 仍能达到 0.816（attention 学到了近似的几何关系）

**修复**：
```python
# plucker.py: 在世界坐标系中反转射线方向（OpenGL → OpenCV）
d_cam = torch.einsum("bsij,bsnj->bsni", K_inv, pixels)
d_cam = -d_cam   # ← 取反，使射线指向相机前方（-z 方向）
d_world = torch.einsum("bsij,bsnj->bsni", R, d_cam)
```

---

### Bug 4：sparsity_loss 权重 0.1 在 render_loss 存在时过强 ★

**位置**：[train_art.py:494](../train_art.py#L494)

**现象**：`sparsity_loss ≈ 0.8`，始终高于 mask_loss（应压低 sparsity 使动态槽不消亡）。

**根因**：
```
sparsity_loss = 0.1 × mean(dynamic_slot_mass)
dynamic_slot_mass ≈ 8-9 patches（37×37=1369 总 patch）
→ sparsity_loss ≈ 0.1 × 8.5 ≈ 0.85
```
该损失持续推动动态槽质量归零。当 render_loss 同时反传错误梯度，模型找到鞍点：槽 0 吸收所有质量（满足 sparsity），同时 CE 损失对 log(p≈1)=0 满意。Argmax 下 IoU=0。

**修复**：在修复 Bug1 之后，将 sparsity 权重降低到 0.01-0.02，仅保留温和的正则效果。

---

## 四、优化空间

### 4.1 per_part_alpha_render_loss 性能瓶颈

**位置**：[train_art.py:325-413](../train_art.py#L325)

当前实现是 `O(B × P × S)` 的三重 Python for 循环，每次内层还有 `B` 次 `apply_rigid_transform` 调用：

```python
for t in range(S):          # S=2
    for p in range(P):      # P=8
        for b in range(B):  # B=1
            apply_rigid_transform(...)   # Python 函数调用
```

这是整个训练最慢的部分。可以用向量化改写：
- 将 `axis/pivot/scalar` 展开到 `[B, P, S, 3/3/1]`
- 用矩阵运算一次完成所有 `(b, p, t)` 的刚体变换
- 用广播替换 kernel 计算的内层循环

### 4.2 Hungarian 匹配成本矩阵计算瓶颈

**位置**：[dggt/utils/hungarian_matching.py:51-56](../dggt/utils/hungarian_matching.py#L51)

当前是双重 for 循环（P × P_gt = 8 × 8 = 64 次迭代/样本）：

```python
for i in range(P):
    for j in range(P_gt):
        inter = (pred_bin[i] * gt_masks[j]).sum()
```

可以完全向量化：
```python
# pred_bin: [P, H, W] → expand → [P, 1, H, W]
# gt_masks: [P_gt, H, W] → expand → [1, P_gt, H, W]
inter = (pred_bin.unsqueeze(1) * gt_masks.unsqueeze(0)).sum(dim=[-1,-2])  # [P, P_gt]
union = (pred_bin.unsqueeze(1) + gt_masks.unsqueeze(0)).clamp(0,1).sum(dim=[-1,-2])
cost_mat = 1.0 - inter / (union + 1e-6)
```

### 4.3 axis_loss 应屏蔽近零运动关节

**位置**：[train_art.py:171-177](../train_art.py#L171)

当关节几乎不动时（`gt_scalars.abs().mean() < threshold`），轴向量在物理上无意义（任意方向都等效）。但当前代码会产生梯度噪声：

```python
# 建议：对近零运动关节屏蔽轴向损失
motion_magnitude = gt_scalars[alive].abs().mean(dim=-1)   # [K]
axis_weight = (motion_magnitude > 0.05).float()           # [K]
losses["axis"] = weights["axis"] * ((1.0 - cos_sim) * axis_weight).sum() / axis_weight.sum().clamp(1)
```

这直接修复 v12 中 scene 47632 的 axis_error=90° 问题（近静止关节）。

### 4.4 sparsity 应改为对数形式（渐进惩罚）

当前线性 L1：`loss = w × mass`（uniform pressure）

建议改为对数形式，对完全死亡的槽没有惩罚（已经足够稀疏），对高质量槽增加压力：
```python
# 温和的熵正则化：鼓励二值化但不强制槽消亡
entropy = -(assign_maps * torch.log(assign_maps + 1e-8)).sum(dim=1).mean()  # [B]
sparsity_loss = w * entropy.mean()
```

### 4.5 渲染损失的 sigma 过大导致 alpha 图弥散

`sigma_patches=0.8` 时，256 个高斯覆盖约 1024/1369 ≈ **75% 的 patch**（理论值），无法形成清晰的零件边界。建议：
- 降低到 `sigma_patches=0.3-0.5`
- 或改用自适应 sigma：`sigma = scale_max / depth`（近处小、远处大）

---

## 五、修复优先级总结

| 优先级 | Bug | 影响 | 修复位置 |
|--------|-----|------|---------|
| **P0** | 深度符号错误（OpenGL vs OpenCV） | bbox_loss=0, render 翻转, IoU 崩溃 | `train_art.py:260,362` |
| **P1** | alive 掩码污染（unmatched GT alive） | axis_loss 量化虚高，指标失真 | `train_art.py:148-153` |
| **P2** | sparsity 权重过强 | assign_maps 崩溃 | `train_art.py:494`（weight 0.01） |
| **P3** | Plücker 射线方向反转 | 几何先验不准确 | `plucker.py:61` |
| **P4** | 近零运动轴向损失 | 噪声梯度 | `train_art.py:171` |
| **Opt** | render 循环向量化 | 训练速度 2-4x | `train_art.py:325` |
| **Opt** | Hungarian 向量化 | 每步节省 ~10ms | `hungarian_matching.py:51` |
| **Opt** | sigma 过大 | render 无法形成边界 | `train_art.py:289` |

---

## 六、下一步建议

修复 P0 和 P1 后，**从 v12 的 step 36000 checkpoint 重新启动训练**，配置：

```bash
--resume /data2/cyt/checkpoints/art_v12_phase1b/ckpt_036000.pth
--l1_sparsity 0.02           # 大幅降低（P2）
--w_render 0.3               # 保持，但需先修复 P0
--w_axis 1.0
--w_scalar 0.5
--w_pivot 0.3
```

修复 P0 后预期：
- bbox_loss 恢复正常（0.05-0.2 量级）
- render_loss 开始下降（从 2.0 降至 0.5 以下）
- assign_maps 不再崩溃（IoU 维持在 0.8+）
- axis_loss 真实值暴露（预期 < 0.2）

---

*分析日期：2026-03-30 | v13 step 50000 | 50 val scenes*
