# ArtVGGT 训练代码 Bug 分析报告

## 项目架构概览

```
ArtVGGT (FAST-4D)
├── Aggregator (VGGT encoder, DINOv2-based)  ← 不变
├── CameraHead (位姿预测)                      ← 可选
├── PartSlotRouter (跨注意力解码器)             ← 新增
│     P=8 learnable slot tokens 作为 Query
│     Aggregator patch tokens + Plücker rays + 时间戳 作为 K/V
├── KinematicHead (运动类型/轴/旋转中心)        ← 新增
├── DynamicsHead (per-slot per-frame 运动标量)  ← 新增
└── ArtGaussianHead (canonical 3DGS 解码)      ← 新增

训练分三阶段：
  Phase 1a: warmup，只优化 mask 损失（KinematicHead/DynamicsHead/GaussianHead 冻结）
  Phase 1b: 全监督，所有 head 解冻
  Phase 2:  弱监督域适应，使用真实数据 + SAM2 伪掩码
```

数据流：`[B,S,3,H,W]` → Aggregator → PartSlotRouter → `assign_maps[B,P,H_p,W_p]` + `slot_features[B,P,D]` → KinematicHead / DynamicsHead / GaussianHead

---

## Bug 清单

### [严重 - BUG-1] 恢复训练时 LR Scheduler 被双重推进

**位置：** [train_art.py:450-454](train_art.py#L450-L454)

```python
if not cfg.reset_scheduler and "scheduler" in _resume_ckpt:
    scheduler.load_state_dict(_resume_ckpt["scheduler"])
    # Advance scheduler to start_step without stepping optimizer
    for _ in range(start_step):
        scheduler.step()   # ← BUG
```

**问题：** `load_state_dict` 已经将 scheduler 的 `last_epoch` 恢复到 `start_step`，之后又循环调用 `step()` 共 `start_step` 次，导致 scheduler 被推进到 `2 × start_step`，学习率远低于预期值。

**后果：** 从 checkpoint 恢复后 LR 极低，训练几乎停滞，且很难调试（损失表面上在更新）。

**修复方向：** 二选一：
- `load_state_dict` 后删除循环（state 已经正确了）；
- 或者：不 `load_state_dict`，只在新 scheduler 上 `for _ in range(start_step): scheduler.step()`。

---

### [严重 - BUG-2] Phase 2 完全不使用 `real_data_root`

**位置：** [train_art.py:399-412](train_art.py#L399-L412)

```python
train_dataset = ArticulatedDataset(
    data_root = cfg.data_root,          # ← Phase 2 应该用 real_data_root
    ...
    phase = "1" if cfg.phase in ("1a", "1b") else "2",
)
```

**问题：** `parse_args` 定义了 `--real_data_root` 参数，文档也说明 Phase 2 使用真实数据做域适应，但 `cfg.real_data_root` 在训练脚本中从未被读取或使用。Phase 2 仍然在合成数据上训练。

**后果：** Phase 2 的域适应完全失效，等同于在重复训练 Phase 1b 的数据。

---

### [严重 - BUG-3] 训练集和验证集使用完全相同的数据

**位置：** [train_art.py:399-428](train_art.py#L399-L428)

```python
train_dataset = ArticulatedDataset(data_root=cfg.data_root, ...)
val_dataset   = ArticulatedDataset(data_root=cfg.data_root, ...)  # 完全相同
```

**问题：** 两个 Dataset 对象都指向同一 `data_root`，没有 train/val 划分逻辑（如 scene_list 文件或 split 参数）。

**后果：** Warmup 的门控逻辑依赖 `eval_mean_iou`，但它实际上是在度量训练集上的 IoU，不能反映泛化能力，导致 warmup 过早结束（即使模型根本没有泛化能力）。

---

### [严重 - BUG-4] Phase 2 伪掩码损失用的是 GT mask 而非 SAM2 伪掩码

**位置：** [train_art.py:295-297](train_art.py#L295-L297)

```python
if cfg.phase == "2" and batch.get("has_pseudo_masks", False):
    l_pseudo = cfg.w_pseudo_mask * mask_loss(assign_maps, gt_masks, matches)
    # ↑ 用的是 gt_masks，不是 pseudo_masks
```

**问题：** 代码中 `gt_masks` 来自 `batch["part_masks"]`（GT ground-truth），而不是 SAM2 生成的伪掩码。注释与实现不符，且 `batch` 中也没有 `pseudo_masks` 字段。

**后果：** `w_pseudo_mask=0.05` 实际上只是在 GT mask loss 上再加了一个额外权重，而非弱监督信号。

---

### [中等 - BUG-5] `eval_mean_iou` 异常时模型保持 eval 模式

**位置：** [train_art.py:314-347](train_art.py#L314-L347)

```python
def eval_mean_iou(model, val_loader, device, cfg) -> float:
    model.eval()
    ...
    for batch in val_loader:
        ...   # 如果这里抛异常
    model.train()  # ← 不会被执行
    return ...
```

**问题：** 若数据加载或前向传播抛出异常，`model.train()` 不会被执行，后续训练步骤将在 eval 模式下进行（BatchNorm/Dropout 行为错误）。

**修复方向：** 用 `try...finally` 包裹，或使用上下文管理器。

---

### [中等 - BUG-6] `PartSlotRouter` 假设图像为正方形

**位置：** [dggt/heads/part_slot_router.py:151-155](dggt/heads/part_slot_router.py#L151-L155)

```python
H_p = W_p = int(math.isqrt(N_patches))
assert H_p * W_p == N_patches, (...)
```

**问题：** 代码强制假设 patch token 数量是完全平方数（即输入图像为正方形）。对于 `img_size=518`，`518/14=37`，`37²=1369`，确实满足。但若使用非方形分辨率（如 `512×384`），`N_patches=37*27=999`，`isqrt(999)=31`，`31²≠999`，断言失败并崩溃，而不是给出有意义的错误信息。

**修复方向：** 分别计算 `H_p = H // patch_size`，`W_p = W // patch_size`，不用 `isqrt`。

---

### [中等 - BUG-7] 分布式训练中 `DistributedSampler.set_epoch` 传入的是 step 而非 epoch

**位置：** [train_art.py:479](train_art.py#L479)

```python
if is_dist:
    train_sampler.set_epoch(step)  # step 不是 epoch
```

**问题：** `set_epoch` 的设计用意是将 epoch 编号作为随机种子，保证各 rank 的 shuffle 一致性。传入 step 数值（通常是数百到数万）会导致每次重新加载数据迭代器时用不同的高随机性种子，功能上虽然可用，但与 PyTorch 官方约定不符，且在跨 epoch 边界处无法复现确定性排序。

---

### [中等 - BUG-8] Warmup 后重建 optimizer 会丢失 GradScaler 状态

**位置：** [train_art.py:530-534](train_art.py#L530-L534)

```python
if warmup_iou_checks_passed >= 3:
    ...
    optimizer = torch.optim.AdamW(...)
    scaler = GradScaler("cuda")   # ← scale 因子从 65536 重置
```

**问题：** Warmup 完成后，重新创建了 `GradScaler`，其 `loss_scale` 从初始值重新开始增长。此时 KinematicHead 等刚解冻的 head 梯度可能较大，而 scale 从 65536 起步可能导致短暂的梯度 overflow 和 scaler underflow，表现为初始几步 loss 为 NaN 或 step 被跳过。

---

### [轻微 - BUG-9] `ArticulatedDataset._orig_w/_orig_h` 是类变量，多实例间共享

**位置：** [datasets/articulated_dataset.py:100-101, 143-145](datasets/articulated_dataset.py#L100-L145)

```python
class ArticulatedDataset(Dataset):
    _orig_w: int = 640   # ← 类变量
    _orig_h: int = 480

    def __init__(self, ...):
        ...
        ArticulatedDataset._orig_w = w   # ← 修改类变量
        ArticulatedDataset._orig_h = h
```

**问题：** `train_dataset` 和 `val_dataset` 共享同一类变量。若两个 dataset 目录中图像分辨率不同，后初始化的 dataset 会覆盖前者设定的值，影响 `_adjust_intrinsics` 的结果。推荐改为实例变量 `self._orig_w`。

---

### [轻微 - BUG-10] `ArticulatedDataset` 未使用 mask ID=1 的掩码

**位置：** [datasets/articulated_dataset.py:244-260](datasets/articulated_dataset.py#L244-L260)

```python
# joint 0 → part_id=2, joint 1 → part_id=3, ..., root → part_id=n_joints+2
for k in range(n_joints):
    part_id = k + 2   # 从 2 开始，跳过了 ID=1
```

**问题：** mask ID=1 从未被读取。若数据集中 `{fid}_1.png` 存在（例如代表某个特定部件），该部件信息完全丢失。需确认数据集规范中 ID=1 的语义。

---

## 问题严重性汇总

| 编号 | 严重性 | 模块 | 摘要 |
|------|--------|------|------|
| BUG-1 | 🔴 严重 | train_art.py | 恢复时 scheduler 被双重推进，LR 错误 |
| BUG-2 | 🔴 严重 | train_art.py | Phase 2 不使用 real_data_root |
| BUG-3 | 🔴 严重 | train_art.py | 训练/验证集未分割，验证无效 |
| BUG-4 | 🔴 严重 | train_art.py | Phase 2 伪掩码损失实为 GT mask 损失 |
| BUG-5 | 🟡 中等 | train_art.py | eval_mean_iou 异常后模型卡在 eval 模式 |
| BUG-6 | 🟡 中等 | part_slot_router.py | 非方形图像时断言失败 |
| BUG-7 | 🟡 中等 | train_art.py | DDP sampler 用 step 代替 epoch |
| BUG-8 | 🟡 中等 | train_art.py | Warmup 后 GradScaler 重置可能导致 NaN |
| BUG-9 | 🟢 轻微 | articulated_dataset.py | _orig_w/h 类变量多实例共享 |
| BUG-10 | 🟢 轻微 | articulated_dataset.py | mask ID=1 未被读取 |
