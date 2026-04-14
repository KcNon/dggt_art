# Change Log

## 2026-04-14 Phase 1b 逐帧 IoU 评估 + 训练监控

### 动机
原先 Phase 1b 训练时 `eval_mean_iou` 因为 `warmup_done=True` 的分支不进入而从未被调用,无法观测真实 mask 质量;eval_phase1b.py 的 GT 聚合方式又与训练端不一致,导致 52k/70k 评估 IoU 只有 ~0.53,与训练端 Phase 1a 的 0.69~0.70 出现巨大落差。同时 Phase 1a 仅对 frame 0 做 mask 评估,无法反映运动帧上的 slot 对齐情况,决定改为逐帧评估。

### train_art.py
- `eval_mean_iou` 整体重写(约 792-838 行):
  - 外层加 `torch.no_grad()`。
  - 按 `cfg.phase` 分支:Phase 1a 仍走 `assign_maps` frame0 路径,但加入 `if is_dead[b, pi]: continue` 和 `if gt_m.sum() < 1: continue` 过滤;Phase 1b 走新的逐帧 GS 投影路径。
  - Phase 1b 分支:对每帧 t,使用外参构造 `R_w2c / t_w2c`,对每个 slot 调 `apply_rigid_transform(mu_p, rot_p, mp3, axis, pivot, scalars[:,:,t])` 把 canonical GS 变换到该帧,再投影到像素平面、按高斯核渲成 alpha 图、上采样到 H,W,和 GT binary mask 算 IoU。
  - 跳过 dead slot 和空 GT mask,避免污染均值。
- 训练循环新增 Phase 1b IoU 日志块(约 1067 行,warmup gate 之前):
  ```python
  if warmup_done and cfg.phase != "1a" and (step % cfg.val_interval == 0):
      if is_main:
          mean_iou = eval_mean_iou(raw_model, val_loader, device, cfg)
          print(f"[step {step}] per-frame val IoU = {mean_iou:.4f}", flush=True)
          ...
      if is_dist:
          dist.broadcast(iou_tensor, src=0)
  ```
  保证 Phase 1b 每 `val_interval` 打一次逐帧 IoU,并广播到所有 rank。

### eval_phase1b.py
- 149 行 GT mask 聚合修正:
  - 旧:`gt_masks_bin = (part_masks.max(dim=0).values > 0.5).float()` (对时间维 max,得到运动轨迹并集,导致 IoU 严重偏低)
  - 新:`gt_masks_bin = (part_masks[0] > 0.5).float()` (只取 frame 0,与训练端 Phase 1a 对齐)



