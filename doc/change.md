# Change Log


## 2026-04-21 Phase C 点轨迹 (TrackEncoder) 前向注入骨架

### 动机
Phase B 的 motion pseudo-label 仅作为监督信号,模型前向中没有帧间 point correspondence,难以利用 CoWTracker/GT 得到的运动轨迹。Phase C 将 2D tracks 作为 **前向输入**:每条 track 编码为一个 token,与 image / slot token 一起进 PartSlotRouter,让 slot 在 attention 中直接"看"到帧间对应关系。

### 设计要点
- **轨迹描述子**(无参数):`[xy | dxy | ddxy | similarity_residual | stats]`,维度 `8S+3`。`similarity_residual` 通过逐帧加权 Umeyama 拟合 frame0→frame_t 的 2D similarity 并减去,相当于去除相机/整体刚性运动,保留**部件相对运动**(天然兼容静止/运动相机)。
- **PSR cross-attn 方向不变**(Q=image, KV=slot),保证 assign_map 语义不变。
- **注入方式**:①每层 self-attn 扩展为 3 流 `[image, slot, track]`;②在最后一层 cross-attn 之前插入一次 `SlotTrackCrossAttn`(Q=slot, KV=track),`out_proj` 零初始化 → step 0 等价于无 track 模型(安全 bootstrap)。
- **track stream type embedding**:可学习的 `track_type_embed [1,1,D]` 加到 track token 上,让 self-attn 区分流。
- **padding mask**:vis 全 0 的 track 视为 padding,在 3 流 self-attn 和 slot-track cross-attn 中屏蔽。
- **参数组**:backbone lr=`--lr`(默认 1e-4),track 相关参数 lr=`--track_encoder_lr`(默认 3e-4)。
- **Warmup 策略**:`--freeze_track_warmup_steps`(默认 5000)步内冻结 backbone,仅训 track 相关参数;到达边界后重建 optimizer 解冻 backbone(非每步,而是边界一次)。

### 新增文件
- `dggt/heads/track_encoder.py`
  - `compute_dynamics_descriptor(tracks, vis, img_size) → [B, N, 8S+3]`:归一化 xy、一阶/二阶差分、加权 Umeyama 残差、vis/length/res_std 统计。
  - `_weighted_similarity_fit(src, dst, w)`:闭式加权 2D similarity(含反射校正),返回 `(A[B,2,2], t[B,2])`,SVD 失败回退到恒等。
  - `TrackEncoder(num_frames, img_size, dim_inner=512, dim_out=1024)`:`proj_in(F_in→512) → LN → residual MLP → proj_out(512→1024)`,前向 `(tracks[B,S,N,2], vis[B,S,N]) → [B,N,1024]`。

### `dggt/heads/part_slot_router.py`
- `SelfAttentionLayer` 加 `use_track_tokens: bool` ctor 参数;当 True 时额外创建 `ffn_track = _FFN(dim)`;forward 签名改为 `(image_tokens, slot_tokens, track_tokens=None, track_mask=None)`,返回 `(img_out, slot_out, track_out|None)`。当 `track_tokens is None` 时行为与原 2 流版本完全一致(backward-compatible)。attn_mask 对 padding track key 置 -inf。
- 新增 `SlotTrackCrossAttn(dim, num_heads)`:Q=slots(P 个查询)/ KV=tracks(T 个键),带 padding mask;`out_proj.weight` 零初始化;输出残差加到 slots。
- `PartSlotRouter.__init__` 新增 `use_track_tokens: bool=False`、`dim_track: int=1024`:
  - `SelfAttentionLayer` 按 flag 传参。
  - `track_in_proj = Linear(dim_track, dim_slot)` 或 `Identity`。
  - `track_type_embed = nn.Parameter(zeros(1,1,dim_slot))`。
  - `slot_track_fuse = SlotTrackCrossAttn(...)`。
  - `_init_weights()` 执行后,对 `slot_track_fuse.out_proj.weight` 重新置零(防止 xavier 覆盖零初始化)。
- `PartSlotRouter.forward` 新增 `track_tokens / track_mask` 参数:
  - 若启用:`tr = track_in_proj(track_tokens) + track_type_embed`。
  - 计算 `last_cross_idx`,在最后一层 cross-attn 之前调用 `slot_track_fuse(slots, tr, track_mask)`。
  - self-attn 层按 3 流调用 `(image_tokens, slots, tr, track_mask)`。
  - cross-attn 层调用方式不变,assign_map 抽取逻辑不变。

### `dggt/models/art_vggt.py`
- ctor 新增 `use_track_tokens: bool=False`、`num_frames_track: int=8`。
- `PartSlotRouter` 以 `use_track_tokens / dim_track=embed_dim` 实例化。
- 启用时额外创建 `self.track_encoder = TrackEncoder(num_frames=num_frames_track, img_size=img_size, dim_inner=512, dim_out=embed_dim)`。
- `forward` 签名新增 `tracks_2d=None, tracks_vis=None`;启用时计算 `track_tokens = track_encoder(tracks_2d, tracks_vis)`、`track_mask = (tracks_vis.sum(1) > 0).float()`,写入 `preds["track_mask_valid_frac"]` 供 diagnostics;PSR 调用透传。
- `set_phase` 的 phase==1a/1b 解冻循环加入 `self.track_encoder`。

### `datasets/articulated_dataset.py`
- motion_cache 加载分支(465-481 行附近):当 `N_raw ≥ N_t` 时**随机采样**(`np.random.permutation(N_raw)[:N_t]`),对 `tracks_2d / tracks_3d / tracks_vis / track_part_label` 使用**同一组索引**保持对齐;`N_raw < N_t` 时沿用原零填充路径。实现每迭代随机 subsample 的数据增强。

### `train_art.py`
- CLI 新增:
  - `--use_track_tokens`(flag)
  - `--track_encoder_lr`(默认 3e-4)
  - `--freeze_track_warmup_steps`(默认 5000)
  - `--max_tracks_per_sample`(默认 1024 → 传递给 `ArticulatedDataset.max_tracks`)
- `ArtVGGT` 实例化传 `use_track_tokens=cfg.use_track_tokens, num_frames_track=cfg.num_frames`。
- 新增辅助:
  - `_is_track_param(name)`:识别 `track_encoder.*`、`part_slot_router.track_in_proj / track_type_embed / slot_track_fuse`、`ffn_track.*`。
  - `_build_param_groups(m)`:返回两组 `[{backbone, lr=cfg.lr}, {track, lr=cfg.track_encoder_lr}]`(track 组为空时省略)。
- Optimizer 初始化改为 `AdamW(_build_param_groups(raw_model), ...)`。warmup 通过 gate 触发的 optimizer 重建同样走 `_build_param_groups`。
- **Track-warmup 冻结**:`cfg.use_track_tokens and start_step < cfg.freeze_track_warmup_steps` 时,先把所有非 track 参数 `requires_grad_(False)` 再构建 optimizer。
- **边界解冻**:训练循环 `step += 1` 之后,当 `step == cfg.freeze_track_warmup_steps` 时调用 `raw_model.set_phase(...)` 恢复阶段默认可训参数集,并重建 optimizer(非每步检查,而是一次性边界触发)。
- 训练/验证 forward 均改为 `model(..., tracks_2d=..., tracks_vis=...)`,仅在 `cfg.use_track_tokens=True` 时取 batch 中的 tracks。

### 安全 bootstrap 性质
- `slot_track_fuse.out_proj = 0` → 该层是 identity。
- `track_type_embed = 0` 初始化 → 初始 track token 方向无偏。
- `ffn_track` 中 `_FFN` 本身是 residual 结构,参数 xavier init 下输出较小。
- 新加的 3 流 self-attn key/value 若 backbone 在 warmup 内冻结,对 image/slot 的更新仅通过 `slot_track_fuse` 的零残差 + 新的 FFN 分支,初期对原轨迹几乎无扰动。

### 未完成 / 延后
- PSR 未导出 `track_attn_weight_mean / track_residual_norm_mean` diagnostics(需在 `slot_track_fuse` 内保存 attn weight 与 residual norm 并通过返回值传出)。
- `launch_phase_c.sh` 未创建(用户自行)。
- `launch_phase1a.sh` 的 nproc / port bug 用户自行修。
- RGB 采样分支、`L_track_consistency / L_track_rigidity`、CoWTracker 编码器解冻、PSR self-attn 去冗余消融,均留到 Phase C 验证后再做。


## 2026-04-23 训练管线 Bug 修复(Phase C 上线前)

针对整套训练流程的复审,修复 4 项真实问题。对应评审记为 #1 / #2 / #4 / #9。

### #1 `dist.broadcast` 形状不匹配(NCCL 崩溃/挂起)
**位置**:`train_art.py` 中两处 val IoU 同步(Phase 1b 日志块 + warmup gate 块)。

**问题**:rank 0 用 `torch.tensor(mean_iou, device=device)`(0-d,shape `[]`)而非 rank 用 `torch.zeros(1, device=device)`(1-d,shape `[1]`)。`dist.broadcast` 要求所有 rank tensor 形状完全一致,否则 NCCL 报错或静默挂起。

**修复**:统一为 0-d:
```python
iou_tensor = torch.zeros((), device=device)
if is_main:
    mean_iou = eval_mean_iou(...); iou_tensor.fill_(mean_iou)
if is_dist:
    dist.broadcast(iou_tensor, src=0)
    dist.barrier()                 # 见 #4
```

### #2 `reset_slot_tokens` 访问不存在的属性
**位置**:`train_art.py:1158`。

**问题**:代码 `model.part_slot_router.slot_tokens` 但该 `nn.Parameter` 已迁移到 `aggregator.slot_tokens`(见 `dggt/models/aggregator.py:142`)。传 `--reset_slot_tokens` 会直接 `AttributeError`。

**修复**:改为 `model.aggregator.slot_tokens`。

### #4 Warmup eval 只在 rank 0 跑,后续 DDP 状态易漂移
**问题**:`eval_mean_iou` 仅 rank 0 执行(避免多卡同时 val 导致 OOM),ranks 1..N 在 `dist.broadcast` 处阻塞等待。eval 结束 rank 0 可能会 `set_phase(...)` 切换 `requires_grad_`、重建 optimizer;其它 rank 看到的是 broadcast 返回后立即进入下一个 step。中间若有任何 rank 时序偏差都可能让 DDP reducer 状态不一致。

**修复**:在 `broadcast` 后追加 `dist.barrier()`,强制所有 rank 在 `set_phase / 重建 optimizer` 完成后再进下一步前显式同步。两个 val 块(Phase 1b 日志 + warmup gate)均已加。

> 备注:DDP 本身对 "运行时切换 requires_grad" 并不完全安全(reducer bucket 在构造期固化)。本次保留 `find_unused_parameters=True` 让绝大多数 case 可工作;彻底修复需在 `set_phase` 后 `del model; model = DDP(raw_model, ...)` 重新包装,列入后续 TODO。

### #9 `SlotTrackCrossAttn` zero-init `out_proj.weight` 会冻结整条 TrackEncoder 梯度
**位置**:`dggt/heads/part_slot_router.py` `SlotTrackCrossAttn`。

**问题**:为了 "step 0 等价于无 track 模型" 的 safe bootstrap,原先 `nn.init.zeros_(out_proj.weight)`。但链式法则下,`d(slots + out_proj(attn)) / d(attn) = out_proj.weight^T = 0`,因此 track_encoder、q/k/v_proj 在 track-warmup 的前 `freeze_track_warmup_steps` 步内收到 **精确为 0** 的梯度 —— 这些参数完全不被训练,track-warmup 实际变成空转。

**修复**:把 safe-bootstrap 的担子从 `out_proj.weight=0` 换到 **LayerScale 门控** `gamma`:
```python
self.out_proj = nn.Linear(dim, dim, bias=False)   # 正常 xavier init
self.gamma    = nn.Parameter(torch.zeros(dim))    # 零初始 LayerScale
...
return slots + self.gamma * self.out_proj(out)
```
- 初值 `gamma=0` → step 0 输出仍为 `slots`(near-identity 保留)。
- 但 `d(gamma * out_proj(attn)) / d(gamma) = out_proj(attn) ≠ 0`,`gamma` 自身收到非零梯度;`gamma` 一旦离开 0,上游 `out_proj / k_proj / v_proj / track_encoder` 立即得到非零梯度。
- 同步更新 `PartSlotRouter.__init__`:`_init_weights()` 后对 `slot_track_fuse.gamma` 显式置零(原本对 `out_proj.weight` 置零那行)。

### 影响总结
- #1 / #2 必修:不修会在跑任一带 DDP 的 val 或传 `--reset_slot_tokens` 时直接崩。
- #4 是稳定性补丁,不修偶发会出 silent 梯度不同步。
- #9 是 Phase C 语义正确性补丁:不修则 `--freeze_track_warmup_steps` 形同虚设,TrackEncoder 原地不动直到 warmup 边界才开始学。

### 遗留(延后)
- DDP 在 `set_phase` 后需重新 `DDP(...)` 包装(见 #4 备注)。
- `TrackEncoder.compute_dynamics_descriptor` 对 similarity fit 退化情况(`var_s` 极小但 >eps)未做 `s` clamp;bf16 下可能产生大 residual。出现 NaN 再加。
- `GradScaler` 在 bf16 autocast 下未启用,属无害死代码,后续清理。

