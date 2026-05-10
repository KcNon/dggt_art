# Change Log


## 2026-04-21 Phase C 点轨迹 (TrackEncoder) 前向注入骨架

### 动机
Phase B 的 motion pseudo-label 仅作为监督信号,模型前向中没有帧间 point correspondence,难以利用 CoWTracker/GT 得到的运动轨迹。Phase C 将 2D tracks 作为 **前向输入**:每条 track 编码为一个 token,与 image / slot token 一起进 PartSlotRouter,让 slot 在 attention 中直接 "看" 到帧间对应关系。

### 设计要点
- **轨迹描述子**(无参数):`[xy | dxy | ddxy | similarity_residual | stats]`,维度 `8S+3`。`similarity_residual` 通过逐帧加权 Umeyama 拟合 frame0→frame_t 的 2D similarity 并减去,相当于去除相机/整体刚性运动,保留**部件相对运动**(天然兼容静止/运动相机)。
- **PSR cross-attn 方向不变**(Q=image, KV=slot),保证 assign_map 语义不变。
- **注入方式**:① 每层 self-attn 扩展为 3 流 `[image, slot, track]`;② 在最后一层 cross-attn 之前插入一次 `SlotTrackCrossAttn`(Q=slot, KV=track),通过 LayerScale `gamma`(初值 0)做 safe bootstrap。
- **track stream type embedding**:可学习的 `track_type_embed [1,1,D]` 加到 track token 上,让 self-attn 区分流。
- **padding mask**:vis 全 0 的 track 视为 padding,在 3 流 self-attn 和 slot-track cross-attn 中屏蔽。
- **参数组**:backbone lr=`--lr`,track 相关参数 lr=`--track_encoder_lr`(默认 3e-4)。
- **Warmup 策略**:`--freeze_track_warmup_steps` 内冻结 backbone,仅训 track 相关参数;到达边界后重建 optimizer 解冻 backbone(非每步,而是边界一次)。

### 新增文件
- `dggt/heads/track_encoder.py`
  - `_weighted_similarity_fit(src, dst, w)`:闭式加权 2D similarity(含反射校正),返回 `(A[B,2,2], t[B,2])`,SVD 失败回退到恒等。
  - `compute_dynamics_descriptor(tracks, vis, img_size, freq_bands=None) → [B, N, F]`:归一化 xy(可选 Fourier lift)、一阶/二阶差分、加权 Umeyama 残差、vis/length/res_std 统计。
  - `_fourier_encode(xy, freq_bands)`:sin/cos × x/y × K bands → 4K 维。
  - `TrackEncoder(num_frames, img_size, dim_inner=512, dim_out=1024, num_freq_bands=6, dim_vis_feat=0)`:`proj_in → LN → residual MLP → proj_out`,前向 `(tracks, vis, vis_feat=None) → [B, N, dim_out]`。

### `dggt/heads/part_slot_router.py`
- `SelfAttentionLayer` 加 `use_track_tokens: bool` ctor 参数;当 True 时额外创建 `ffn_track = _FFN(dim)`;forward 签名改为 `(image_tokens, slot_tokens, track_tokens=None, track_mask=None)`,返回 `(img_out, slot_out, track_out|None)`。当 `track_tokens is None` 时行为与原 2 流版本完全一致(backward-compatible)。attn_mask 对 padding track key 置 -inf。
- 新增 `SlotTrackCrossAttn(dim, num_heads)`:Q=slots(P 个查询)/ KV=tracks(T 个键),带 padding mask;输出 `slots + γ ⊙ out_proj(attn)`,`gamma` 初值 0(LayerScale)。
- `PartSlotRouter.__init__` 新增 `use_track_tokens=False, dim_track=1024`:
  - `SelfAttentionLayer` 按 flag 传参。
  - `track_in_proj = Linear(dim_track, dim_slot)` 或 `Identity`。
  - `track_type_embed = nn.Parameter(zeros(1,1,dim_slot))`。
  - `slot_track_fuse = SlotTrackCrossAttn(...)`。
  - `_init_weights()` 执行后,对 `slot_track_fuse.gamma` 重新置零(防止 xavier 覆盖零初始化)。
- `PartSlotRouter.forward` 新增 `track_tokens / track_mask` 参数:
  - 若启用:`tr = track_in_proj(track_tokens) + track_type_embed`。
  - 计算 `last_cross_idx`,在最后一层 cross-attn 之前调用 `slot_track_fuse(slots, tr, track_mask)`。
  - self-attn 层按 3 流调用 `(image_tokens, slots, tr, track_mask)`。
  - cross-attn 层调用方式不变,assign_map 抽取逻辑不变。

### `dggt/models/art_vggt.py`
- ctor 新增 `use_track_tokens: bool=False`、`num_frames_track: int=8`。
- `PartSlotRouter` 以 `use_track_tokens / dim_track=embed_dim` 实例化。
- 启用时额外创建 `self.track_encoder = TrackEncoder(num_frames=num_frames_track, img_size=img_size, dim_inner=512, dim_out=embed_dim, num_freq_bands=6, dim_vis_feat=embed_dim)`。
- `forward` 签名新增 `tracks_2d=None, tracks_vis=None, disable_tracks=False`:
  - 启用时:计算 `track_mask = (tracks_vis.sum(1) > 0).float()`;
  - 从 frame-0 DINOv2 patch token reshape 成 `[B, D, H_p, W_p]`,对每条 track 的 frame-0 像素 `F.grid_sample(padding_mode="border")` 得到 `vis_feat [B, N, D]`;
  - `track_tokens = track_encoder(tracks_2d, tracks_vis, vis_feat=vis_feat)`,写入 `preds["track_mask_valid_frac"]`,PSR 调用透传。
- `set_phase` 的 phase==1a/1b 解冻循环加入 `self.track_encoder`。
- `disable_tracks=True` 时跳过整条 track 注入分支(val-time ablation)。

### `datasets/articulated_dataset.py`
- motion_cache 加载分支:当 `N_raw ≥ N_t` 时**随机采样**(`np.random.permutation(N_raw)[:N_t]`),对 `tracks_2d / tracks_3d / tracks_vis / track_part_label` 使用**同一组索引**保持对齐;`N_raw < N_t` 时沿用原零填充路径。

### `train_art.py`
- CLI 新增:
  - `--use_track_tokens`(flag)
  - `--track_encoder_lr`(默认 3e-4)
  - `--freeze_track_warmup_steps`(默认 5000;Phase C resume 场景下设为 0 显式关闭)
  - `--max_tracks_per_sample`(默认 1024 → 传递给 `ArticulatedDataset.max_tracks`)
- `ArtVGGT` 实例化传 `use_track_tokens=cfg.use_track_tokens, num_frames_track=cfg.num_frames`。
- 新增辅助:
  - `_is_track_param(name)`:识别 `track_encoder.*`、`part_slot_router.track_in_proj / track_type_embed / slot_track_fuse`、`ffn_track.*`。
  - `_build_param_groups(m)`:返回两组 `[{backbone, lr=cfg.lr}, {track, lr=cfg.track_encoder_lr}]`(track 组为空时省略)。
- Optimizer 初始化与 warmup gate 触发的重建均改为 `AdamW(_build_param_groups(raw_model), ...)`。
- **Track-warmup 冻结**:`cfg.use_track_tokens and start_step < cfg.freeze_track_warmup_steps` 时,先把所有非 track 参数 `requires_grad_(False)` 再构建 optimizer。
- **边界解冻**:训练循环 `step += 1` 之后,当 `step == cfg.freeze_track_warmup_steps` 时调用 `raw_model.set_phase(...)` 恢复阶段默认可训参数集,并重建 optimizer。
- 训练/验证 forward 均改为 `model(..., tracks_2d=..., tracks_vis=..., disable_tracks=...)`,仅在 `cfg.use_track_tokens=True` 时取 batch 中的 tracks。

### 安全 bootstrap 性质
- `slot_track_fuse.gamma = 0` → 该层是 identity(对 slots 无残差贡献);但 `d/dγ ≠ 0`,梯度可流到上游 `out_proj / k_proj / v_proj / track_encoder`(避免 zero-init out_proj 的死链路)。
- `track_type_embed = 0` 初始化 → 初始 track token 方向无偏。
- `ffn_track` 中 `_FFN` 本身是 residual 结构,参数 xavier init 下输出较小。

### 未完成 / 延后
- PSR 未导出 `track_attn_weight_mean / track_residual_norm_mean` diagnostics(需在 `slot_track_fuse` 内保存 attn weight 与 residual norm 并通过返回值传出)。
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


## 2026-04-23 训练诊断:gamma / te_grad 日志 + val 时 no-track 消融

为了实时判断 Phase C 是否真的起作用,加两个低成本探针。

### 日志增量 — γ 范数 + TrackEncoder 梯度范数
`train_art.py` 训练循环的 `log_interval` 块,在每行末尾追加:
```
gamma=<L2 of slot_track_fuse.gamma>   te_grad=<L2 of track_encoder grads>
```
判据:
- `te_grad` **从第 1 步起非零** → 修复 #9 生效,梯度链路通。
- `gamma` **几百步内脱离 0**(向 0.05 → 0.1 → 0.5 量级走) → 模型在 "采用" tracks。
- `gamma` 持续严格 0 → 数据/链路有死路,需排查。

实现细节:取自 backward 之后、`zero_grad` 之前的 `p.grad`,只在 rank 0 打印。

### val-time ablation:同一 ckpt,关闭 track 注入跑一次
`ArtVGGT.forward` 新增 `disable_tracks: bool = False`:
```python
if self.use_track_tokens and not disable_tracks and self.track_encoder is not None ...
```
`eval_mean_iou(model, val_loader, device, cfg, disable_tracks: bool = False)` 接受同名 kwarg,内部把 `disable_tracks` 透传给 model。

Phase 1b 的 per-frame IoU 日志块改为跑两次:
```python
mean_iou      = eval_mean_iou(...)
iou_no_tracks = eval_mean_iou(..., disable_tracks=True)
gap = mean_iou - iou_no_tracks
print(f"per-frame val IoU = {mean_iou:.4f}  (no-tracks={iou_no_tracks:.4f}  gap={gap:+.4f})")
```
判据:**gap > 0 且随训练扩大** → tracks 真的为下游提供新信息;gap ≈ 0 即使 gamma 大 → tracks 无增益。

成本:val_interval(默认 500)上 rank 0 单次 eval 时间 ×2,可接受。


## 2026-04-23 Phase C 增强:Fourier xy + DINOv2 视觉锚点

参考外部对比方案(per-point Fourier positional encoding + visual feature sampling),把两点能直接吸收的特性加入 TrackEncoder。

### 动机
原始 TrackEncoder 只有几何信号,模型需要从轨迹形状反推 "这条轨迹属于哪个部件"。但 Aggregator/DINOv2 已经在每个像素位置算好了语义特征,**白白浪费**。同时,xy 的线性归一化经 MLP 学高频空间结构效率低,Fourier 是经典解法。

### `_fourier_encode(xy, freq_bands)`
对 `xy ∈ [-1, 1]^2` 做正弦/余弦位置编码:
```
freq_bands = [π, 2π, 4π, ..., 2^(K-1)·π]      # K bands, 默认 K=6
xy_f = xy.unsqueeze(-1) * freq_bands           # [..., 2, K]
out  = concat([sin(xy_f), cos(xy_f)], dim=-1)  # [..., 2, 2K]
return out.flatten(-2)                          # [..., 4K]
```
覆盖空间尺度从全图(λ ≈ 2.0)到 ~16 px(在 518×518 图上)。`dxy / ddxy / residual` 是差分量,Fourier 冗余,**保持线性**。

### `compute_dynamics_descriptor` 接受 `freq_bands` 参数
- `freq_bands is None` → 旧行为(线性 xy,2S 维)
- `freq_bands is not None` → Fourier xy(每帧 4K 维),其余通道不变

### `TrackEncoder` ctor 新增两个开关
```python
TrackEncoder(
    num_frames, img_size,
    dim_inner=512, dim_out=1024, mlp_ratio=4,
    num_freq_bands=6,        # 0 关闭 Fourier
    dim_vis_feat=0,          # >0 时启用视觉锚点输入
)
```
- `register_buffer("freq_bands", ...)`(non-persistent)
- `F_in = 4K·S + 6·S + 3 + dim_vis_feat`
- `proj_in: Linear(F_in, dim_inner)`

forward 新增 `vis_feat: Tensor | None`:启用时 concat 进描述子末尾。

### `art_vggt.py` 视觉锚点采样
启用时构造 `TrackEncoder(num_freq_bands=6, dim_vis_feat=embed_dim)`。

forward 中,在调用 track_encoder 之前:
```python
D_dino = dino_tokens.shape[-1]
H_p = H // self.patch_size; W_p = W // self.patch_size
dino0 = dino_tokens[:, 0, patch_start_idx:, :]                 # [B, N_p, D]
fmap  = dino0.reshape(B, H_p, W_p, D_dino).permute(0, 3, 1, 2)  # [B, D, H_p, W_p]
gx = tracks_2d[:, 0, :, 0] / max(W-1, 1) * 2.0 - 1.0
gy = tracks_2d[:, 0, :, 1] / max(H-1, 1) * 2.0 - 1.0
grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)               # [B, 1, N, 2]
vis_feat = F.grid_sample(
    fmap.float(), grid.float(),
    mode="bilinear", padding_mode="border", align_corners=True,
).squeeze(2).transpose(1, 2)                                    # [B, N, D]
vis_feat = vis_feat.to(dino_tokens.dtype)
```
- 只在 frame 0(rest pose)采样,等价于 "这条轨迹起点的语义"。多帧采样代价大且边际收益低。
- `padding_mode="border"`:轨迹意外越界(浮点累积误差)时取最近边缘像素,不污染。
- 内部强制 fp32 进 `grid_sample`(部分 PyTorch 版本对 bf16 grid_sample 支持不稳定),输出再转回 dino dtype。

### 维度变化
| 段 | 旧 | 新 |
|---|---|---|
| xy 编码 | 2S = 16 | 4K·S = 192 (K=6, S=8) |
| dxy/ddxy/residual | 6S = 48 | 6S = 48 |
| stats | 3 | 3 |
| 视觉锚点 | — | `embed_dim` = 1024 |
| **`F_in` 总计** | **67** | **1267** |

`proj_in` 参数:34K → 649K。整体 TrackEncoder 仍 < 5M 参数。

### 安全性
- `gamma=0`(LayerScale)依然提供 step 0 与 no-track 模型的等价性 —— 视觉特征只影响 track_tokens,而 track_tokens → slots 的通路被 gamma 门控。
- `disable_tracks=True` 时整条分支(含视觉采样)跳过,val ablation 干净。
- ckpt resume 仍 `strict=False`:旧 ckpt 没 freq_bands buffer / 没新 proj_in 形状,新参数全部 xavier 初始化;旧 backbone / aggregator / camera_head / part_slot_router 加载成功。


## 2026-04-23 Phase C 启动脚本

新增 `scripts/launch_phase_c.sh`(基于 `launch_phase1b_motion.sh` + 上述所有 Phase C 改动):
- `--phase 1b`(跳过 1a warmup gate,直接全监督)
- `--lr 2e-5  --grad_clip 0.5`(对齐 1b_motion)
- `--w_type 0.5  --w_axis 1.0  --w_pivot 0.3  --w_scalar 0.2`
- `--w_render 0.3  --w_render_global 0.1  --w_bbox 0.1  --w_dead_opacity 0.1  --w_pose_enc 0.1`
- `--motion_cache_name motion_cache_gt.npz`
- `--w_motion_mask 0.3  --w_motion_track 0.3  --motion_warmup_steps 5000`
- `--use_track_tokens  --track_encoder_lr 3e-4  --freeze_track_warmup_steps 0`(start_step≥21000 已经超出任何合理边界,显式关闭 track-warmup 避免歧义)
- `--max_tracks_per_sample 1024`
- `--gradient_checkpointing  --use_bf16`
- `CUDA_VISIBLE_DEVICES=4,5,6,7  --master_port=29504`(避开同机其他训练)
- `--save_interval 5000  --val_interval 500`
- 输出 `/data2/cyt/checkpoints/art_v20_phase_c/`

resume 路径在新一轮训练改为 `art_v20_phase_c/ckpt_024000.pth`(在 Phase C 自身的 checkpoint 上继续训)。


## 2026-05-01 Phase D 诊断 — sim→real 域差距定位

ckpt_100000 评估结果(`scripts/eval_phase_c.py` + `--dataset_type real`):

| metric | sim (189 scenes) | real (4 scenes) |
|---|---|---|
| fg_iou_mean (identity-free) | **0.771** | **0.175** |
| assign_iou_mean (slot-aligned) | 0.766 | 0.145 |
| mIoU_alpha (GS pipeline)      | 0.601 | 0.002 |
| type_acc / axis_cos           | 0.965 / 0.927 | NaN(无 GT) |

`fg_iou` 是为绕开 SAM2 过分割对 Hungarian 的影响新加的"前景并集 IoU",仍然只有 0.18 →
排除"指标不公平"假设,**域差距是根本性的**。

### 诊断脚本 `scripts/diag_features_pca.py`

对 sim/real 各 4 个场景跑 aggregator,提取 frame-0 patch tokens,做 PCA→RGB:

- **DINOv2(frozen)**:`norm_mean=32.0, std=0.0` 在 sim 和 real 上完全一致(L2-normalized 输出)。
- **Aggregator 输出**:sim `std≈21`,real `std≈6` —— **patch 间方差掉到 sim 的 30%**,即 token 趋同 / 特征坍塌。

### Per-layer 坍塌追踪

记录 24 层 aggregator 每一层输出的 `norm-std`:

```
            层 0-13       层 14-23           峰值
sim  std    1-2           2 → 23 (单调爆涨)  23.0
real std    1-3           3 → 7  (轻涨)      6.6 (在 13 层达峰,之后回落)
```

- 前半层(0-13)在 sim/real 上数值近似,跨域稳定。
- 后半层(14-23)在 sim 上方差从 2.4 涨到 23(10×),在 real 上只从 3 涨到 7(2×),
  甚至在 real_002 上 13 层后**反向衰减** —— 后半层主动把 real 特征压平。

结论:**aggregator 后 ~10 层过拟合了 sim 的合成纹理统计量**;DINOv2 + 前半 aggregator 是跨域可用的。

### 后续策略草案

新增 `scripts/launch_ft_real.sh`(DRAFT,**未运行**):

- 只解冻后 10 个 frame_blocks + global_blocks + 全部 heads,frozen 前 14 层 + DINOv2。
- real 数据无 GT kin/depth,损失只用:fg-mask BCE / motion-mask CE / motion-track。
- `--w_render = 0`(关闭 GS 渲染,避免 alpha=0.002 的退化项目继续搬运梯度)。
- LR `5e-6`,total_steps `5000`,bf16 + grad_ckpt + batch=1。

需要的训练脚手架改动(launch script 头部已列):
1. `train_art.py --dataset_type {sim,real}` + 路由到 `iTACORealDataset`。
2. `train_art.py --n_unfreeze_blocks N`,传入 `set_phase(phase="2", n_unfreeze_blocks=N)`。
3. `art_vggt.py::set_phase()` 在 phase="2" 接受 `n_unfreeze_blocks` 参数(默认 4,向后兼容)。
4. `compute_loss` 在 `dataset_type=="real"` 时跳过 kin/render/bbox/pose_enc。
5. (可选)`ArticulatedDataset` 加 ColorJitter/Blur 增强,sim→real 同时拉近。

未自动执行 —— 等用户确认 scope 后再落地。


### 回归对比:Phase C 训练让 real 域 aggregator 坍塌加重

对 ckpt_024000(Phase C 起点)和 ckpt_100000(终点)分别跑 `diag_features_pca.py`:

| | sim agg std | real agg std |
|---|---|---|
| ckpt 24k | 22–27 | **~13** |
| ckpt 100k | 20–23 | **~6** |

sim 域特征分布几乎不变,但 real 域 patch token 方差在 76k 步训练中**腰斩**。
含义:Phase C 引入的 sim-only 监督(Fourier track 输入 + motion_track + motion_mask + render losses)
持续把 aggregator 拉向 sim 分布,real 上的可分性被牺牲。

→ 进一步支持"只解冻后 ~10 层 + heads 在 real 上 fine-tune"的策略,
  并提示**未来重训 Phase C 时应加 sim-side 域增强或并行 real 弱监督**,
  避免 backbone 越训 real 越差。


## 2026-05-01 iTACO sim_data 接入 — 新 `iTACOSimDataset`

**目的**:Path B —— 抛弃 `data_root_refine`,改用 iTACO sim_data + iTACO real_data 共享同一条 SAM2/MonST3R 弱监督流水线,消除"sim 用 GT mask、real 用 SAM2 mask"造成的输入分布差异。

**数据规模**:786 个场景,11 类,512 hinge + 274 slider 关节(每个 (Cat, instance, joint, view) 算一个独立场景)。

### 关键探测结论(`scripts/inspect_itaco_sim.py` + `scripts/probe_itaco_sim.py`)

1. **GT kin 在全局 JSON**:`/data2/cyt/video2articulation/new_partnet_mobility_dataset_correct_intr_meta.json`,
   每个 `(Cat, instance)` 有 `interaction_list`:`{id, type:hinge|slider, joint:{axis:{origin, direction}, limit}}`。
   axis/pivot 在**对象局部坐标系**,`limit` 单位:hinge=度,slider=米。
2. **场景层 `meta.json`** 给出 `joint_id`,与 interaction_list 中某项匹配。
3. **camera_pose.npy** 是 `(T, 7) = xyz + quat(wxyz)`,c2w direct(USB 测试 bg_centroid_drift=0.106m vs inv 0.81m)。
4. **depth/*.npz** 是 `uint16 mm`(key=`'a'`),需 `astype(float32) / 1000` 转米。
5. **segment/*.npz** 是 `uint8` link-id(key=`'a'`),值 1-9,**不是 part-id**。
6. **MonST3R masks** 18 个稀疏关键帧(整个视频 90-105 帧,等价 stride ≈ T/18)。
7. **actor_pose.pkl** 字典 `{actor_X: list[T] of [7] xyz+wxyz}`,actor_X 后缀对应 segment 值。

### Plan A 实现(`datasets/itaco_sim_dataset.py`)

- **active-part 检测**:对每个 actor 算 `trans_std + rot_std`,排序;最高 var 的 actor 后缀 → active segment id。其他 movable actor → 辅助 slot 2..k。验证:Box/Laptop/Storage/USB 4 个场景全部正确分离 active flap。
- **GT axis/pivot 拟合**:**直接从 active actor 的世界位姿时间序列拟合**,绕过 object-local→world 转换坑(避免 URDF 根节点歧义和 OpenGL/SAPIEN spawn pose 不可知问题):
  - revolute:`R_rel = R_t @ R_0^T`,从 skew(R_rel - R_rel^T)/(2sinθ) 取轴向;`pivot = pinv(I-R_rel) @ (T_t - R_rel @ T_0)`。
  - prismatic:`axis = (T_far - T_0).normalize()`,`pivot = T_0`。
- **gt_scalars**:从 `gt_joint_value.npy` 读,经 `_normalize_scalars` rest-shift 到 `[-1, 1]`。
- **part_masks 构造**:slot 0 = 非 movable 像素(bg + URDF root + 静态附属),slot 1 = active actor 的 segment,slots 2..k = 其他 movable actor。
- **motion_mask** 复用 MonST3R dynamic_mask 联合,统一了 sim 和 real 的 motion 监督路径。
- **tracks** 暂时全零(`has_motion_data=False`),后续可单独跑 CoTracker 预计算填上。

### Smoke-test 结果(`scripts/smoke_test_itaco_sim.py`)

可视化 6 个 Box 场景在 `/data2/cyt/eval/smoke_itaco_sim/`,4 列:
- RGB
- 红 = active part overlay → 与移动的 flap 像素一致
- 绿 = 所有 movable parts overlay → 覆盖全部前景
- 蓝 = MonST3R 动态 mask → 与运动轨迹对齐

Pivot 投影 sanity-check 在 smoke test 里仍是 OpenGL convention 没调对(投到 px=None 或越界),但**数据本身正确**(axis=[1,0,0] 与 Box 顶部铰链方向匹配,pivot=(0, -3.42, 1.0) 落在 box 顶部边缘世界坐标处)。Train 时模型代码用 `depth = -z_cam`(OpenGL),与数据一致。

### 待办(下一步训练前必须落地)

1. **`train_art.py --dataset_type {itaco_sim, itaco_real, mixed}` + ConcatDataset 加权采样**(real 上采样到 30%)。
2. **`compute_loss` 按 `dataset_tag` 分支**:
   - itaco_sim:全部 loss(kin + render + motion + mask)
   - itaco_real:仅 motion + mask + dead_opacity + l1_sparsity(无 GT kin)
3. **`set_phase("2", n_unfreeze_blocks=10)`** —— 只解冻后 10 个 frame_blocks + global_blocks + heads,其余冻结。
4. **(可选)** 给 sim 侧加 ColorJitter / GaussianBlur,sim→real 同时拉近。
5. 先跑一版**纯 itaco_sim**(无 real 混)的 Phase 1b,看 sim 收敛是否正常(验证 dataset 接入无 bug),再加入 real。


## 2026-05-01 Phase D 启动 — sim+real 混训

### 训练脚手架改动(`train_art.py`)

**1. `--dataset_type {legacy, itaco_sim, itaco_real, mixed}`**
- `legacy`:默认,ArticulatedDataset on `data_root`(向后兼容)
- `itaco_sim`:`iTACOSimDataset` 786 场景,带 GT kin
- `itaco_real`:`iTACORealDataset` 4 场景,无 GT kin
- `mixed`:ConcatDataset(itaco_sim + real_repeat),real 内部重复 N 倍后用 DistributedSampler 均匀采样

**2. `--n_unfreeze_blocks 10`**(phase=2)
- 修复了旧代码的 bug:原来不分 frame_blocks 和 global_blocks 各自冻结前 N-4 层,新逻辑分别冻结前 `(depth - n_unfreeze_blocks)` 层
- `set_phase("2", n_unfreeze_blocks=N)` 已传播到 3 个调用点

**3. `compute_loss` 无需改动**
- `has_kin_gt=False` 分支(line 760-780)已处理 real 批次:跳过 kin/render/bbox/pose_enc,仅保留 mask + sparsity + motion_mask + motion_track
- sim 批次走完整 loss(kin + render + motion + mask)

**4. DDP 修复(WeightedRandomSampler → ConcatDataset repeat)**
- `WeightedRandomSampler` 不支持 `set_epoch()`,与 DDP 不兼容
- 改为 real 数据集内部重复 71×,与 sim 形成 953 总样本(`real_frac≈0.30`),用标准 DistributedSampler

### 启动脚本(`scripts/launch_phase_d.sh`)

| 参数 | 值 | 说明 |
|---|---|---|
| `--dataset_type` | `mixed` | sim+real 混训 |
| `--phase` | `2` | fine-tune 模式 |
| `--n_unfreeze_blocks` | `10` | 解冻后 10 对 frame+global blocks(前 14 对冻结) |
| `--lr` | `5e-6` | 轻步防止灾难遗忘 |
| `--total_steps` | `130000` | 从 100000 resume,训 30000 步 |
| `--real_mix_ratio` | `0.3` | real 占 30% |
| `--w_motion_mask` | `0.5` | 弱监督主信号(sim/real 共享 MonST3R) |
| `--w_render` | `0.1` | 降低(仅 sim 侧) |
| `--w_type/axis/pivot/scalar` | `0.3/0.5/0.2/0.1` | 保持 GT 监督(仅 sim 侧) |
| `--real_data_root` | `/data2/cyt/video2articulation/real_data` | iTACO real |
| `--data_root` | `/data2/cyt/video2articulation/sim_data` | iTACO sim |

### 当前状态

训练已启动,4×GPU(4,5,6,7),进程稳定,无 NaN,约 gpu_mem 9.6GB/rank。

预计每 500 步一次 val(仅 sim 验证集,117 场景),每 2500 步保存一次 ckpt。
首个 val checkpoint 在 step 100500 附近,届时可看 sim 侧 assign_iou 是上升还是下降。

### 待观察指标

1. **sim assign_iou / fg_iou**:Phase C 100k=0.77,Phase D 不应显著下降(超过 0.03)
2. **real fg_iou**(手动 eval):每 5000 步跑一次 `eval_phase_c.py --dataset_type real`
3. **gamma 值**:TrackEncoder LayerScale gamma,应保持在 0.1-0.3 范围
