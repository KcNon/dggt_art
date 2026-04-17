"""
eval_gs.py — Phase 1b GS点质量专项评估

专注于 per_part_alpha_render_loss 所衡量的渲染质量，以及GS点自身的
分布质量。在已有 eval_phase1b.py 的运动学指标基础上，补充：

渲染质量指标:
  - Alpha Render IoU    : 每帧每个匹配part，GS投影alpha图 vs GT mask的IoU
  - Alpha Render Dice   : 同train_art中per_part_alpha_render_loss计算逻辑
  - Alpha Render BCE    : binary cross-entropy（用于对比训练曲线）

GS点分布质量指标:
  - BBox Coverage       : GS中心点落在预测bbox内的比例（每slot）
  - Mean Opacity        : 每slot的平均opacity（越高越健康）
  - Dead Slot Ratio     : 平均opacity < 0.05的slot占比
  - Opacity Entropy     : 各slot opacity分布的熵（度量活跃度均匀性）

可视化:
  - per_scene/: 每场景一张图，包含：
      输入帧 | 每帧alpha图 | GT mask | alpha图叠加
  - canonical/: 每场景canonical GS点云的2D投影 (XY/XZ/YZ三视图)
  - opacity_dist.png : 全局opacity分布直方图
  - summary.png      : 各指标直方图汇总

用法:
  python eval_gs.py \\
      --checkpoint /data2/cyt/checkpoints/art_v15_phase1b/ckpt_050000.pth \\
      --data_root  /data2/cyt/data_root \\
      --output_dir ./eval_results/gs_v15_step50000 \\
      [--num_scenes 0]   # 0 = 全部val集
      [--gpu 0]
      [--max_vis 30]
      [--sigma 0.8]      # 与训练一致的Gaussian kernel sigma
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from dggt.models.art_vggt import ArtVGGT
from dggt.utils.rigid_transform import apply_rigid_transform
from datasets.articulated_dataset import ArticulatedDataset


# ============================================================================
# 常量
# ============================================================================

SLOT_COLORS = np.array([
    [0.85, 0.85, 0.85],  # slot 0: 静态背景
    [0.95, 0.20, 0.20],  # slot 1: 红
    [0.20, 0.75, 0.20],  # slot 2: 绿
    [0.20, 0.40, 0.95],  # slot 3: 蓝
    [0.95, 0.70, 0.10],  # slot 4: 橙
    [0.70, 0.20, 0.90],  # slot 5: 紫
    [0.10, 0.85, 0.85],  # slot 6: 青
    [0.95, 0.40, 0.70],  # slot 7: 粉
], dtype=np.float32)

DEAD_OPACITY_THRESH = 0.05   # mean opacity below this → dead slot


# ============================================================================
# GS alpha map 计算（与 train_art.per_part_alpha_render_loss 逻辑一致）
# ============================================================================

@torch.no_grad()
def compute_slot_alpha_map(
    mu_p:          torch.Tensor,   # [N_g, 3]  canonical positions
    opacity_p:     torch.Tensor,   # [N_g]     opacity values ∈ (0,1)
    motion_probs:  torch.Tensor,   # [3]       softmax [static, prismatic, revolute]
    axis:          torch.Tensor,   # [3]
    pivot:         torch.Tensor,   # [3]
    scalar:        torch.Tensor,   # []        per-frame scalar
    R_w2c:         torch.Tensor,   # [3, 3]   world-to-cam rotation
    t_w2c:         torch.Tensor,   # [3]       world-to-cam translation
    K:             torch.Tensor,   # [3, 3]   intrinsics
    H_p: int, W_p: int,
    patch_size: int = 14,
    sigma: float = 0.8,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将一个slot的canonical GS点经刚体变换投影到图像patch空间，
    生成软alpha图。

    Returns:
        alpha_map : [H_p, W_p]  ∈ [0, 1]
        behind    : [N_g]  bool, True = 在相机后面
    """
    if device is None:
        device = mu_p.device

    N_g = mu_p.shape[0]
    # 生成identity四元数（位置变换不依赖orientation）
    quats = torch.zeros(N_g, 4, device=device)
    quats[:, 0] = 1.0   # w=1 identity

    # 刚体变换: canonical → world
    mu_world, _ = apply_rigid_transform(
        mu_p, quats, motion_probs, axis, pivot, scalar
    )   # [N_g, 3]

    # world → camera
    mu_cam = mu_world @ R_w2c.T + t_w2c.unsqueeze(0)   # [N_g, 3]

    # 深度 & behind判断 (OpenGL: z_cam < 0 for front-facing)
    behind = mu_cam[:, 2] >= 0.0   # [N_g]
    depth  = (-mu_cam[:, 2]).clamp(min=0.01)

    # 投影到patch坐标系
    fx = K[0, 0] / patch_size
    fy = K[1, 1] / patch_size
    cx = K[0, 2] / patch_size
    cy = K[1, 2] / patch_size

    u = fx * mu_cam[:, 0] / depth + cx   # [N_g]
    v = fy * mu_cam[:, 1] / depth + cy   # [N_g]

    # patch坐标网格 [H_p, W_p, 2]
    gy = torch.arange(H_p, device=device, dtype=torch.float32) + 0.5
    gx = torch.arange(W_p, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')

    # Gaussian kernel splatting
    dx = grid_x.unsqueeze(-1) - u.unsqueeze(0).unsqueeze(0)   # [H_p, W_p, N_g]
    dy = grid_y.unsqueeze(-1) - v.unsqueeze(0).unsqueeze(0)
    dist2  = dx * dx + dy * dy
    kernel = torch.exp(-dist2 / (2 * sigma ** 2))              # [H_p, W_p, N_g]

    # 在相机后面的点opacity置零
    op = opacity_p * (~behind).float()   # [N_g]
    alpha_map = (kernel * op.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    alpha_map = alpha_map.clamp(0, 1)   # [H_p, W_p]

    return alpha_map, behind


# ============================================================================
# Hungarian matching（与 eval_phase1b.py 一致：argmax二值化）
# ============================================================================

@torch.no_grad()
def argmax_hungarian_match(
    pred_maps: torch.Tensor,   # [P, H, W]
    gt_masks:  torch.Tensor,   # [P_gt, H, W]  binary
) -> tuple[np.ndarray, np.ndarray]:
    P    = pred_maps.shape[0]
    P_gt = gt_masks.shape[0]
    dev  = pred_maps.device

    pred_argmax = pred_maps.argmax(dim=0)
    pred_bin = (
        torch.arange(P, device=dev).view(P, 1, 1) == pred_argmax.unsqueeze(0)
    ).float()

    cost_mat = torch.zeros(P, P_gt, device=dev)
    for i in range(P):
        for j in range(P_gt):
            inter = (pred_bin[i] * gt_masks[j]).sum()
            union = (pred_bin[i] + gt_masks[j]).clamp(0, 1).sum()
            cost_mat[i, j] = 1.0 - inter / (union + 1e-6)

    gt_active = [j for j in range(1, P_gt) if gt_masks[j].sum() > 0]
    if not gt_active:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    cost_sub = cost_mat[1:, :][:, gt_active].cpu().numpy()
    row_sub, col_sub = linear_sum_assignment(cost_sub)
    pred_idx = row_sub + 1
    gt_idx   = np.array(gt_active)[col_sub]
    return pred_idx, gt_idx


# ============================================================================
# 单场景 GS 评估
# ============================================================================

@torch.no_grad()
def evaluate_gs_scene(
    model:      torch.nn.Module,
    batch:      dict,
    device:     torch.device,
    patch_size: int   = 14,
    sigma:      float = 0.8,
) -> dict:
    """
    对单场景评估GS点渲染质量和分布质量。

    Returns dict with scalar metrics and raw tensors (prefixed '_').
    """
    images     = batch["images"].unsqueeze(0).to(device)       # [1, S, 3, H, W]
    extrinsics = batch["extrinsics"].unsqueeze(0).to(device)   # [1, S, 4, 4]
    intrinsics = batch["intrinsics"].unsqueeze(0).to(device)   # [1, 3, 3]
    timestamps = batch["timestamps"].unsqueeze(0).to(device)   # [1, S]
    part_masks = batch["part_masks"].to(device)                # [S, P_gt, H, W]

    S, P_gt, H, W = part_masks.shape
    H_p = H // patch_size
    W_p = W // patch_size

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        preds = model(images, extrinsics, intrinsics, timestamps)

    # Unpack predictions (batch dim 0), ensure float32 (autocast可能产出fp16)
    assign_maps   = preds["assign_maps"][0].float()            # [P, H_p, W_p]
    motion_logits = preds["motion_type_logits"][0].float()     # [P, 2]
    axis_pred     = preds["axis"][0].float()                   # [P, 3]
    pivot_pred    = preds["pivot"][0].float()                  # [P, 3]
    scalars_pred  = preds["scalars"][0].float()                # [P, S]
    gs_mu         = preds["gs_mu"][0].float()                  # [P, N_g, 3]
    gs_opacity    = preds["gs_opacity"][0].float().squeeze(-1) # [P, N_g]
    bbox_center   = preds["bbox_center"][0].float()            # [P, 3]
    bbox_size     = preds["bbox_size"][0].float()              # [P, 3]

    P  = assign_maps.shape[0]
    N_g = gs_mu.shape[1]
    K   = intrinsics[0]   # [3, 3]

    # softmax motion probs [P, 3] (前置static列)
    motion_probs_2 = torch.softmax(motion_logits.float(), dim=-1)    # [P, 2]
    static_col     = torch.zeros(P, 1, device=device)
    motion_probs   = torch.cat([static_col, motion_probs_2], dim=-1) # [P, 3]
    motion_probs[0] = torch.tensor([1., 0., 0.], device=device)      # slot0 = static

    # Upsample assign_maps for matching
    assign_up = F.interpolate(
        assign_maps.unsqueeze(0), (H, W), mode="bilinear", align_corners=False
    ).squeeze(0)   # [P, H, W]

    # GT masks: max-project over frames → binary [P_gt, H, W]
    gt_masks_bin = (part_masks.max(dim=0).values > 0.5).float()

    # Hungarian matching
    pred_idx, gt_idx = argmax_hungarian_match(assign_up, gt_masks_bin)

    # ── 1. Per-frame alpha render metrics ────────────────────────────────────
    # 对每个匹配的(slot, 帧)计算 alpha map，与GT mask对比
    alpha_ious_per_slot  = {}   # pred_slot → list of per-frame IoU
    alpha_dice_per_slot  = {}
    alpha_maps_all       = {}   # (slot, frame) → alpha_map [H_p, W_p]

    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        alpha_ious_per_slot[pi]  = []
        alpha_dice_per_slot[pi]  = []

        for t in range(S):
            E_c2w = extrinsics[0, t]                  # [4, 4]
            R_c2w = E_c2w[:3, :3]
            t_c2w = E_c2w[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -(R_w2c @ t_c2w)

            alpha_map, _ = compute_slot_alpha_map(
                mu_p         = gs_mu[pi],
                opacity_p    = gs_opacity[pi],
                motion_probs = motion_probs[pi],
                axis         = axis_pred[pi],
                pivot        = pivot_pred[pi],
                scalar       = scalars_pred[pi, t],
                R_w2c        = R_w2c,
                t_w2c        = t_w2c,
                K            = K,
                H_p          = H_p, W_p = W_p,
                patch_size   = patch_size,
                sigma        = sigma,
                device       = device,
            )   # [H_p, W_p]

            alpha_maps_all[(pi, t)] = alpha_map.cpu()

            # Downsample GT mask to patch resolution
            gt_p = F.adaptive_avg_pool2d(
                part_masks[t, gi].float().unsqueeze(0).unsqueeze(0),
                (H_p, W_p)
            ).squeeze()   # [H_p, W_p]

            # IoU: threshold alpha at 0.5
            alpha_bin = (alpha_map > 0.5).float()
            inter = (alpha_bin * gt_p).sum().item()
            union = (alpha_bin + gt_p).clamp(0, 1).sum().item()
            iou   = inter / (union + 1e-6)
            alpha_ious_per_slot[pi].append(iou)

            # Dice
            inter_s = (alpha_map * gt_p).sum().item()
            denom   = alpha_map.sum().item() + gt_p.sum().item()
            dice    = (2 * inter_s + 1e-6) / (denom + 1e-6)
            alpha_dice_per_slot[pi].append(1.0 - dice)   # dice loss

    # Flatten: mean over slots and frames
    all_ious  = [v for vals in alpha_ious_per_slot.values() for v in vals]
    all_dices = [v for vals in alpha_dice_per_slot.values() for v in vals]
    mean_alpha_iou  = float(np.mean(all_ious))  if all_ious  else 0.0
    mean_alpha_dice = float(np.mean(all_dices)) if all_dices else 1.0

    # ── 2. GS点分布质量 ──────────────────────────────────────────────────────
    # BBox Coverage: GS中心落在预测bbox内的比例
    bc = bbox_center   # [P, 3]
    bs = bbox_size     # [P, 3]  half-extent
    lo = bc - bs       # [P, 3]
    hi = bc + bs       # [P, 3]
    # gs_mu: [P, N_g, 3]
    inside = (
        (gs_mu >= lo.unsqueeze(1)) & (gs_mu <= hi.unsqueeze(1))
    ).all(dim=-1).float()   # [P, N_g]
    bbox_coverage_per_slot = inside.mean(dim=-1).cpu().tolist()   # [P]

    # Per-slot mean opacity
    mean_opacity_per_slot = gs_opacity.mean(dim=-1).cpu().tolist()   # [P]

    # Dead slot ratio
    dead_mask    = [op < DEAD_OPACITY_THRESH for op in mean_opacity_per_slot]
    dead_ratio   = float(sum(dead_mask)) / P

    # Opacity entropy per slot (measure of "peakiness")
    # entropy = -sum(p * log(p)) for the N_g opacity values treated as a distribution
    def _entropy(vals: torch.Tensor) -> float:
        vals = vals.clamp(1e-8, 1.0)
        p    = vals / vals.sum()
        return float(-(p * p.log()).sum().item())

    opacity_entropy_per_slot = [
        _entropy(gs_opacity[p]) for p in range(P)
    ]

    # Coverage only for matched slots
    matched_coverage = [bbox_coverage_per_slot[pi] for pi in pred_idx.tolist()]

    return {
        "scene_id":           batch["scene_id"],
        "n_matched":          len(pred_idx),
        # Render metrics
        "mean_alpha_iou":     mean_alpha_iou,
        "mean_alpha_dice":    mean_alpha_dice,
        # GS分布
        "bbox_coverage":      float(np.mean(matched_coverage)) if matched_coverage else 0.0,
        "mean_opacity":       float(np.mean(mean_opacity_per_slot)),
        "dead_ratio":         dead_ratio,
        "opacity_entropy":    float(np.mean(opacity_entropy_per_slot)),
        # Per-slot breakdowns (list of P values)
        "_pred_idx":          pred_idx,
        "_gt_idx":            gt_idx,
        "_mean_opacity":      mean_opacity_per_slot,
        "_bbox_coverage":     bbox_coverage_per_slot,
        "_dead_mask":         dead_mask,
        "_alpha_ious":        alpha_ious_per_slot,
        # Tensors for visualization
        "_images":            batch["images"],       # [S, 3, H, W]
        "_assign_up":         assign_up.cpu(),       # [P, H, W]
        "_gt_masks_bin":      gt_masks_bin.cpu(),    # [P_gt, H, W]
        "_gs_mu":             gs_mu.cpu(),           # [P, N_g, 3]
        "_gs_opacity":        gs_opacity.cpu(),      # [P, N_g]
        "_alpha_maps":        alpha_maps_all,        # {(pi,t): [H_p,W_p]}
        "_part_masks":        part_masks.cpu(),      # [S, P_gt, H, W]
        "_motion_probs":      motion_probs.cpu(),    # [P, 3]
        "_scalars":           scalars_pred.cpu(),    # [P, S]
        "_bbox_center":       bc.cpu(),              # [P, 3]
        "_bbox_size":         bs.cpu(),              # [P, 3]
        "_axis_pred":         axis_pred.cpu(),       # [P, 3]
        "_pivot_pred":        pivot_pred.cpu(),      # [P, 3]
    }


# ============================================================================
# 可视化: 单场景 alpha图质量
# ============================================================================

def save_alpha_vis(result: dict, out_path: str, patch_size: int = 14):
    """
    每个匹配的slot一行，显示：
      [帧0 原图] [帧0 alpha图] [帧0 GT mask] | [帧1 ...] | ...
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pred_idx    = result["_pred_idx"]
    gt_idx      = result["_gt_idx"]
    images      = result["_images"]          # [S, 3, H, W]
    alpha_maps  = result["_alpha_maps"]      # {(pi,t): [H_p,W_p]}
    part_masks  = result["_part_masks"]      # [S, P_gt, H, W]
    dead_mask   = result["_dead_mask"]
    mean_op     = result["_mean_opacity"]

    S = images.shape[0]
    n_matched = len(pred_idx)
    if n_matched == 0:
        return

    n_cols = S * 3   # per frame: image + alpha + GT mask
    n_rows = n_matched

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    for row, (pi, gi) in enumerate(zip(pred_idx.tolist(), gt_idx.tolist())):
        for t in range(S):
            col_base = t * 3

            img_np = images[t].permute(1, 2, 0).numpy()   # [H, W, 3]
            H, W   = img_np.shape[:2]
            H_p    = H // patch_size
            W_p    = W // patch_size

            # --- col 0: input image ---
            ax = axes[row, col_base]
            ax.imshow(img_np)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Frame {t}\nInput", fontsize=7)

            # --- col 1: alpha map (upsampled) ---
            alpha = alpha_maps.get((pi, t))
            ax = axes[row, col_base + 1]
            if alpha is not None:
                alpha_up = F.interpolate(
                    alpha.unsqueeze(0).unsqueeze(0),
                    (H, W), mode="bilinear", align_corners=False
                ).squeeze().numpy()
                # 叠加在图像上
                color = SLOT_COLORS[pi % len(SLOT_COLORS)]
                overlay = img_np.copy()
                overlay += alpha_up[:, :, None] * color[None, None, :]
                ax.imshow(np.clip(overlay, 0, 1))
                iou = result["_alpha_ious"].get(pi, [0.0] * S)
                iou_t = iou[t] if t < len(iou) else 0.0
                if row == 0:
                    ax.set_title(f"Frame {t}\nAlpha (IoU)", fontsize=7)
                ax.set_xlabel(f"IoU={iou_t:.3f}", fontsize=7)
            ax.axis("off")

            # --- col 2: GT mask ---
            ax = axes[row, col_base + 2]
            gt_m = part_masks[t, gi].numpy()   # [H, W]
            ax.imshow(np.clip(img_np + gt_m[:, :, None] * 0.6, 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Frame {t}\nGT Mask", fontsize=7)

        # Row label: slot info
        dead_str = "DEAD" if dead_mask[pi] else ""
        axes[row, 0].set_ylabel(
            f"Slot {pi}→GT{gi}\nop={mean_op[pi]:.3f} {dead_str}",
            fontsize=7,
        )

    fig.suptitle(
        f"Scene: {result['scene_id']} | "
        f"AlphaIoU={result['mean_alpha_iou']:.3f} | "
        f"Dice={result['mean_alpha_dice']:.3f} | "
        f"DeadRatio={result['dead_ratio']:.2f}",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 可视化: canonical GS点云三视图
# ============================================================================

def save_canonical_vis(result: dict, out_path: str):
    """
    Canonical空间GS点云: XY, XZ, YZ三个投影平面。
    每个slot用不同颜色，点大小和透明度由opacity决定。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gs_mu      = result["_gs_mu"]       # [P, N_g, 3]
    gs_opacity = result["_gs_opacity"]  # [P, N_g]
    pred_idx   = result["_pred_idx"]
    dead_mask  = result["_dead_mask"]

    P, N_g, _ = gs_mu.shape
    planes = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, xi, yi) in zip(axes, planes):
        for p in range(P):
            mu   = gs_mu[p].numpy()       # [N_g, 3]
            op   = gs_opacity[p].numpy()  # [N_g]
            col  = SLOT_COLORS[p % len(SLOT_COLORS)]
            dead = dead_mask[p]
            lbl  = f"Slot {p}" + (" [DEAD]" if dead else "")

            # 只画opacity > 0.02的点以避免图面太乱
            mask = op > 0.02
            if mask.sum() == 0:
                continue

            ax.scatter(
                mu[mask, xi], mu[mask, yi],
                s   = (op[mask] * 30).clip(1, 60),
                c   = [col],
                alpha = float(np.mean(op[mask])) * 0.8 + 0.1,
                label = lbl,
                linewidths = 0,
            )

        ax.set_xlabel(["X", "Y", "Z"][xi], fontsize=9)
        ax.set_ylabel(["X", "Y", "Z"][yi], fontsize=9)
        ax.set_title(f"Canonical GS {name}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle(
        f"Scene: {result['scene_id']} — Canonical GS Points "
        f"(BBoxCov={result['bbox_coverage']:.3f}  "
        f"MeanOp={result['mean_opacity']:.3f})",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# PLY 导出: canonical + per-frame world (按 slot 着色)
# ============================================================================

def _write_ply(
    xyz:   np.ndarray,   # [N, 3] float
    rgb:   np.ndarray,   # [N, 3] uint8
    path:  str,
) -> None:
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    lines = [
        f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
        f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}"
        for i in range(n)
    ]
    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(lines))
        f.write("\n")


def _stack_slots_colored(
    points_per_slot: list,    # list of [N_g, 3] np arrays
    opacities:       list,    # list of [N_g] np arrays
    op_thresh:       float = 0.02,
) -> tuple:
    """Stack slot points, color each by SLOT_COLORS, filter low opacity."""
    xyz_all, rgb_all = [], []
    for p, (mu, op) in enumerate(zip(points_per_slot, opacities)):
        keep = op > op_thresh
        if keep.sum() == 0:
            continue
        col = SLOT_COLORS[p % len(SLOT_COLORS)]
        rgb = np.tile((col * 255).astype(np.uint8)[None, :], (int(keep.sum()), 1))
        xyz_all.append(mu[keep])
        rgb_all.append(rgb)
    if not xyz_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(xyz_all, 0), np.concatenate(rgb_all, 0)


def save_canonical_ply(result: dict, out_path: str, op_thresh: float = 0.02) -> int:
    """Export canonical-space GS centres, colored per slot."""
    gs_mu      = result["_gs_mu"].numpy()        # [P, N_g, 3]
    gs_opacity = result["_gs_opacity"].numpy()   # [P, N_g]
    P = gs_mu.shape[0]
    xyz, rgb = _stack_slots_colored(
        [gs_mu[p] for p in range(P)],
        [gs_opacity[p] for p in range(P)],
        op_thresh=op_thresh,
    )
    _write_ply(xyz, rgb, out_path)
    return xyz.shape[0]


def save_world_ply(
    result:    dict,
    out_path:  str,
    frame_idx: int,
    op_thresh: float = 0.02,
) -> int:
    """Apply per-slot rigid transform at given frame, export world-space PLY."""
    from dggt.utils.rigid_transform import apply_rigid_transform

    gs_mu        = result["_gs_mu"]              # [P, N_g, 3]
    gs_opacity   = result["_gs_opacity"].numpy() # [P, N_g]
    motion_probs = result["_motion_probs"]       # [P, 3]
    axis_pred    = result["_axis_pred"]          # [P, 3]
    pivot_pred   = result["_pivot_pred"]         # [P, 3]
    scalars      = result["_scalars"]            # [P, S]
    P, N_g, _    = gs_mu.shape

    pts_per_slot = []
    for p in range(P):
        dummy_q = torch.zeros(N_g, 4, dtype=gs_mu.dtype)
        dummy_q[:, 0] = 1.0
        world_pts, _ = apply_rigid_transform(
            points            = gs_mu[p],
            quats             = dummy_q,
            motion_type_probs = motion_probs[p],
            axis              = axis_pred[p],
            pivot             = pivot_pred[p],
            scalar            = scalars[p, frame_idx],
        )
        pts_per_slot.append(world_pts.detach().cpu().numpy())

    xyz, rgb = _stack_slots_colored(
        pts_per_slot,
        [gs_opacity[p] for p in range(P)],
        op_thresh=op_thresh,
    )
    _write_ply(xyz, rgb, out_path)
    return xyz.shape[0]


# ============================================================================
# 全局指标直方图
# ============================================================================

def save_summary_plot(all_results: list, step: int, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = {
        "Alpha Render IoU":   [r["mean_alpha_iou"]   for r in all_results],
        "Alpha Dice Loss":    [r["mean_alpha_dice"]   for r in all_results],
        "BBox Coverage":      [r["bbox_coverage"]     for r in all_results],
        "Mean Opacity":       [r["mean_opacity"]      for r in all_results],
        "Dead Slot Ratio":    [r["dead_ratio"]        for r in all_results],
        "Opacity Entropy":    [r["opacity_entropy"]   for r in all_results],
    }
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "grey"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for ax, (title, vals), color in zip(axes_flat, metrics.items(), colors):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            ax.set_title(title)
            continue
        mean_v = float(np.mean(vals))
        ax.hist(vals, bins=25, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(mean_v, color="red", linestyle="--",
                   label=f"mean={mean_v:.4f}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"GS Eval @ step {step}  ({len(all_results)} val scenes)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_opacity_dist_plot(all_results: list, out_path: str):
    """全局opacity分布: 按slot分组的violin plot。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Collect per-slot opacity lists (across all scenes)
    max_slots = 8
    op_per_slot = [[] for _ in range(max_slots)]
    for r in all_results:
        for p, op_mean in enumerate(r.get("mean_opacity_per_slot", [])):
            if p < max_slots:
                op_per_slot[p].append(op_mean)

    # 过滤空列表，记录对应positions
    valid_data = [(p, op) for p, op in enumerate(op_per_slot) if len(op) >= 2]
    if not valid_data:
        return   # 样本太少，跳过violin plot

    positions, data = zip(*valid_data)
    fig, ax = plt.subplots(figsize=(12, 5))
    parts = ax.violinplot(
        list(data),
        positions=list(positions),
        showmeans=True, showmedians=True,
    )
    for i, pc in enumerate(parts["bodies"]):
        col = SLOT_COLORS[positions[i] % len(SLOT_COLORS)]
        pc.set_facecolor(col)
        pc.set_alpha(0.7)

    ax.axhline(DEAD_OPACITY_THRESH, color="red", linestyle="--",
               label=f"dead threshold ({DEAD_OPACITY_THRESH})")
    ax.set_xticks(range(max_slots))
    ax.set_xticklabels([f"Slot {p}" for p in range(max_slots)])
    ax.set_ylabel("Mean Opacity")
    ax.set_title("Per-Slot Mean Opacity Distribution (across val scenes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir  = Path(cfg.output_dir)
    vis_dir  = out_dir / "per_scene"
    can_dir  = out_dir / "canonical"
    ply_dir  = out_dir / "ply"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    can_dir.mkdir(exist_ok=True)
    if cfg.export_ply:
        ply_dir.mkdir(exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"Loading: {cfg.checkpoint}")
    ckpt        = torch.load(cfg.checkpoint, map_location="cpu")
    saved_cfg   = ckpt.get("cfg", {})
    step        = ckpt.get("step", 0)
    n_gaussians = saved_cfg.get("n_gaussians", 256)
    img_size    = saved_cfg.get("img_size", 518)
    scene_radius = saved_cfg.get("scene_radius", 1.0)
    num_frames  = saved_cfg.get("num_frames", cfg.num_frames)
    val_ratio   = saved_cfg.get("val_ratio", 0.15)
    exclude_cams = set(saved_cfg.get("exclude_cams", ["cam_00"]))
    print(f"  Step: {step}  |  n_gaussians: {n_gaussians}  |  num_frames: {num_frames}")

    # ── Build model ────────────────────────────────────────────────────────
    model = ArtVGGT(
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        num_slots=8,
        n_gaussians=n_gaussians,
        scene_radius=scene_radius,
        use_camera_head=True,
        stop_gradient_plucker=False,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Dataset (val split only, 与训练时一致) ─────────────────────────────
    full_ds = ArticulatedDataset(
        data_root    = cfg.data_root,
        target_size  = img_size,
        num_frames   = num_frames,
        max_parts    = 8,
        split        = "val",
        val_ratio    = val_ratio,
        split_seed   = 42,
        exclude_cams = exclude_cams,
    )
    val_ds = full_ds

    if cfg.num_scenes > 0:
        indices = list(range(min(cfg.num_scenes, len(val_ds))))
    else:
        indices = list(range(len(val_ds)))

    print(f"  Val scenes: {len(indices)} / {len(full_ds)}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_results = []

    for i, idx in enumerate(indices):
        try:
            batch    = val_ds[idx]
        except Exception as e:
            print(f"[{i+1:3d}/{len(indices)}] dataset load ERROR: {e}")
            continue
        scene_id = batch["scene_id"]
        print(f"[{i+1:3d}/{len(indices)}] {scene_id}  ", end="", flush=True)

        try:
            result = evaluate_gs_scene(
                model, batch, device,
                patch_size=14, sigma=cfg.sigma,
            )
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            continue

        result["_step"] = step
        print(
            f"AlphaIoU={result['mean_alpha_iou']:.3f}  "
            f"Dice={result['mean_alpha_dice']:.3f}  "
            f"BBoxCov={result['bbox_coverage']:.3f}  "
            f"MeanOp={result['mean_opacity']:.3f}  "
            f"Dead={result['dead_ratio']:.2f}"
        )

        # 序列化时去掉tensor字段，但保留per-slot列表（用于聚合图）
        rec = {k: v for k, v in result.items() if not k.startswith("_")}
        rec["mean_opacity_per_slot"]  = result["_mean_opacity"]   # list[P]
        rec["bbox_coverage_per_slot"] = result["_bbox_coverage"]  # list[P]
        all_results.append(rec)

        # Visualisations
        if i < cfg.max_vis:
            safe_id = scene_id.replace("/", "_").replace(os.sep, "_")

            # Alpha质量图
            save_alpha_vis(result, str(vis_dir / f"{safe_id}.png"), patch_size=14)

            # Canonical点云图
            save_canonical_vis(result, str(can_dir / f"{safe_id}.png"))

            # PLY 导出: canonical + 首/中/末帧 world
            if cfg.export_ply:
                n_can = save_canonical_ply(
                    result, str(ply_dir / f"{safe_id}_canonical.ply")
                )
                S_frames = result["_scalars"].shape[1]
                frames_to_dump = sorted(set([0, S_frames // 2, S_frames - 1]))
                for fi in frames_to_dump:
                    save_world_ply(
                        result,
                        str(ply_dir / f"{safe_id}_world_f{fi:02d}.ply"),
                        frame_idx=fi,
                    )
                print(f"           ply: {n_can} canonical pts, "
                      f"{len(frames_to_dump)} world frames")

    if not all_results:
        print("No results collected.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────
    def _mean(key):
        vals = [r[key] for r in all_results if not math.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "mean_alpha_iou":   _mean("mean_alpha_iou"),
        "mean_alpha_dice":  _mean("mean_alpha_dice"),
        "bbox_coverage":    _mean("bbox_coverage"),
        "mean_opacity":     _mean("mean_opacity"),
        "dead_ratio":       _mean("dead_ratio"),
        "opacity_entropy":  _mean("opacity_entropy"),
    }

    print("\n" + "=" * 65)
    print(f"  GS Eval @ step {step}  ({len(all_results)} val scenes)")
    print("-" * 65)
    print(f"  {'Alpha Render IoU':<32}  {summary['mean_alpha_iou']:>8.4f}")
    print(f"  {'Alpha Dice Loss':<32}  {summary['mean_alpha_dice']:>8.4f}")
    print(f"  {'BBox Coverage':<32}  {summary['bbox_coverage']:>8.4f}")
    print(f"  {'Mean Opacity':<32}  {summary['mean_opacity']:>8.4f}")
    print(f"  {'Dead Slot Ratio':<32}  {summary['dead_ratio']:>8.4f}")
    print(f"  {'Opacity Entropy':<32}  {summary['opacity_entropy']:>8.4f}")
    print("=" * 65)

    # ── Save JSON ──────────────────────────────────────────────────────────
    report = {
        "checkpoint":  cfg.checkpoint,
        "step":        step,
        "n_scenes":    len(all_results),
        "sigma":       cfg.sigma,
        "summary":     summary,
        "per_scene":   all_results,
    }
    report_path = out_dir / "gs_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"\nReport: {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    save_summary_plot(all_results, step, str(out_dir / "summary.png"))
    save_opacity_dist_plot(all_results, str(out_dir / "opacity_dist.png"))
    if cfg.max_vis > 0:
        print(f"Alpha vis:     {vis_dir}/")
        print(f"Canonical vis: {can_dir}/")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ArtVGGT Phase 1b GS点质量评估")
    p.add_argument("--checkpoint", required=True,
                   help="训练好的.pth checkpoint路径")
    p.add_argument("--data_root",  required=True,
                   help="数据集根目录（与训练时--data_root一致）")
    p.add_argument("--output_dir", default="./eval_results/gs_eval",
                   help="结果输出目录")
    p.add_argument("--num_scenes", type=int, default=0,
                   help="评估的val场景数量（0 = 全部）")
    p.add_argument("--num_frames", type=int, default=2,
                   help="每场景帧数（若checkpoint未记录则使用此值）")
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--max_vis",    type=int, default=30,
                   help="保存可视化的场景数量上限")
    p.add_argument("--sigma",      type=float, default=0.8,
                   help="Gaussian kernel sigma（patch单位，与训练一致）")
    p.add_argument("--export_ply", action="store_true",
                   help="导出 canonical + per-frame world PLY (按 slot 着色)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
