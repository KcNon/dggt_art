"""
train_art.py — Three-phase training for ArtVGGT (FAST-4D).

Phase 1a  (steps 0         … warmup_steps):
  Warmup: only Mask Dice/BCE loss on assign_maps.
  KinematicHead, DynamicsHead, GaussianHead are frozen.
  Hungarian matching runs every step (cheap IoU cost matrix).
  Transition trigger: validation mean-IoU ≥ warmup_iou_threshold
  (checked every val_interval steps; must hold for 3 consecutive checks).

Phase 1b  (steps warmup_steps … phase1_steps):
  Full supervised training on synthetic PartNet-Mobility data.
  All heads unfrozen. Hungarian matching + axis sign-fix every step.

Phase 2   (steps phase1_steps … total_steps):
  Weak-supervised domain adaptation on real data.
  CameraHead predicts pose; Plücker rays are stop-grad from pose.
  Aggregator early layers frozen (all but last 4).
  Pseudo-masks from SAM2 (weight 0.05).

Usage:
  torchrun --nproc_per_node=8 train_art.py \
      --data_root /data/partnet_mobility \
      --real_data_root /data/articulat3d_real \
      --output_dir ./checkpoints/art \
      [--phase 1a|1b|2] [--resume path/to/ckpt.pth]
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dggt.models.art_vggt import ArtVGGT
from dggt.utils.dead_slot_gating import (
    detect_dead_slots, slot_sparsity_loss, dead_slot_opacity_loss
)
from dggt.utils.hungarian_matching import match_and_fix, reorder_by_match
from scipy.optimize import linear_sum_assignment
from dggt.utils.rigid_transform import apply_rigid_transform
from dggt.render.sdf_volume import generate_rays, render_rays_static
from datasets.articulated_dataset import ArticulatedDataset


# ============================================================================
# Loss functions
# ============================================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for mask supervision.
    pred, target: [*, H, W]  (any leading batch dims)
    """
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)


def mask_loss(
    assign_maps: torch.Tensor,   # [B, P, H_p, W_p]   P part slots (slot 0 = base part)
    gt_masks: torch.Tensor,      # [B, P_gt, H, W]    gt part 0 = base part (no bg)
    matches: list,
    bg_map: torch.Tensor = None, # [B, 1, H_p, W_p]   background sink (paper-aligned)
    fg_weight: float = 10.0,
    dice_weight: float = 2.0,
) -> torch.Tensor:
    """
    NLL (foreground-weighted cross-entropy) + Dice loss on matched slot-part pairs.

    NLL pushes P(correct_slot | pixel) up per-pixel but allows diffuse predictions.
    Dice directly optimizes soft-mask overlap ≈ IoU, bridging the gap between NLL
    and the argmax-IoU eval metric.  Combined loss breaks the ~0.67 IoU plateau.

    Background handling: when `bg_map` is given (paper-aligned slot0 = base part),
    a background CLASS (index P) is appended so background pixels — which belong to
    NO part slot — are classified into the sink instead of being forced into slot 0.
    Without bg_map, the legacy behaviour (label 0 = catch-all) is used.
    """
    B, P, H_p, W_p = assign_maps.shape
    H, W = gt_masks.shape[-2:]

    if bg_map is not None:
        # P part channels + background sink → [B, P+1, H, W]; bg label = P.
        maps = torch.cat([assign_maps, bg_map], dim=1)
        bg_label = P
    else:
        maps = assign_maps
        bg_label = None

    pred_up = F.interpolate(
        maps, (H, W), mode="bilinear", align_corners=False
    )   # [B, P(+1), H, W]
    log_pred = torch.log(pred_up.clamp(min=1e-8))

    total = pred_up.new_zeros(1)

    for b, (pred_idx, gt_idx) in enumerate(matches):
        # Build per-pixel label map. Default = background:
        #   - paper-aligned: bg class P (pixels belonging to no part)
        #   - legacy:        slot 0 (catch-all)
        default = bg_label if bg_label is not None else 0
        label_map = torch.full((H, W), default, dtype=torch.long, device=pred_up.device)
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            label_map[gt_masks[b, gi] > 0.5] = pi

        # Per-pixel weight: upweight part (foreground) pixels over background.
        is_fg = (label_map != default) if bg_label is not None else (label_map > 0)
        pixel_weight = torch.where(is_fg,
                                   label_map.new_full((), fg_weight).float(),
                                   label_map.new_ones(()).float())   # [H, W]

        # NLL loss
        nll = F.nll_loss(
            log_pred[b].unsqueeze(0),
            label_map.unsqueeze(0),
            reduction="none",
        ).squeeze(0)
        total = total + (nll * pixel_weight).sum() / pixel_weight.sum()

        # Dice loss per matched (slot, part) pair (part channels only)
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            pred_mask = pred_up[b, pi]            # [H, W] soft
            gt_mask   = gt_masks[b, gi]           # [H, W] binary
            inter = (pred_mask * gt_mask).sum()
            dice  = (2.0 * inter) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
            total = total + dice_weight * (1.0 - dice)

    return total / max(B, 1)


def kinematic_loss(
    motion_type_logits: torch.Tensor,  # [B, P, 2]
    axis: torch.Tensor,                # [B, P, 3]  (sign-fixed, matched)
    pivot: torch.Tensor,               # [B, P, 3]  (matched)
    scalars: torch.Tensor,             # [B, P, S]  (sign-fixed, matched)
    gt_motion_type: torch.Tensor,      # [B, P_gt]  long
    gt_axis: torch.Tensor,             # [B, P_gt, 3]
    gt_pivot: torch.Tensor,            # [B, P_gt, 3]
    gt_scalars: torch.Tensor,          # [B, P_gt, S]
    is_dead: torch.Tensor,             # [B, P]  bool
    matches: list,
    weights: dict,
) -> dict:
    """
    Compute all kinematic supervision losses.

    Returns dict of scalar losses.
    """
    B, P, _ = motion_type_logits.shape
    P_gt = gt_motion_type.shape[1]

    # Reorder predictions to GT ordering
    logits_r  = reorder_by_match(motion_type_logits, matches, P_gt)  # [B, P_gt, 2]
    pivot_r   = reorder_by_match(pivot,              matches, P_gt)  # [B, P_gt, 3]

    # is_dead reordered
    is_dead_r = reorder_by_match(is_dead.float().unsqueeze(-1), matches, P_gt)
    is_dead_r = is_dead_r.squeeze(-1).bool()   # [B, P_gt]

    # Mark unmatched GT positions as dead: reorder_by_match zero-fills them,
    # causing 0-axis vs 0-gt-axis comparisons that inflate the axis loss.
    is_matched = motion_type_logits.new_zeros(B, P_gt, dtype=torch.bool)
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(gt_idx) > 0:
            is_matched[b, gt_idx] = True
    is_dead_r = is_dead_r | (~is_matched)

    # Alive mask (skip dead slots AND Slot 0 which is always static)
    # In GT space, Slot 0 = static so its type loss is trivially correct
    alive = ~is_dead_r                          # [B, P_gt]
    alive[:, 0] = False                         # skip Slot 0 (static; no loss needed)

    losses = {}

    # ── Motion type CE (2-class: prismatic=0, revolute=1) ─────────────────
    # GT labels: 1=prismatic, 2=revolute. Map to 0-indexed: subtract 1.
    # Slot 0 (static) is excluded from alive mask above.
    if alive.any():
        logits_alive  = logits_r[alive]              # [K, 2]
        gt_type_alive = gt_motion_type[alive] - 1    # [K]  map 1→0, 2→1
        gt_type_alive = gt_type_alive.clamp(0, 1)    # safety clamp
        losses["type"] = weights["type"] * F.cross_entropy(logits_alive, gt_type_alive)
    else:
        losses["type"] = logits_r.new_zeros(1).squeeze()

    # ── Axis cosine loss (symmetric, after sign fix) ──────────────────────
    # axis and scalars are already sign-fixed and in GT order
    if alive.any():
        ax_alive = axis[alive]                   # [K, 3]  (axis already reordered)
        gt_ax_alive = gt_axis[alive]             # [K, 3]
        # symmetric: 1 - |cos| = 1 - |dot(pred, gt)|
        cos_sim = (ax_alive * gt_ax_alive).sum(dim=-1).abs()  # [K]
        losses["axis"] = weights["axis"] * (1.0 - cos_sim).mean()
    else:
        losses["axis"] = axis.new_zeros(1).squeeze()

    # ── Pivot L2 loss ────────────────────────────────────────────────────
    if alive.any():
        pv_alive = pivot_r[alive]                # [K, 3]
        gt_pv_alive = gt_pivot[alive]            # [K, 3]
        losses["pivot"] = weights["pivot"] * F.mse_loss(pv_alive, gt_pv_alive)
    else:
        losses["pivot"] = pivot.new_zeros(1).squeeze()

    # ── Scalar Huber loss ────────────────────────────────────────────────
    # Huber loss (delta=1.0) clips the gradient for large errors, preventing
    # the 4× MSE spikes seen when sign-fix is uncertain (|dot| near zero).
    # scalars: [B, P_gt, S] (already reordered by match_and_fix)
    if alive.any():
        sc_alive    = scalars[alive]             # [K, S]
        gt_sc_alive = gt_scalars[alive]          # [K, S]
        losses["scalar"] = weights["scalar"] * F.huber_loss(sc_alive, gt_sc_alive, delta=1.0)
    else:
        losses["scalar"] = scalars.new_zeros(1).squeeze()

    return losses


def bbox_loss(
    bbox_center: torch.Tensor,   # [B, P, 3]   predicted bbox centres
    bbox_size:   torch.Tensor,   # [B, P, 3]   predicted half-extents
    assign_maps: torch.Tensor,   # [B, P, H_p, W_p]   soft assignment
    extrinsics:  torch.Tensor,   # [B, S, 4, 4]   cam-to-world
    intrinsics:  torch.Tensor,   # [B, 3, 3]
    gt_masks:    torch.Tensor,   # [B, S, P, H, W]  per-frame GT part masks
    patch_size:  int = 14,
) -> torch.Tensor:
    """
    Soft bbox supervision via 2D projection.

    For each slot p, we estimate the GT 2D centroid from the GT mask (averaged
    over frames) and compare it to the projected bbox_center in each view.
    This gives a cheap spatial anchor without needing 3D GT bounding boxes.

    Loss = MSE between projected bbox_center and GT mask 2D centroid,
    weighted by GT mask area (so small / absent parts contribute less).

    Returns scalar loss.
    """
    B, P, H_p, W_p = assign_maps.shape
    B2, S, P2, H, W = gt_masks.shape
    device = bbox_center.device

    # Average GT masks over frames → [B, P, H, W]
    gt_avg = gt_masks.float().to(device).mean(dim=1)

    # Compute GT 2D centroid for each slot (weighted by mask)
    # ys, xs: pixel coordinate grids
    ys = torch.arange(H, device=device, dtype=torch.float32) / H  # [H] ∈ [0,1]
    xs = torch.arange(W, device=device, dtype=torch.float32) / W  # [W] ∈ [0,1]
    ys = ys.view(1, 1, H, 1).expand(B, P, H, W)
    xs = xs.view(1, 1, 1, W).expand(B, P, H, W)

    mask_sum = gt_avg.sum(dim=(-2, -1)).clamp(min=1.0)             # [B, P]
    gt_cy = (gt_avg * ys).sum(dim=(-2, -1)) / mask_sum             # [B, P] ∈ [0,1]
    gt_cx = (gt_avg * xs).sum(dim=(-2, -1)) / mask_sum             # [B, P] ∈ [0,1]
    # Map to pixel coordinates
    gt_cx_px = gt_cx * W                                            # [B, P]
    gt_cy_px = gt_cy * H                                            # [B, P]

    # Project bbox_center to image plane (use first frame for simplicity)
    E = extrinsics[:, 0]                   # [B, 4, 4] cam-to-world, first frame
    # World-to-cam = E^{-1} (approximate as E.T for rotation part)
    R = E[:, :3, :3]                       # [B, 3, 3]
    t = E[:, :3, 3]                        # [B, 3]
    # cam_pos = R^T (world_pos - cam_origin) = R^T @ world_pos - R^T @ t
    # bbox_center: [B, P, 3]
    bc = bbox_center                       # [B, P, 3]
    # Transform each slot's bbox_center to camera space
    # R^T: [B, 3, 3]; bc - t: [B, P, 3]
    cam_t = t.unsqueeze(1)                 # [B, 1, 3]
    bc_cam = torch.einsum('bij,bpj->bpi', R.transpose(-1,-2), bc - cam_t)
    # [B, P, 3]  in camera space

    # Project with intrinsics K: [B, 3, 3]
    # OpenGL/Blender cam-to-world convention: camera looks along -Z,
    # so z_cam < 0 for points in FRONT of the camera.
    behind = (bc_cam[..., 2] >= 0.0)                  # [B, P]  z_cam >= 0 → behind camera
    depth = (-bc_cam[..., 2]).clamp(min=0.5)          # [B, P]  positive depth = -z_cam
    K = intrinsics                                     # [B, 3, 3]
    pred_cx = K[:, 0, 0].unsqueeze(1) * bc_cam[..., 0] / depth + K[:, 0, 2].unsqueeze(1)
    pred_cy = K[:, 1, 1].unsqueeze(1) * bc_cam[..., 1] / depth + K[:, 1, 2].unsqueeze(1)

    weight = (mask_sum / (H * W)).clamp(0, 1)          # [B, P] ∈ [0,1]
    weight = weight * (~behind).float()               # zero weight for behind-camera slots

    # Normalise coordinates to [0,1] for scale-invariant MSE
    dx = (pred_cx - gt_cx_px) / W
    dy = (pred_cy - gt_cy_px) / H
    dist2 = (dx ** 2 + dy ** 2).clamp(max=1.0)        # [B, P]  per-slot cap to prevent spike

    loss = (weight * dist2).sum() / (weight.sum().clamp(min=1.0))
    return loss


def per_part_alpha_render_loss(
    gs_mu:           torch.Tensor,   # [B, P, N_g, 3]  canonical positions
    gs_opacity:      torch.Tensor,   # [B, P, N_g, 1]
    motion_type_logits: torch.Tensor, # [B, P, 2]
    axis:            torch.Tensor,   # [B, P, 3]
    pivot:           torch.Tensor,   # [B, P, 3]
    scalars:         torch.Tensor,   # [B, P, S]
    extrinsics:      torch.Tensor,   # [B, S, 4, 4]
    intrinsics:      torch.Tensor,   # [B, 3, 3]
    gt_masks_seq:    torch.Tensor,   # [B, S, P, H, W]  per-frame GT masks
    is_dead:         torch.Tensor,   # [B, P]
    patch_size:      int = 14,
    sigma_patches:   float = 0.8,    # Gaussian kernel sigma in patch units
) -> torch.Tensor:
    """
    Per-part differentiable alpha rendering loss (no gsplat required).

    For each (slot, frame), projects Gaussian centers to image plane and
    accumulates a soft alpha map using a Gaussian kernel. Compares with
    GT part masks at patch resolution.

    Activated in Phase 1b (post-warmup).
    """
    from dggt.utils.rigid_transform import apply_rigid_transform

    B, P, N_g, _ = gs_mu.shape
    B2, S, P2, H, W = gt_masks_seq.shape
    device = gs_mu.device

    H_p = H // patch_size
    W_p = W // patch_size

    # Motion probs: pad 2-class [prismatic, revolute] with leading 0 for static
    # → [B, P, 3] where index 0=static, 1=prismatic, 2=revolute
    motion_probs_2 = torch.softmax(motion_type_logits.float(), dim=-1)  # [B, P, 2]
    static_col = torch.zeros(B, P, 1, device=device, dtype=motion_probs_2.dtype)
    # Slot 0: all weight on static; other slots: weight from 2-class prediction
    motion_probs_3 = torch.cat([static_col, motion_probs_2], dim=-1)    # [B, P, 3]
    motion_probs_3[:, 0, :] = torch.tensor([1.0, 0.0, 0.0], device=device)

    # Precompute patch-center coordinate grid [H_p, W_p, 2] (in patch units)
    gy = torch.arange(H_p, device=device, dtype=torch.float32) + 0.5   # [H_p]
    gx = torch.arange(W_p, device=device, dtype=torch.float32) + 0.5   # [W_p]
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')             # [H_p, W_p]
    grid = torch.stack([grid_x, grid_y], dim=-1)                        # [H_p, W_p, 2]

    total_loss = gs_mu.new_zeros(1).squeeze()
    n_terms = 0

    for t in range(S):
        # Extrinsics for this frame: cam-to-world [B, 4, 4]
        E_c2w = extrinsics[:, t]                          # [B, 4, 4]
        # World-to-cam rotation + translation
        R_c2w = E_c2w[:, :3, :3]                          # [B, 3, 3]
        t_c2w = E_c2w[:, :3, 3]                           # [B, 3]
        R_w2c = R_c2w.transpose(-1, -2)                   # [B, 3, 3]
        t_w2c = -torch.einsum('bij,bj->bi', R_w2c, t_c2w) # [B, 3]

        K = intrinsics                                     # [B, 3, 3]
        scalar_t = scalars[:, :, t]                        # [B, P]

        for p in range(P):
            if is_dead[:, p].all():
                continue

            # Apply rigid transform for this slot, this frame
            mu_p    = gs_mu[:, p]                          # [B, N_g, 3]
            rot_p   = torch.zeros(B, N_g, 4, device=device)  # dummy quats (not used for pos)
            rot_p[:, :, 0] = 1.0                           # identity quaternion
            mp      = motion_probs_3[:, p]                 # [B, 3]
            ax      = axis[:, p]                           # [B, 3]
            pv      = pivot[:, p]                          # [B, 3]
            sc      = scalar_t[:, p]                       # [B]

            # apply_rigid_transform is single-sample; loop over batch
            mu_t = torch.stack([
                apply_rigid_transform(mu_p[b], rot_p[b], mp[b], ax[b], pv[b], sc[b])[0]
                for b in range(B)
            ], dim=0)
            # mu_t: [B, N_g, 3]  world-space positions at time t

            # Project to camera space
            mu_cam = torch.einsum('bij,bnj->bni', R_w2c, mu_t) + t_w2c.unsqueeze(1)
            # [B, N_g, 3]

            # OpenGL convention: z_cam < 0 for front-facing, depth = -z_cam
            behind_g = (mu_cam[..., 2] >= 0.0)            # [B, N_g]
            depth = (-mu_cam[..., 2]).clamp(min=0.01)     # [B, N_g]  positive depth

            # Project to image (patch units)
            fx = K[:, 0, 0].unsqueeze(1) / patch_size     # [B, 1]
            fy = K[:, 1, 1].unsqueeze(1) / patch_size     # [B, 1]
            cx = K[:, 0, 2].unsqueeze(1) / patch_size     # [B, 1]
            cy = K[:, 1, 2].unsqueeze(1) / patch_size     # [B, 1]

            u = fx * mu_cam[..., 0] / depth + cx          # [B, N_g]
            v = fy * mu_cam[..., 1] / depth + cy          # [B, N_g]

            # Compute soft alpha map via Gaussian kernel accumulation
            # grid: [H_p, W_p, 2];  u, v: [B, N_g]
            # dist2: [B, H_p, W_p, N_g]
            u_b = u.unsqueeze(1).unsqueeze(1)              # [B, 1, 1, N_g]
            v_b = v.unsqueeze(1).unsqueeze(1)              # [B, 1, 1, N_g]
            dx  = grid[..., 0].unsqueeze(0).unsqueeze(-1) - u_b  # [B, H_p, W_p, N_g]
            dy  = grid[..., 1].unsqueeze(0).unsqueeze(-1) - v_b
            dist2 = dx * dx + dy * dy                      # [B, H_p, W_p, N_g]

            kernel = torch.exp(-dist2 / (2 * sigma_patches ** 2))  # [B, H_p, W_p, N_g]

            op = gs_opacity[:, p].squeeze(-1)              # [B, N_g]
            op = op * (~behind_g).float()                  # zero opacity for behind-camera Gaussians
            op_b = op.unsqueeze(1).unsqueeze(1)            # [B, 1, 1, N_g]

            alpha_map = (kernel * op_b).sum(dim=-1)        # [B, H_p, W_p]
            alpha_map = alpha_map.clamp(0, 1)

            # GT mask for this slot/frame at patch resolution
            gt_mask_t = gt_masks_seq[:, t, p].float().to(device)  # [B, H, W]
            gt_p = F.adaptive_avg_pool2d(
                gt_mask_t.unsqueeze(1), (H_p, W_p)
            ).squeeze(1)                                   # [B, H_p, W_p]

            # Per-part mask loss (Dice + BCE)
            alpha_flat = alpha_map.reshape(B, -1)
            gt_flat    = gt_p.reshape(B, -1)

            # Dice
            inter = (alpha_flat * gt_flat).sum(dim=-1)
            l_dice = 1.0 - (2 * inter + 1e-6) / (
                alpha_flat.sum(-1) + gt_flat.sum(-1) + 1e-6
            )  # [B]

            # BCE
            logit = torch.logit(alpha_map.float().clamp(1e-6, 1-1e-6))
            l_bce = F.binary_cross_entropy_with_logits(logit, gt_p.float(), reduction='mean')

            alive_b = ~is_dead[:, p]                       # [B]
            if alive_b.any():
                total_loss = total_loss + l_dice[alive_b].mean() + l_bce
                n_terms += 1

    if n_terms == 0:
        return gs_mu.new_zeros(1).squeeze()
    return total_loss / n_terms


def global_render_loss(
    gs_mu:              torch.Tensor,   # [B, P, N_g, 3]  canonical positions
    gs_rot:             torch.Tensor,   # [B, P, N_g, 4]  unit quaternions (w,x,y,z)
    gs_scale:           torch.Tensor,   # [B, P, N_g, 3]  positive scales
    gs_color:           torch.Tensor,   # [B, P, N_g, 3]  RGB ∈ [0, 1]
    gs_opacity:         torch.Tensor,   # [B, P, N_g, 1]  ∈ (0, 1)
    motion_type_logits: torch.Tensor,   # [B, P, 2]
    axis:               torch.Tensor,   # [B, P, 3]
    pivot:              torch.Tensor,   # [B, P, 3]
    scalars:            torch.Tensor,   # [B, P, S]
    extrinsics:         torch.Tensor,   # [B, S, 4, 4]  cam-to-world
    intrinsics:         torch.Tensor,   # [B, 3, 3]
    gt_images:          torch.Tensor,   # [B, S, 3, H, W]  in [0, 1]
    is_dead:            torch.Tensor,   # [B, P]
    max_frames:         int = 2,        # max frames rendered per sample (efficiency)
) -> torch.Tensor:
    """
    Global composited rendering loss using gsplat rasterization.

    For each (batch, frame) pair, applies rigid transforms to ALL part Gaussians,
    concatenates them, renders a full RGB image via gsplat, and computes L1 loss
    against the GT image. Dead slots have their opacities zeroed out.

    Activated in Phase 1b (post-warmup) alongside per-part alpha loss.
    """
    from gsplat.rendering import rasterization as gsplat_rasterize

    B, P, N_g, _ = gs_mu.shape
    S = scalars.shape[-1]
    H, W = gt_images.shape[-2:]
    device = gs_mu.device

    # Motion probs [B, P, 3]: static, prismatic, revolute
    motion_probs_2 = torch.softmax(motion_type_logits.float(), dim=-1)  # [B, P, 2]
    static_col = torch.zeros(B, P, 1, device=device, dtype=motion_probs_2.dtype)
    motion_probs_3 = torch.cat([static_col, motion_probs_2], dim=-1)    # [B, P, 3]
    motion_probs_3[:, 0, :] = torch.tensor([1.0, 0.0, 0.0], device=device)

    # Alive mask: zero opacities for dead slots
    slot_alive = (~is_dead).float()   # [B, P]

    total_loss = gs_mu.new_zeros(1).squeeze()
    n_terms = 0

    for b in range(B):
        # Randomly sample frames to render (capped at max_frames for efficiency)
        frame_indices = list(range(S))
        if len(frame_indices) > max_frames:
            frame_indices = random.sample(frame_indices, max_frames)

        for t in frame_indices:
            scalar_t = scalars[b, :, t]   # [P]

            # Build world-to-cam from cam-to-world extrinsics
            c2w = extrinsics[b, t].float()   # [4, 4]
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3:4]
            R_w2c = R_c2w.T                           # [3, 3]
            t_w2c = -R_w2c @ t_c2w                    # [3, 1]
            w2c = torch.eye(4, device=device, dtype=torch.float32)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = t_w2c.squeeze(-1)
            viewmat = w2c.unsqueeze(0)                # [1, 4, 4]
            K = intrinsics[b].float().unsqueeze(0)    # [1, 3, 3]

            all_means, all_quats, all_scales, all_colors, all_opacities = [], [], [], [], []

            for p in range(P):
                mu_p  = gs_mu[b, p].float()           # [N_g, 3]
                rot_p = gs_rot[b, p].float()           # [N_g, 4]
                sc_p  = gs_scale[b, p].float()         # [N_g, 3]
                col_p = gs_color[b, p].float()         # [N_g, 3]
                op_p  = gs_opacity[b, p, :, 0].float() # [N_g]

                mp = motion_probs_3[b, p]              # [3]
                ax = axis[b, p]                        # [3]
                pv = pivot[b, p]                       # [3]
                sc = scalar_t[p]                       # []

                mu_t, rot_t = apply_rigid_transform(mu_p, rot_p, mp, ax, pv, sc)

                # Zero out opacity for dead slots (no gradient, just masking)
                op_t = op_p * slot_alive[b, p]

                all_means.append(mu_t)
                all_quats.append(rot_t)
                all_scales.append(sc_p)
                all_colors.append(col_p)
                all_opacities.append(op_t)

            means     = torch.cat(all_means,     dim=0)  # [P*N_g, 3]
            quats     = torch.cat(all_quats,     dim=0)  # [P*N_g, 4]
            scales    = torch.cat(all_scales,    dim=0)  # [P*N_g, 3]
            colors    = torch.cat(all_colors,    dim=0)  # [P*N_g, 3]
            opacities = torch.cat(all_opacities, dim=0)  # [P*N_g]

            try:
                render_out, _, _ = gsplat_rasterize(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmat,
                    Ks=K,
                    width=W,
                    height=H,
                    near_plane=0.01,
                    far_plane=1e4,
                    sh_degree=None,
                )
                # render_out: [1, H, W, 3] → [3, H, W]
                rendered = render_out[0].permute(2, 0, 1).clamp(0.0, 1.0)
                gt_img   = gt_images[b, t].float().to(device)   # [3, H, W]

                total_loss = total_loss + F.l1_loss(rendered, gt_img)
                n_terms += 1
            except Exception as _e:
                # Log once per process to diagnose why gsplat is failing
                import sys, traceback
                print(f"[global_render_loss] gsplat failed b={b} t={t}: {_e}", flush=True)
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                break  # only log once per forward pass

    if n_terms == 0:
        return gs_mu.new_zeros(1).squeeze()
    return total_loss / n_terms


def motion_aux_loss(
    assign_maps: torch.Tensor,       # [B, P, H_p, W_p]  softmax
    motion_mask: torch.Tensor,       # [B, P, H_p, W_p]  soft pseudo-label (sum_P=1)
    tracks_2d: torch.Tensor,         # [B, S, N, 2]      pixel coords at (H, W)
    tracks_vis: torch.Tensor,        # [B, S, N]
    track_part_label: torch.Tensor,  # [B, N, P]         soft pseudo-label (sum_P=1)
    has_motion: torch.Tensor,        # [B]  bool
    img_hw: tuple[int, int],
    w_mask: float,
    w_track: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Auxiliary motion supervision from precomputed pseudo-labels.

    Hungarian-matches pred slots → pseudo slots (slot 0 forced static in
    precompute). Then:
      L_mask:  soft-CE(assign_maps, motion_mask_aligned)
      L_track: soft-CE(sample(assign_maps, tracks_f0), track_part_label_aligned)
    Samples without motion data contribute 0.
    """
    B, P, H_p, W_p = assign_maps.shape
    H, W = img_hw
    device = assign_maps.device
    z = assign_maps.new_zeros(())

    if not bool(has_motion.any()):
        return z, z

    l_mask_sum = z.clone()
    l_trk_sum  = z.clone()
    n_mask = 0
    n_trk  = 0

    for b in range(B):
        if not bool(has_motion[b]):
            continue
        am_b = assign_maps[b]     # [P, H_p, W_p]
        mm_b = motion_mask[b]     # [P, H_p, W_p]

        # Hungarian alignment (argmax IoU, slots 0..P-1 both sides)
        with torch.no_grad():
            am_arg = am_b.argmax(0)
            mm_arg = mm_b.argmax(0)
            cost = torch.zeros(P, P, device=device)
            for i in range(P):
                ai = (am_arg == i)
                for j in range(P):
                    mj = (mm_arg == j)
                    inter = (ai & mj).sum().float()
                    union = (ai | mj).sum().float()
                    cost[i, j] = 1.0 - inter / (union + 1e-6)
            row, col = linear_sum_assignment(cost.cpu().numpy())
        # perm[pred_slot_i] = pseudo_slot_j matched to it
        perm = torch.as_tensor(col, dtype=torch.long, device=device)[
            torch.argsort(torch.as_tensor(row, dtype=torch.long, device=device))
        ]

        # L_mask: align pseudo channels to pred channels then soft-CE
        mm_aligned = mm_b[perm]                              # [P, H_p, W_p]
        log_am = torch.log(am_b.clamp(min=1e-8))
        l_mask_sum = l_mask_sum + -(mm_aligned * log_am).sum(0).mean()
        n_mask += 1

        # L_track: sample pred at frame-0 track positions
        vis0 = tracks_vis[b, 0]                              # [N]
        if float(vis0.sum()) < 1.0:
            continue
        pts = tracks_2d[b, 0]                                # [N, 2] pixel
        gx = pts[:, 0] / max(W - 1, 1) * 2.0 - 1.0
        gy = pts[:, 1] / max(H - 1, 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(
            am_b.unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(0).squeeze(1).transpose(0, 1)              # [N, P]
        tpl_aligned = track_part_label[b][:, perm]           # [N, P]
        log_s = torch.log(sampled.clamp(min=1e-8))
        per_trk = -(tpl_aligned * log_s).sum(-1)             # [N]
        w = vis0
        l_trk_sum = l_trk_sum + (per_trk * w).sum() / (w.sum() + 1e-6)
        n_trk += 1

    l_mask = l_mask_sum / max(n_mask, 1)
    l_trk  = l_trk_sum  / max(n_trk, 1)
    return w_mask * l_mask, w_track * l_trk


# ============================================================================
# Training loop
# ============================================================================

def sdf_render_loss(
    head,                              # HexaPlaneSDFHead (.query)
    planes: torch.Tensor,             # [B, P, 6, Cf, R, R]
    bbox_center: torch.Tensor,        # [B, P, 3]
    bbox_size: torch.Tensor,          # [B, P, 3]
    is_dead: torch.Tensor,            # [B, P] bool (pred-slot order)
    matches: list,                    # per-batch (pred_idx, gt_idx)
    extrinsics: torch.Tensor,         # [B, S, 4, 4] cam-to-world
    intrinsics: torch.Tensor,         # [B, 3, 3]
    gt_images: torch.Tensor,          # [B, S, 3, H, W] in [0,1]
    gt_masks_seq: torch.Tensor,       # [B, S, P_gt, H, W]  (slot 0 = base part, no bg)
    motion_type_logits: torch.Tensor, # [B, P, 2]
    axis: torch.Tensor,               # [B, P, 3]
    pivot: torch.Tensor,              # [B, P, 3]
    scalars: torch.Tensor,            # [B, P, S]
    gt_depth: torch.Tensor = None,    # [B, S, H, W]  camera z-depth (0 = background)
    scene_radius: float = 1.0,
    beta: float = 0.1,
    n_rays: int = 1024,
    n_samples: int = 48,
    max_frames: int = 4,
    w_rgb: float = 1.0,
    w_sil: float = 1.0,
    w_part: float = 1.0,
    w_depth: float = 1.0,
) -> torch.Tensor:
    """
    SDF volume rendering loss over multiple articulated frames.

    The STATIC base part (pred slot 0 ↔ GT part 0, paper-aligned pure static body)
    is rendered alongside the movable parts, so the largest/most-solid part — the
    strongest geometric signal — is supervised. For each selected frame s, movable
    parts are moved to their stage-s pose by inverse-transforming the rays
    (ray_transform; the base part is static, no transform), then rendered + supervised:
      • per-part opacity (silhouette)  vs per-part GT mask_s  (BCE), incl. base
      • union opacity                  vs union of ALL rendered masks_s (BCE)
      • composited RGB                 vs GT image_s, on object foreground (L1)
      • expected ray distance          vs GT depth_s, on object foreground (L1)
    Background belongs to no part (router bg-sink); it is simply absent from the
    union. Gradients flow to planes/bbox AND axis/pivot/scalar/motion-type.
    """
    B, P = planes.shape[:2]
    S = scalars.shape[-1]
    H, W = gt_images.shape[-2:]
    device = planes.device
    total = planes.new_zeros(())
    n_terms = 0

    # Frames to render: spread across the sequence (always include rest frame 0).
    n_f = min(max_frames, S)
    frame_ids = torch.linspace(0, S - 1, n_f).round().long().tolist()

    with torch.amp.autocast("cuda", enabled=False):
        planes_f = planes.float()
        center_f = bbox_center.float()
        size_f   = bbox_size.float()
        axis_f   = axis.float()
        pivot_f  = pivot.float()
        scal_f   = scalars.float()
        # motion probs [B,P,3] = [static, prismatic, revolute]; static=0 for movable
        mp2 = torch.softmax(motion_type_logits.float(), dim=-1)         # [B,P,2]
        motion_probs = torch.cat([mp2.new_zeros(B, P, 1), mp2], dim=-1)  # [B,P,3]
        # Force pred slot 0 (the base part) fully static so its rays are NOT
        # inverse-transformed (slot-0 type logits are zeroed → mp2≈uniform otherwise).
        motion_probs[:, 0] = motion_probs.new_tensor([1.0, 0.0, 0.0])

        for b in range(B):
            # pred-slot → GT part map (shared across frames).
            # Base part: pred slot 0 ↔ GT part 0 (fixed by design, not in `matches`).
            alive = torch.zeros(P, dtype=torch.bool, device=device)
            pred2gt = {}
            if not bool(is_dead[b, 0]) and (gt_masks_seq[b, :, 0] > 0.5).any():
                alive[0] = True
                pred2gt[0] = 0
            for pi, gi in zip(matches[b][0].tolist(), matches[b][1].tolist()):
                if gi >= 1 and not bool(is_dead[b, pi]):
                    alive[pi] = True
                    pred2gt[pi] = gi
            if not bool(alive.any()):
                continue
            rendered_gt = [pred2gt[pi] for pi in range(P) if alive[pi]]

            for s in frame_ids:
                masks = (gt_masks_seq[b, s] > 0.5).float()       # [P_gt, H, W]
                union = masks[rendered_gt].sum(0).clamp(0, 1)    # [H, W] full object fg

                # Sample pixels: half foreground, half random
                fg_idx = torch.nonzero(union.reshape(-1) > 0.5, as_tuple=False).squeeze(-1)
                n_fg = min(n_rays // 2, fg_idx.numel())
                sel = []
                if n_fg > 0:
                    sel.append(fg_idx[torch.randint(fg_idx.numel(), (n_fg,), device=device)])
                sel.append(torch.randint(H * W, (n_rays - n_fg,), device=device))
                pix_flat = torch.cat(sel)
                pix_xy = torch.stack([(pix_flat % W).float(), (pix_flat // W).float()], dim=-1)

                ro, rd = generate_rays(extrinsics[b, s].float(), intrinsics[b].float(), pix_xy)
                out = render_rays_static(
                    head, planes_f[b], center_f[b], size_f[b], alive, ro, rd,
                    beta=beta, n_samples=n_samples,
                    motion_probs=motion_probs[b], axis=axis_f[b], pivot=pivot_f[b],
                    scalar=scal_f[b, :, s], scene_radius=scene_radius,
                )

                gt_rgb = gt_images[b, s].float().reshape(3, -1)[:, pix_flat].transpose(0, 1)
                gt_un  = union.reshape(-1)[pix_flat]
                fg_mask = gt_un > 0.5

                opacity = out["opacity"][:, 0].clamp(1e-5, 1 - 1e-5)
                l_sil = F.binary_cross_entropy(opacity, gt_un)
                l_part = planes.new_zeros(())
                for pi in range(P):
                    if not bool(alive[pi]):
                        continue
                    gt_p = masks[pred2gt[pi]].reshape(-1)[pix_flat]
                    po = out["part_opacity"][:, pi].clamp(1e-5, 1 - 1e-5)
                    l_part = l_part + F.binary_cross_entropy(po, gt_p)
                l_part = l_part / max(len(rendered_gt), 1)

                l_rgb = (out["rgb"][fg_mask] - gt_rgb[fg_mask]).abs().mean() \
                    if fg_mask.any() else planes.new_zeros(())

                # Depth: expected ray distance vs GT camera-z depth → ray distance.
                l_depth = planes.new_zeros(())
                if gt_depth is not None and w_depth > 0:
                    K = intrinsics[b].float()
                    u = pix_xy[:, 0]; v = pix_xy[:, 1]
                    raylen = torch.sqrt(((u - K[0, 2]) / K[0, 0]) ** 2
                                        + ((v - K[1, 2]) / K[1, 1]) ** 2 + 1.0)
                    t_gt = gt_depth[b, s].float().reshape(-1)[pix_flat] * raylen
                    dm = fg_mask & (t_gt > 0)
                    if dm.any():
                        l_depth = (out["depth"][:, 0][dm] - t_gt[dm]).abs().mean()

                total = total + (w_sil * l_sil + w_part * l_part
                                 + w_rgb * l_rgb + w_depth * l_depth)
                n_terms += 1

    return total / max(n_terms, 1)


def compute_loss(
    preds: dict,
    batch: dict,
    step: int,
    cfg: argparse.Namespace,
    is_warmup: bool,
    head=None,
) -> tuple[torch.Tensor, dict]:
    """
    Unified loss computation (all phases).

    Returns (total_loss, loss_dict).
    """
    B = preds["assign_maps"].shape[0]
    H, W = batch["images"].shape[-2:]
    device = preds["assign_maps"].device

    # Use frame 0 GT masks (canonical rest state) — consistent with assign_maps
    # which now uses frame 0 cross-attn weights only.  Frame 0 is always the
    # rest pose, giving unambiguous single-position part masks.
    gt_masks = (batch["part_masks"][:, 0] > 0.5).float().to(device)
    # [B, max_parts, H, W] — index 0 = background; 1..n_joints = moving parts;
    # n_joints+1..7 = zero-padded (objects with fewer joints than max_parts)
    gt_motion_type = batch["gt_motion_type"].to(device)   # [B, P]
    gt_axis        = batch["gt_axis"].to(device)           # [B, P, 3]
    gt_pivot       = batch["gt_pivot"].to(device)          # [B, P, 3]
    gt_scalars     = batch["gt_scalars"].to(device)        # [B, P, S]

    assign_maps = preds["assign_maps"]  # [B, P, H_p, W_p]
    P_gt = gt_masks.shape[1]

    # ── Dead-slot detection ────────────────────────────────────────────────
    is_dead = detect_dead_slots(assign_maps)   # [B, P]

    # ── Hungarian matching + sign fix ─────────────────────────────────────
    match_result = match_and_fix(
        pred_maps   = assign_maps,
        gt_masks    = gt_masks,
        pred_axis   = preds["axis"],
        pred_scalar = preds["scalars"],
        gt_axis     = gt_axis,
    )
    matches      = match_result["matches"]
    axis_fixed   = match_result["pred_axis"]    # [B, P_gt, 3]
    scalar_fixed = match_result["pred_scalar"]  # [B, P_gt, S]

    loss_dict = {}

    # ── Resolve extrinsics for rendering losses ───────────────────────────
    # Prefer GT extrinsics when available (Phase 1a/1b); fall back to
    # CameraHead prediction (Phase 2 real data without GT poses).
    # This prevents unstable early CameraHead predictions from slowing
    # down GaussianHead convergence during Phase 1b supervised training.
    if "extrinsics" in batch:
        render_extrinsics = batch["extrinsics"].to(device)
    elif "predicted_extrinsics" in preds:
        render_extrinsics = preds["predicted_extrinsics"].detach()
    else:
        render_extrinsics = None

    # ── Mask loss (always active) ─────────────────────────────────────────
    # Pass bg_map so background pixels classify into the sink, keeping slot 0 a
    # pure base part (paper-aligned). gt_masks[:,0] must be the base-part mask.
    l_mask = mask_loss(assign_maps, gt_masks, matches, bg_map=preds.get("bg_map"))
    loss_dict["mask"] = l_mask

    # ── Slot sparsity (always active, weight ramps up after warmup) ───────
    sparsity_w = cfg.l1_sparsity_warmup if is_warmup else cfg.l1_sparsity
    l_sparse = slot_sparsity_loss(assign_maps, sparsity_w)
    loss_dict["sparsity"] = l_sparse

    # ── Auxiliary motion loss (precomputed pseudo-labels) ─────────────────
    w_mm_max = getattr(cfg, "w_motion_mask", 0.0)
    w_mt_max = getattr(cfg, "w_motion_track", 0.0)
    if (w_mm_max > 0.0 or w_mt_max > 0.0) and "has_motion_data" in batch:
        has_motion = batch["has_motion_data"]
        if not torch.is_tensor(has_motion):
            has_motion = torch.tensor(has_motion, dtype=torch.bool)
        has_motion = has_motion.to(device)
        ramp = min(1.0, step / max(getattr(cfg, "motion_warmup_steps", 1), 1))
        l_mm, l_mt = motion_aux_loss(
            assign_maps      = assign_maps,
            motion_mask      = batch["motion_mask"].to(device),
            tracks_2d        = batch["tracks_2d"].to(device),
            tracks_vis       = batch["tracks_vis"].to(device),
            track_part_label = batch["track_part_label"].to(device),
            has_motion       = has_motion,
            img_hw           = (H, W),
            w_mask           = ramp * w_mm_max,
            w_track          = ramp * w_mt_max,
        )
    else:
        l_mm = assign_maps.new_zeros(()).squeeze()
        l_mt = assign_maps.new_zeros(()).squeeze()
    loss_dict["motion_mask"]  = l_mm
    loss_dict["motion_track"] = l_mt

    if is_warmup:
        # Warmup: only mask + sparsity losses (+ motion aux if enabled)
        total = l_mask + l_sparse + l_mm + l_mt
        loss_dict["total"] = total
        return total, loss_dict

    # ── Kinematic losses ──────────────────────────────────────────────────
    kin_losses = kinematic_loss(
        motion_type_logits = preds["motion_type_logits"],
        axis               = axis_fixed,
        pivot              = preds["pivot"],
        scalars            = scalar_fixed,
        gt_motion_type     = gt_motion_type,
        gt_axis            = gt_axis,
        gt_pivot           = gt_pivot,
        gt_scalars         = gt_scalars,
        is_dead            = is_dead,
        matches            = matches,
        weights            = {
            "type":   cfg.w_type,
            "axis":   cfg.w_axis,
            "pivot":  cfg.w_pivot,
            "scalar": cfg.w_scalar,
        },
    )
    loss_dict.update(kin_losses)

    # ── Dead-slot opacity penalty (GS-specific; N/A for SDF) ─────────────
    l_dead_op = assign_maps.new_zeros(()).squeeze()
    loss_dict["dead_opacity"] = l_dead_op

    # ── Reorder bbox to GT-slot order for the slot-index-direct bbox loss ─
    def _to_gt_order(t: torch.Tensor) -> torch.Tensor:
        out = reorder_by_match(t, matches, P_gt)
        out[:, 0] = t[:, 0]
        return out

    bbox_center_gt = _to_gt_order(preds["bbox_center"])        # [B, P_gt, 3]
    bbox_size_gt   = _to_gt_order(preds["bbox_size"])          # [B, P_gt, 3]

    # ── SDF volume rendering loss (replaces GS per-part + gsplat render) ──
    if cfg.w_render > 0.0 and render_extrinsics is not None and head is not None:
        l_render = cfg.w_render * sdf_render_loss(
            head         = head,
            planes       = preds["planes"],
            bbox_center  = preds["bbox_center"],
            bbox_size    = preds["bbox_size"],
            is_dead      = is_dead,
            matches      = matches,
            extrinsics   = render_extrinsics,
            intrinsics   = batch["intrinsics"].to(device),
            gt_images    = batch["images"].to(device),
            gt_masks_seq = batch["part_masks"].to(device),
            motion_type_logits = preds["motion_type_logits"],
            axis         = preds["axis"],
            pivot        = preds["pivot"],
            scalars      = preds["scalars"],
            gt_depth     = batch["depth"].to(device) if "depth" in batch else None,
            scene_radius = cfg.scene_radius,
            beta         = getattr(cfg, "sdf_beta", 0.1),
            n_rays       = getattr(cfg, "sdf_rays", 1024),
            max_frames   = getattr(cfg, "sdf_frames", 4),
            w_depth      = getattr(cfg, "w_depth", 1.0),
        )
    else:
        l_render = assign_maps.new_zeros(()).squeeze()
    loss_dict["render"] = l_render

    l_render_global = assign_maps.new_zeros(()).squeeze()
    loss_dict["render_global"] = l_render_global

    # ── BBox centroid projection loss (slot-index-direct, GT-ordered) ────
    if render_extrinsics is not None:
        l_bbox = cfg.w_bbox * bbox_loss(
            bbox_center = bbox_center_gt,
            bbox_size   = bbox_size_gt,
            assign_maps = assign_maps,
            extrinsics  = render_extrinsics,
            intrinsics  = batch["intrinsics"].to(device),
            gt_masks    = batch["part_masks"].to(device),
            patch_size  = 14,
        )
    else:
        l_bbox = preds["gs_mu"].new_zeros(1).squeeze()
    loss_dict["bbox"] = l_bbox

    # Phase 2 pseudo-mask loss (weak supervision from SAM2)
    if cfg.phase == "2" and batch.get("has_pseudo_masks", False):
        # Average pseudo-masks over frames → [B, P, H, W]
        pseudo_masks = batch["pseudo_masks"].mean(dim=1).to(device)
        l_pseudo = cfg.w_pseudo_mask * mask_loss(assign_maps, pseudo_masks, matches)
        loss_dict["pseudo_mask"] = l_pseudo
    else:
        l_pseudo = assign_maps.new_zeros(1).squeeze()

    # ── Camera pose encoding loss ─────────────────────────────────────────
    # Supervise CameraHead using GT extrinsics so it is trained during Phase 1b.
    # Without this, camera_head parameters receive no gradient and remain at
    # their VGGT-pretrained init — causing cold-start failure when Phase 2
    # switches to predicted (not GT) extrinsics.
    #
    # GT extrinsics in the dataset are cam-to-world [B, S, 4, 4].
    # extri_intri_to_pose_encoding expects world-to-cam [B, S, 3, 4], OpenCV.
    w_pose = getattr(cfg, "w_pose_enc", 0.0)
    if w_pose > 0.0 and "pose_enc" in preds and batch.get("has_pose", True):
        from dggt.utils.pose_enc import extri_intri_to_pose_encoding
        gt_c2w = batch["extrinsics"].to(device).float()  # [B, S, 4, 4]
        intr   = batch["intrinsics"].to(device).float()  # [B, 3, 3]
        # Invert cam-to-world → world-to-cam [B, S, 4, 4]
        R_c2w = gt_c2w[:, :, :3, :3]           # [B, S, 3, 3]
        t_c2w = gt_c2w[:, :, :3, 3:4]          # [B, S, 3, 1]
        R_w2c = R_c2w.transpose(-1, -2)         # [B, S, 3, 3]
        t_w2c = -R_w2c @ t_c2w                 # [B, S, 3, 1]
        w2c_34 = torch.cat([R_w2c, t_w2c], dim=-1)  # [B, S, 3, 4]
        # Expand intrinsics to [B, S, 3, 3]
        S_frames = gt_c2w.shape[1]
        intr_bs = intr.unsqueeze(1).expand(-1, S_frames, -1, -1)
        gt_pose_enc = extri_intri_to_pose_encoding(
            w2c_34, intr_bs, image_size_hw=(H, W)
        )  # [B, S, 9]
        l_pose = w_pose * F.mse_loss(preds["pose_enc"].float(), gt_pose_enc.detach())
        loss_dict["pose_enc"] = l_pose
    else:
        l_pose = assign_maps.new_zeros(1).squeeze()

    total = (l_mask + l_sparse
             + kin_losses["type"] + kin_losses["axis"]
             + kin_losses["pivot"] + kin_losses["scalar"]
             + l_dead_op + l_render + l_render_global + l_bbox + l_pseudo + l_pose
             + l_mm + l_mt)
    loss_dict["total"] = total
    return total, loss_dict


# ============================================================================
# Validation IoU (for warmup transition)
# ============================================================================

@torch.no_grad()
def eval_mean_iou(model, val_loader, device, cfg) -> float:
    """Compute mean mask IoU over validation set.

    Phase 1a: assign_maps vs frame-0 GT (canonical rest state).
              Dead/empty slots are excluded.
    Phase 1b: per-frame alpha-projection IoU — full pipeline
              (assign_maps → GS → ArticulationHead(scalar_t) → project → mask_t)
              compared against each frame's GT mask.
              Dead slots and frames where GT is empty are excluded.

    Hungarian matching aligns predicted slots to GT parts in both phases.
    """
    from dggt.utils.hungarian_matching import batch_hungarian_match
    model.eval()
    iou_sum, count = 0.0, 0
    is_phase1b = (cfg.phase != "1a")

    try:
        with torch.no_grad():
            for batch in val_loader:
                images     = batch["images"].to(device)
                extrinsics = batch["extrinsics"].to(device)
                intrinsics = batch["intrinsics"].to(device)
                timestamps = batch["timestamps"].to(device)
                part_masks_raw = batch["part_masks"].to(device)   # [B, S, P_gt, H, W]
                B, S, P_gt, H, W = part_masks_raw.shape

                preds = model(images, extrinsics, intrinsics, timestamps)
                assign_maps = preds["assign_maps"]                # [B, P, H_p, W_p]
                _, P, H_p, W_p = assign_maps.shape

                is_dead = detect_dead_slots(assign_maps)          # [B, P]

                # Hungarian matching on frame-0 assign_maps vs frame-0 GT
                gt_masks_f0 = (part_masks_raw[:, 0] > 0.5).float()   # [B, P_gt, H, W]
                pred_up_f0  = F.interpolate(assign_maps, (H, W),
                                            mode="bilinear", align_corners=False)
                matches = batch_hungarian_match(pred_up_f0, gt_masks_f0)

                if not is_phase1b:
                    # ── Phase 1a: assign_maps vs frame-0 GT ─────────────────
                    # Include the background sink in the argmax (if present) so
                    # background pixels resolve to bg (class P), not a part slot.
                    bg_map = preds.get("bg_map")
                    if bg_map is not None:
                        bg_up = F.interpolate(bg_map, (H, W), mode="bilinear",
                                              align_corners=False)
                        argmax_src = torch.cat([pred_up_f0, bg_up], dim=1)  # [B, P+1, H, W]
                    else:
                        argmax_src = pred_up_f0
                    pred_argmax = argmax_src.argmax(dim=1)        # [B, H, W]
                    arange_p = torch.arange(P, device=device).view(1, P, 1, 1)
                    pred_bin = (arange_p == pred_argmax.unsqueeze(1)).float()  # [B, P, H, W]

                    for b in range(B):
                        pred_idx, gt_idx = matches[b]
                        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
                            if is_dead[b, pi]:
                                continue
                            gt_m = gt_masks_f0[b, gi]
                            if gt_m.sum() < 1:
                                continue
                            inter = (pred_bin[b, pi] * gt_m).sum()
                            union = (pred_bin[b, pi] + gt_m).clamp(0, 1).sum()
                            iou_sum += (inter / (union + 1e-6)).item()
                            count += 1

                else:
                    # ── Phase 1b: per-frame alpha-projection IoU ─────────────
                    # For each matched (slot pi → GT part gi), project slot pi's
                    # Gaussians to frame t's camera and compare alpha_map with
                    # the GT part mask at frame t.  Averages over all S frames
                    # and all matched pairs that have a non-empty GT for that frame.
                    gs_mu          = preds["gs_mu"]               # [B, P, N_g, 3]
                    gs_opacity     = preds["gs_opacity"]          # [B, P, N_g, 1]
                    axis           = preds["axis"]                # [B, P, 3]
                    pivot          = preds["pivot"]               # [B, P, 3]
                    scalars        = preds["scalars"]             # [B, P, S]
                    motion_logits  = preds["motion_type_logits"]  # [B, P, 2]

                    patch_size    = 14
                    sigma_patches = 0.8

                    # Motion probs [B, P, 3]: [static, prismatic, revolute]
                    mp2 = torch.softmax(motion_logits.float(), dim=-1)       # [B, P, 2]
                    static_col = torch.zeros(B, P, 1, device=device, dtype=mp2.dtype)
                    mp3 = torch.cat([static_col, mp2], dim=-1)               # [B, P, 3]
                    mp3[:, 0] = torch.tensor([1., 0., 0.], device=device)    # slot 0 always static

                    # Patch-centre grid [H_p, W_p, 2]
                    gy = torch.arange(H_p, device=device, dtype=torch.float32) + 0.5
                    gx = torch.arange(W_p, device=device, dtype=torch.float32) + 0.5
                    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
                    grid = torch.stack([grid_x, grid_y], dim=-1)             # [H_p, W_p, 2]

                    for t in range(S):
                        E_c2w  = extrinsics[:, t]                 # [B, 4, 4]
                        R_w2c  = E_c2w[:, :3, :3].transpose(-1, -2)
                        t_w2c  = -torch.einsum('bij,bj->bi', R_w2c, E_c2w[:, :3, 3])
                        K      = intrinsics                       # [B, 3, 3]
                        sc_t   = scalars[:, :, t]                 # [B, P]

                        for b in range(B):
                            pred_idx, gt_idx = matches[b]
                            for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
                                if is_dead[b, pi]:
                                    continue
                                gt_bin = (part_masks_raw[b, t, gi] > 0.5).float()
                                if gt_bin.sum() < 1:
                                    continue

                                # Project slot pi's Gaussians at frame t
                                mu_p  = gs_mu[b, pi]              # [N_g, 3]
                                N_g   = mu_p.shape[0]
                                rot_p = torch.zeros(N_g, 4, device=device)
                                rot_p[:, 0] = 1.0                 # identity quats

                                mu_t, _ = apply_rigid_transform(
                                    mu_p, rot_p,
                                    mp3[b, pi], axis[b, pi], pivot[b, pi], sc_t[b, pi],
                                )                                 # [N_g, 3] world coords

                                # World → camera
                                mu_cam = (R_w2c[b] @ mu_t.T + t_w2c[b].unsqueeze(-1)).T
                                behind = mu_cam[:, 2] >= 0.0
                                depth  = (-mu_cam[:, 2]).clamp(min=0.01)

                                fx = K[b, 0, 0] / patch_size
                                fy = K[b, 1, 1] / patch_size
                                cx_k = K[b, 0, 2] / patch_size
                                cy_k = K[b, 1, 2] / patch_size
                                u = fx * mu_cam[:, 0] / depth + cx_k   # [N_g]
                                v = fy * mu_cam[:, 1] / depth + cy_k

                                # Soft alpha map [H_p, W_p] via Gaussian kernels
                                dx    = grid[..., 0].unsqueeze(-1) - u.view(1, 1, -1)
                                dy    = grid[..., 1].unsqueeze(-1) - v.view(1, 1, -1)
                                kern  = torch.exp(-(dx*dx + dy*dy) / (2 * sigma_patches**2))
                                op    = gs_opacity[b, pi].squeeze(-1) * (~behind).float()
                                alpha = (kern * op.view(1, 1, -1)).sum(-1).clamp(0, 1)

                                # Upsample to image resolution → binary IoU
                                alpha_up = F.interpolate(
                                    alpha.unsqueeze(0).unsqueeze(0),
                                    (H, W), mode='bilinear', align_corners=False,
                                ).squeeze()
                                pred_bin = (alpha_up > 0.5).float()

                                inter = (pred_bin * gt_bin).sum()
                                union = (pred_bin + gt_bin).clamp(0, 1).sum()
                                iou_sum += (inter / (union + 1e-6)).item()
                                count   += 1
    finally:
        model.train()

    return iou_sum / max(count, 1)


# ============================================================================
# Main
# ============================================================================

def main(cfg: argparse.Namespace):
    # ── Distributed setup ─────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    if is_dist:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ArtVGGT(
        img_size=cfg.img_size,
        patch_size=14,
        embed_dim=1024,
        num_slots=8,
        n_gaussians=cfg.n_gaussians,
        scene_radius=cfg.scene_radius,
        use_camera_head=False,   # always use GT camera params (no pose prediction)
        stop_gradient_plucker=(cfg.phase == "2"),
        gradient_checkpointing=getattr(cfg, "gradient_checkpointing", False),
    ).to(device)

    # Set initial phase and warmup state
    is_warmup = (cfg.phase == "1a")
    model.set_phase(cfg.phase, warmup=is_warmup)

    _resume_ckpt = None
    if cfg.resume:
        _resume_ckpt = torch.load(cfg.resume, map_location=device)
        if getattr(cfg, "partial_resume", False):
            # Load only aggregator + camera_head; leave other modules at random init.
            # Used when PartSlotRouter architecture changed between runs.
            saved = _resume_ckpt["model"]
            partial = {k: v for k, v in saved.items()
                       if k.startswith("aggregator.") or k.startswith("camera_head.")}
            model.load_state_dict(partial, strict=False)
            start_step = 0          # fresh step count; optimizer/scheduler not restored
            _resume_ckpt = None     # skip optimizer/scheduler restore below
            if is_main:
                print(f"Partial resume (aggregator only) from {cfg.resume}")
        else:
            model.load_state_dict(_resume_ckpt["model"], strict=False)
            start_step = _resume_ckpt.get("step", 0)
            if is_main:
                print(f"Resumed from {cfg.resume} at step {start_step}")
    else:
        start_step = 0

    if getattr(cfg, "reset_slot_tokens", False):
        with torch.no_grad():
            nn.init.trunc_normal_(model.part_slot_router.slot_tokens, std=0.02)
        if is_main:
            print("Slot tokens re-initialized (reset_slot_tokens=True)")

    if is_dist:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True,
                    gradient_as_bucket_view=True,
                    bucket_cap_mb=25)

    raw_model = model.module if is_dist else model

    # ── Datasets ───────────────────────────────────────────────────────────
    ds_phase = "1" if cfg.phase in ("1a", "1b") else "2"

    # Phase 2: use real data if provided, else fall back to synthetic
    train_data_root = (
        cfg.real_data_root
        if cfg.phase == "2" and cfg.real_data_root is not None
        else cfg.data_root
    )
    # Val always uses synthetic data (has GT masks for the warmup IoU gate)
    val_data_root = cfg.data_root

    _exclude_cams = set(cfg.exclude_cams) if cfg.exclude_cams else set()
    train_dataset = ArticulatedDataset(
        data_root    = train_data_root,
        target_size  = cfg.img_size,
        num_frames   = cfg.num_frames,
        max_parts    = 8,
        phase        = ds_phase,
        split        = "train",
        val_ratio    = cfg.val_ratio,
        exclude_cams = _exclude_cams,
    )
    val_dataset = ArticulatedDataset(
        data_root    = val_data_root,
        target_size  = cfg.img_size,
        num_frames   = cfg.num_frames,
        max_parts    = 8,
        phase        = "1",   # always GT masks for validation
        split        = "val",
        val_ratio    = cfg.val_ratio,
        exclude_cams = _exclude_cams,
    )

    train_sampler = DistributedSampler(train_dataset) if is_dist else None
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = (train_sampler is None),
        sampler     = train_sampler,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_dist else None
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        sampler     = val_sampler,
        num_workers = cfg.num_workers,
    )

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_steps, eta_min=cfg.lr * 0.01,
    )

    scaler = GradScaler("cuda")

    # Restore optimizer/scheduler/warmup state if resuming
    if _resume_ckpt is not None:
        if "optimizer" in _resume_ckpt:
            try:
                optimizer.load_state_dict(_resume_ckpt["optimizer"])
                # Reset optimizer LR to cfg.lr (override inherited LR from previous phase)
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr
            except ValueError:
                # Parameter groups changed (e.g., cross-phase resume where frozen heads
                # are now unfrozen). Skip optimizer state; start fresh with cfg.lr.
                if is_main:
                    print("[resume] Optimizer param groups mismatch — starting optimizer fresh")
        if not cfg.reset_scheduler and "scheduler" in _resume_ckpt:
            # load_state_dict already restores last_epoch = start_step;
            # do NOT step again or LR would be at 2*start_step position.
            scheduler.load_state_dict(_resume_ckpt["scheduler"])
        # If reset_scheduler=True, scheduler stays fresh (T_max=cfg.total_steps, starts at lr)

    # ── Warmup gate state ──────────────────────────────────────────────────
    warmup_iou_checks_passed = 0
    warmup_done = not is_warmup   # already done if phase != "1a"
    if _resume_ckpt is not None and is_warmup:
        # Only restore warmup_done from checkpoint when still in warmup phase;
        # phase=1b/2 means warmup is unconditionally done.
        warmup_done = _resume_ckpt.get("warmup_done", warmup_done)

    # ── Training loop ─────────────────────────────────────────────────────
    step = start_step
    epoch = 0
    data_iter = iter(train_loader)

    os.makedirs(cfg.output_dir, exist_ok=True)

    if is_main:
        print(f"Starting training: phase={cfg.phase}, warmup={is_warmup}, "
              f"steps={cfg.total_steps}, device={device}")

    while step < cfg.total_steps:
        # Reload iterator when exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if is_dist:
                train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # ── Forward ───────────────────────────────────────────────────────
        images     = batch["images"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        timestamps = batch["timestamps"].to(device)

        optimizer.zero_grad()

        # BF16 autocast (bfloat16 has float32-range exponent → no overflow risk)
        # Falls back to float32 when use_bf16=False (legacy behaviour)
        with autocast("cuda", dtype=torch.bfloat16, enabled=getattr(cfg, "use_bf16", False)):
            preds = model(images, extrinsics, intrinsics, timestamps)
            loss, loss_dict = compute_loss(
                preds, batch, step, cfg, is_warmup=(not warmup_done),
                head=raw_model.sdf_head,
            )

        # Synchronise NaN/Inf check across all DDP ranks so all ranks skip together.
        # Without this, rank 0 might skip while ranks 1-3 wait for DDP all_reduce → deadlock.
        loss_ok = torch.tensor(float(torch.isfinite(loss)), device=device)
        if is_dist:
            dist.all_reduce(loss_ok, op=dist.ReduceOp.MIN)
        if loss_ok.item() < 0.5:
            if is_main:
                print(f"[step {step}] WARNING: non-finite loss detected, skipping batch", flush=True)
            del preds, loss, loss_dict
            torch.cuda.empty_cache()
            scheduler.step()
            step += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1

        # ── Logging ───────────────────────────────────────────────────────
        if is_main and step % cfg.log_interval == 0:
            loss_str = "  ".join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items()
            )
            print(f"[step {step:6d}] {loss_str}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        # ── Phase 1b per-frame IoU logging ────────────────────────────────
        if warmup_done and cfg.phase != "1a" and (step % cfg.val_interval == 0):
            if is_main:
                mean_iou = eval_mean_iou(raw_model, val_loader, device, cfg)
                print(f"[step {step}] per-frame val IoU = {mean_iou:.4f}", flush=True)
                iou_tensor = torch.tensor(mean_iou, device=device)
            else:
                iou_tensor = torch.zeros(1, device=device)
            if is_dist:
                dist.broadcast(iou_tensor, src=0)

        # ── Warmup gate ───────────────────────────────────────────────────
        if (not warmup_done) and (step % cfg.val_interval == 0):
            # Run val only on rank 0 to avoid OOM from all ranks evaluating
            # simultaneously, which causes torchrun to restart all workers.
            if is_main:
                mean_iou = eval_mean_iou(raw_model, val_loader, device, cfg)
                print(f"[step {step}] warmup val IoU = {mean_iou:.4f} "
                      f"(threshold {cfg.warmup_iou_threshold})")
                iou_tensor = torch.tensor(mean_iou, device=device)
            else:
                iou_tensor = torch.zeros(1, device=device)
            if is_dist:
                dist.broadcast(iou_tensor, src=0)
            mean_iou = iou_tensor.item()

            if mean_iou >= cfg.warmup_iou_threshold:
                warmup_iou_checks_passed += 1
            else:
                warmup_iou_checks_passed = 0

            if warmup_iou_checks_passed >= 3:
                warmup_done = True
                if is_main:
                    print(f"[step {step}] Warmup complete → unfreezing all heads")
                raw_model.set_phase(cfg.phase, warmup=False)
                # Save checkpoint at warmup transition so Phase 1a weights are preserved
                if is_main:
                    ckpt_path = Path(cfg.output_dir) / f"ckpt_warmup_{step:06d}.pth"
                    torch.save({
                        "step":        step,
                        "model":       raw_model.state_dict(),
                        "optimizer":   optimizer.state_dict(),
                        "scheduler":   scheduler.state_dict(),
                        "warmup_done": warmup_done,
                        "cfg":         vars(cfg),
                    }, ckpt_path)
                    print(f"Saved warmup checkpoint: {ckpt_path}")
                # Re-build optimizer with all trainable params (now includes
                # kinematic/dynamics/gaussian heads).
                # Do NOT recreate scaler — keep existing scale factor to avoid
                # gradient overflow spikes from the newly-unfrozen heads.
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=cfg.lr, weight_decay=cfg.weight_decay,
                )

        # ── Checkpointing ─────────────────────────────────────────────────
        if is_main and step % cfg.save_interval == 0:
            ckpt_path = Path(cfg.output_dir) / f"ckpt_{step:06d}.pth"
            torch.save({
                "step":        step,
                "model":       raw_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "warmup_done": warmup_done,
                "cfg":         vars(cfg),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if is_main:
        print("Training complete.")

    if is_dist:
        dist.destroy_process_group()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="FAST-4D ArtVGGT training")

    # Data
    p.add_argument("--data_root",       required=True)
    p.add_argument("--real_data_root",  default=None)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--img_size",        type=int,   default=518)
    p.add_argument("--num_frames",      type=int,   default=8)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument("--exclude_cams",    nargs="*",  default=["cam_00"],
                   help="Camera subdirs to exclude (default: cam_00, the back view). "
                        "Pass --exclude_cams with no args to include all cameras.")
    p.add_argument("--val_ratio",       type=float, default=0.15,
                   help="Fraction of scenes held out for validation")

    # Model
    p.add_argument("--n_gaussians",     type=int,   default=256)
    p.add_argument("--scene_radius",    type=float, default=1.0)

    # Training schedule
    p.add_argument("--phase",           default="1a",
                   choices=["1a", "1b", "2"])
    p.add_argument("--total_steps",     type=int, default=50000)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0)

    # Warmup gate
    p.add_argument("--warmup_iou_threshold", type=float, default=0.6)
    p.add_argument("--val_interval",    type=int, default=500)

    # Loss weights
    p.add_argument("--w_type",          type=float, default=0.5)
    p.add_argument("--w_axis",          type=float, default=0.5)
    p.add_argument("--w_pivot",         type=float, default=0.1)
    p.add_argument("--w_scalar",        type=float, default=0.3)
    p.add_argument("--w_dead_opacity",  type=float, default=0.1)
    p.add_argument("--w_render",        type=float, default=1.0,
                   help="Weight for per-part alpha rendering loss (local, no gsplat needed)")
    p.add_argument("--w_render_global", type=float, default=0.0,
                   help="Weight for global composited RGB rendering loss (gsplat rasterization). "
                        "Set > 0 in Phase 1b to supervise full-scene appearance.")
    p.add_argument("--w_pseudo_mask",   type=float, default=0.05)
    p.add_argument("--w_bbox",          type=float, default=0.5)
    p.add_argument("--w_pose_enc",      type=float, default=0.1,
                   help="Weight for camera pose encoding supervision loss. "
                        "Trains CameraHead during Phase 1b using GT extrinsics.")
    p.add_argument("--l1_sparsity",     type=float, default=0.1)
    p.add_argument("--l1_sparsity_warmup", type=float, default=0.0)

    # Auxiliary motion loss (patch + track pseudo-labels from motion_cache.npz).
    # Adds on top of GT part_mask supervision; disabled by default.
    p.add_argument("--w_motion_mask",  type=float, default=0.0,
                   help="Weight for patch-level motion pseudo-mask CE. "
                        "0 disables motion aux loss entirely.")
    p.add_argument("--w_motion_track", type=float, default=0.0,
                   help="Weight for track-level motion pseudo-label CE.")
    p.add_argument("--motion_warmup_steps", type=int, default=5000,
                   help="Linear ramp 0→1 of motion-loss scaling over this many steps.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing on Aggregator attention blocks "
                        "to reduce activation memory at the cost of ~33%% extra compute. "
                        "Recommended when num_frames > 2.")
    p.add_argument("--use_bf16", action="store_true",
                   help="Enable bfloat16 autocast for forward+loss. "
                        "BF16 has float32-range exponent so no overflow risk. "
                        "Gives ~1.5-2x speedup on Ampere+ GPUs with negligible quality loss.")

    # Logging
    p.add_argument("--log_interval",    type=int, default=50)
    p.add_argument("--save_interval",   type=int, default=2000)
    p.add_argument("--resume",          default=None)
    p.add_argument("--partial_resume",  action="store_true",
                   help="Load only aggregator weights from checkpoint (for arch changes)")
    p.add_argument("--reset_scheduler", action="store_true",
                   help="Start LR scheduler fresh from cfg.lr (ignore saved scheduler state)")
    p.add_argument("--reset_slot_tokens", action="store_true",
                   help="Re-init slot token embeddings after resume to break dominant-slot local minimum")

    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
