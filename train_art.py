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
from dggt.utils.rigid_transform import apply_rigid_transform
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
    assign_maps: torch.Tensor,   # [B, P, H_p, W_p]
    gt_masks: torch.Tensor,      # [B, P_gt, H, W]   (avg over S dim externally)
    matches: list,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """
    Foreground-weighted pixel-wise cross-entropy mask loss.

    For each pixel:
      - Covered by GT part gi matched to pred slot pi (pi >= 1): target = pi
      - All other pixels (background):                            target = 0

    Foreground pixels are upweighted (fg_weight) because background is the
    majority class (~75% of pixels) and dominates the unweighted CE signal,
    causing the model to stall on background routing while ignoring foreground.
    """
    B, P, H_p, W_p = assign_maps.shape
    H, W = gt_masks.shape[-2:]

    pred_up = F.interpolate(
        assign_maps, (H, W), mode="bilinear", align_corners=False
    )   # [B, P, H, W]
    log_pred = torch.log(pred_up.clamp(min=1e-8))   # log-probs for NLL loss

    total = pred_up.new_zeros(1)

    for b, (pred_idx, gt_idx) in enumerate(matches):
        # Build per-pixel label map: 0 = background slot
        label_map = torch.zeros(H, W, dtype=torch.long, device=pred_up.device)
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            # pi ∈ 1..P-1 (slot 0 excluded from foreground matching)
            label_map[gt_masks[b, gi] > 0.5] = pi

        # Per-pixel weight: upweight foreground pixels
        pixel_weight = torch.where(label_map > 0,
                                   label_map.new_full((), fg_weight).float(),
                                   label_map.new_ones(()).float())   # [H, W]

        # NLL loss with per-pixel weights
        nll = F.nll_loss(
            log_pred[b].unsqueeze(0),   # [1, P, H, W]
            label_map.unsqueeze(0),     # [1, H, W]
            reduction="none",
        ).squeeze(0)   # [H, W]

        total = total + (nll * pixel_weight).sum() / pixel_weight.sum()

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


# ============================================================================
# Training loop
# ============================================================================

def compute_loss(
    preds: dict,
    batch: dict,
    step: int,
    cfg: argparse.Namespace,
    is_warmup: bool,
) -> tuple[torch.Tensor, dict]:
    """
    Unified loss computation (all phases).

    Returns (total_loss, loss_dict).
    """
    B = preds["assign_maps"].shape[0]
    H, W = batch["images"].shape[-2:]
    device = preds["assign_maps"].device

    # Max-project GT masks over S frames → binary {0,1}, consistent with eval.
    # Using max (union) instead of mean ensures the full part footprint is covered.
    gt_masks = (batch["part_masks"].max(dim=1).values > 0.5).float().to(device)
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
    l_mask = mask_loss(assign_maps, gt_masks, matches)
    loss_dict["mask"] = l_mask

    # ── Slot sparsity (always active, weight ramps up after warmup) ───────
    sparsity_w = cfg.l1_sparsity_warmup if is_warmup else cfg.l1_sparsity
    l_sparse = slot_sparsity_loss(assign_maps, sparsity_w)
    loss_dict["sparsity"] = l_sparse

    if is_warmup:
        # Warmup: only mask + sparsity losses
        total = l_mask + l_sparse
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

    # ── Dead-slot opacity penalty ─────────────────────────────────────────
    opacity_flat = preds["gs_opacity"].squeeze(-1)   # [B, P, N_g]
    l_dead_op = dead_slot_opacity_loss(opacity_flat, is_dead, cfg.w_dead_opacity)
    loss_dict["dead_opacity"] = l_dead_op

    # ── Build sign-corrected axis/scalar in prediction-slot order ────────
    # Both local and global render losses index by prediction slot, not GT order.
    # Recompute sign flip in pred-slot order to avoid conflicting gradients.
    P_pred = preds["axis"].shape[1]
    flip_mask = preds["axis"].new_zeros(B, P_pred, 1)   # [B, P, 1], no grad
    for b, (pred_idx, gt_idx) in enumerate(matches):
        if len(pred_idx) > 0:
            dot_bm = (preds["axis"][b, pred_idx] * gt_axis[b, gt_idx]).sum(dim=-1)
            flip_mask[b, pred_idx, 0] = (dot_bm < 0).float()
    axis_for_render   = preds["axis"]    * (1.0 - 2.0 * flip_mask)   # [B, P, 3]
    scalar_for_render = preds["scalars"] * (1.0 - 2.0 * flip_mask)   # [B, P, S]

    # ── Per-part alpha rendering loss (local, differentiable, no gsplat) ─
    if cfg.w_render > 0.0 and render_extrinsics is not None:
        gt_masks_seq = batch["part_masks"].to(device)   # [B, S, P, H, W]
        l_render = cfg.w_render * per_part_alpha_render_loss(
            gs_mu              = preds["gs_mu"],
            gs_opacity         = preds["gs_opacity"],
            motion_type_logits = preds["motion_type_logits"],
            axis               = axis_for_render,
            pivot              = preds["pivot"],
            scalars            = scalar_for_render,
            extrinsics         = render_extrinsics,
            intrinsics         = batch["intrinsics"].to(device),
            gt_masks_seq       = gt_masks_seq,
            is_dead            = is_dead,
            patch_size         = 14,
        )
    else:
        l_render = preds["gs_mu"].new_zeros(1).squeeze()
    loss_dict["render"] = l_render

    # ── Global composited RGB rendering loss (gsplat rasterization) ──────
    w_render_global = getattr(cfg, "w_render_global", 0.0)
    if w_render_global > 0.0 and render_extrinsics is not None:
        l_render_global = w_render_global * global_render_loss(
            gs_mu              = preds["gs_mu"],
            gs_rot             = preds["gs_rot"],
            gs_scale           = preds["gs_scale"],
            gs_color           = preds["gs_color"],
            gs_opacity         = preds["gs_opacity"],
            motion_type_logits = preds["motion_type_logits"],
            axis               = axis_for_render,
            pivot              = preds["pivot"],
            scalars            = scalar_for_render,
            extrinsics         = render_extrinsics,
            intrinsics         = batch["intrinsics"].to(device),
            gt_images          = batch["images"].to(device),
            is_dead            = is_dead,
            max_frames         = 2,
        )
    else:
        l_render_global = preds["gs_mu"].new_zeros(1).squeeze()
    loss_dict["render_global"] = l_render_global

    # ── BBox centroid projection loss ─────────────────────────────────────
    if render_extrinsics is not None:
        l_bbox = cfg.w_bbox * bbox_loss(
            bbox_center = preds["bbox_center"],
            bbox_size   = preds["bbox_size"],
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
             + l_dead_op + l_render + l_render_global + l_bbox + l_pseudo + l_pose)
    loss_dict["total"] = total
    return total, loss_dict


# ============================================================================
# Validation IoU (for warmup transition)
# ============================================================================

@torch.no_grad()
def eval_mean_iou(model, val_loader, device, cfg) -> float:
    """Compute mean mask IoU over validation set (used for warmup gate).

    Uses Hungarian matching to align predicted slots to GT parts before
    computing IoU, so slot ordering ambiguity does not penalise the metric.
    """
    from dggt.utils.hungarian_matching import batch_hungarian_match
    model.eval()
    iou_sum, count = 0.0, 0

    try:
        for batch in val_loader:
            images     = batch["images"].to(device)
            extrinsics = batch["extrinsics"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            timestamps = batch["timestamps"].to(device)
            # Max-project GT masks over S frames
            part_masks_raw = batch["part_masks"].to(device)      # [B, S, P, H, W]
            gt_masks = (part_masks_raw.max(dim=1).values > 0.5).float()  # [B, P_gt, H, W]

            preds = model(images, extrinsics, intrinsics, timestamps)
            assign_maps = preds["assign_maps"]   # [B, P, H_p, W_p]
            B, P, H_p, W_p = assign_maps.shape
            H, W = gt_masks.shape[-2:]

            pred_up = F.interpolate(assign_maps, (H, W), mode="bilinear", align_corners=False)

            # Argmax assignment: each pixel → slot with highest probability.
            # This is mutually exclusive and consistent with binary GT masks.
            pred_argmax = pred_up.argmax(dim=1)   # [B, H, W]  slot index per pixel
            arange_p = torch.arange(P, device=device).view(1, P, 1, 1)
            pred_bin = (arange_p == pred_argmax.unsqueeze(1)).float()   # [B, P, H, W]

            # Hungarian matching: align pred slots → GT parts
            matches = batch_hungarian_match(pred_up, gt_masks)   # list[(pred_idx, gt_idx)]

            for b in range(B):
                pred_idx, gt_idx = matches[b]
                for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
                    inter = (pred_bin[b, pi] * gt_masks[b, gi]).sum()
                    union = (pred_bin[b, pi] + gt_masks[b, gi]).clamp(0, 1).sum()
                    iou_sum += (inter / (union + 1e-6)).item()
                    count += 1
    finally:
        model.train()   # always restore training mode

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
        use_camera_head=True,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = 2,
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

        # Use float32 (no AMP) to avoid float16 overflow NaN in render/attention ops
        preds = model(images, extrinsics, intrinsics, timestamps)
        loss, loss_dict = compute_loss(
            preds, batch, step, cfg, is_warmup=(not warmup_done)
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
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing on Aggregator attention blocks "
                        "to reduce activation memory at the cost of ~33%% extra compute. "
                        "Recommended when num_frames > 2.")

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
