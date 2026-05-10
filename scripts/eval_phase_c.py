"""
eval_phase_c.py — Comprehensive evaluation for Phase C (TrackEncoder + tracks-as-input).

Computes:
  • Mask metrics:      mIoU, per-part IoU, gap vs no-tracks ablation
  • Kinematic metrics: motion-type acc, axis cos, pivot L2, scalar L1
  • Track-level:       per-track classification acc (track → predicted part)

Visualizes (per scene, top-N):
  00_frames.png       S RGB frames
  01_tracks_gt.png    tracks colored by GT part label, overlaid on frame 0
  02_tracks_pred.png  tracks colored by *predicted* part (assign_map at track px)
  03_tracks_diff.png  red=mis-classified track, gray=correct, on frame 0
  04_masks_grid.png   GT vs Pred masks at first/middle/last frame
  06_joints.png       predicted axis + pivot per part on frame 0
  08_ablation.png     pred mask  with tracks   vs   without tracks   (frame 0)

Aggregates (full val set):
  summary/iou_histogram.png       per-scene mIoU distribution (with vs without)
  summary/tracks_help_scatter.png with-tracks IoU vs no-tracks IoU (each pt = scene)
  summary/track_acc_by_part.png   track classification acc, per GT part index

Usage:
  python scripts/eval_phase_c.py \
      --ckpt /data2/cyt/checkpoints/art_v20_phase_c/ckpt_NNNNNN.pth \
      --data_root /data2/cyt/data_root_refine \
      --output_dir ./eval_output/phase_c_NNNNNN \
      --num_vis_scenes 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dggt.models.art_vggt import ArtVGGT
from dggt.utils.dead_slot_gating import detect_dead_slots
from dggt.utils.hungarian_matching import batch_hungarian_match
from dggt.utils.rigid_transform import apply_rigid_transform
from datasets.articulated_dataset import ArticulatedDataset


# ─────────────────────────────────────────────────────────────────────────────
#  Color palette (consistent across GT / Pred visualisations)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = np.array([
    [ 30,  30,  30],   # 0 = background / dead
    [230,  25,  75],   # 1 part
    [ 60, 180,  75],   # 2
    [255, 225,  25],   # 3
    [  0, 130, 200],   # 4
    [245, 130,  48],   # 5
    [145,  30, 180],   # 6
    [ 70, 240, 240],   # 7
    [240,  50, 230],   # 8
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, cfg, device) -> ArtVGGT:
    saved_cfg = {}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "cfg" in ckpt:
        saved_cfg = ckpt["cfg"]
    # Default to False (Phase 1a / 1b without tracks). Phase C ckpt's cfg has it True.
    use_track_tokens = saved_cfg.get("use_track_tokens", False)
    num_frames_track = saved_cfg.get("num_frames", cfg.num_frames)
    n_gaussians      = saved_cfg.get("n_gaussians", 256)
    scene_radius     = saved_cfg.get("scene_radius", 1.0)

    model = ArtVGGT(
        img_size=cfg.img_size,
        patch_size=14,
        embed_dim=1024,
        num_slots=8,
        n_gaussians=n_gaussians,
        scene_radius=scene_radius,
        use_camera_head=True,
        stop_gradient_plucker=False,
        gradient_checkpointing=False,
        use_track_tokens=use_track_tokens,
        num_frames_track=num_frames_track,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[load] step={ckpt.get('step', '?')}  use_track_tokens={use_track_tokens}")
    if missing:
        print(f"[load] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[load] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model.eval()
    return model, ckpt.get("step", 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-frame alpha-projection IoU (mirrors eval_mean_iou in train_art.py)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def alpha_projection_per_part(
    preds: dict,
    extrinsics: torch.Tensor,    # [B, S, 4, 4]
    intrinsics: torch.Tensor,    # [B, 3, 3]
    H: int, W: int,
    patch_size: int = 14,
    sigma_patches: float = 0.8,
) -> torch.Tensor:
    """
    Returns alpha map per (slot, frame) at image resolution: [B, S, P, H, W] ∈ [0, 1].
    """
    device       = extrinsics.device
    B, S, _, _   = extrinsics.shape
    P            = preds["axis"].shape[1]
    H_p, W_p     = H // patch_size, W // patch_size

    motion_logits = preds["motion_type_logits"]
    axis          = preds["axis"]
    pivot         = preds["pivot"]
    scalars       = preds["scalars"]
    gs_mu         = preds["gs_mu"]
    gs_opacity    = preds["gs_opacity"]
    K             = intrinsics

    # motion probs [B, P, 3]
    mp2 = torch.softmax(motion_logits.float(), dim=-1)
    static_col = torch.zeros(B, P, 1, device=device, dtype=mp2.dtype)
    mp3 = torch.cat([static_col, mp2], dim=-1)
    mp3[:, 0] = torch.tensor([1., 0., 0.], device=device)

    gy = torch.arange(H_p, device=device, dtype=torch.float32) + 0.5
    gx = torch.arange(W_p, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)            # [H_p, W_p, 2]

    out = torch.zeros(B, S, P, H, W, device=device)

    for t in range(S):
        E = extrinsics[:, t]
        R_w2c = E[:, :3, :3].transpose(-1, -2)
        t_w2c = -torch.einsum("bij,bj->bi", R_w2c, E[:, :3, 3])
        sc_t  = scalars[:, :, t]                            # [B, P]

        for b in range(B):
            for p in range(P):
                mu_p = gs_mu[b, p]                          # [N_g, 3]
                N_g  = mu_p.shape[0]
                rot_p = torch.zeros(N_g, 4, device=device); rot_p[:, 0] = 1.0
                mu_t, _ = apply_rigid_transform(
                    mu_p, rot_p, mp3[b, p], axis[b, p], pivot[b, p], sc_t[b, p],
                )
                mu_cam = (R_w2c[b] @ mu_t.T + t_w2c[b].unsqueeze(-1)).T
                behind = mu_cam[:, 2] >= 0.0
                depth  = (-mu_cam[:, 2]).clamp(min=0.01)

                fx = K[b, 0, 0] / patch_size
                fy = K[b, 1, 1] / patch_size
                cx = K[b, 0, 2] / patch_size
                cy = K[b, 1, 2] / patch_size
                u = fx * mu_cam[:, 0] / depth + cx
                v = fy * mu_cam[:, 1] / depth + cy

                dx = grid[..., 0].unsqueeze(-1) - u.view(1, 1, -1)
                dy = grid[..., 1].unsqueeze(-1) - v.view(1, 1, -1)
                kern = torch.exp(-(dx*dx + dy*dy) / (2 * sigma_patches**2))
                op   = gs_opacity[b, p].squeeze(-1) * (~behind).float()
                alpha = (kern * op.view(1, 1, -1)).sum(-1).clamp(0, 1)

                up = F.interpolate(alpha.unsqueeze(0).unsqueeze(0),
                                   (H, W), mode="bilinear", align_corners=False)
                out[b, t, p] = up.squeeze()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Track classification accuracy
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def track_classification(
    assign_maps: torch.Tensor,         # [B, P, H_p, W_p]
    tracks_2d:   torch.Tensor,         # [B, S, N, 2]  pixel
    tracks_vis:  torch.Tensor,         # [B, S, N]
    track_part_label: torch.Tensor,    # [B, N, P_gt]  soft (sum=1) or one-hot
    matches: list,                      # list of (pred_idx, gt_idx) per b
    H: int, W: int,
):
    """
    Returns dict with per-batch arrays:
      pred_part : [B, N] long   predicted GT-part id (via Hungarian remap)
      gt_part   : [B, N] long   argmax of track_part_label
      visible   : [B, N] bool   visible in any frame
      correct   : [B, N] bool   pred_part == gt_part (only valid where visible)
    """
    B, P, H_p, W_p = assign_maps.shape
    device = assign_maps.device
    P_gt = track_part_label.shape[-1]

    pred_part = torch.full((B, tracks_2d.shape[2]), -1, dtype=torch.long, device=device)
    gt_part   = track_part_label.argmax(dim=-1).long()                        # [B, N]
    visible   = (tracks_vis.sum(dim=1) > 0)                                   # [B, N]

    # Sample assign_maps at frame-0 track pixels
    gx = tracks_2d[:, 0, :, 0] / max(W - 1, 1) * 2.0 - 1.0
    gy = tracks_2d[:, 0, :, 1] / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)                          # [B, 1, N, 2]
    sampled = F.grid_sample(
        assign_maps, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze(2)                                                               # [B, P, N]
    pred_slot = sampled.argmax(dim=1)                                          # [B, N]

    # Map pred slot → GT part via Hungarian
    for b in range(B):
        slot2part = torch.full((P,), -1, dtype=torch.long, device=device)
        pred_idx, gt_idx = matches[b]
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            slot2part[pi] = gi
        pred_part[b] = slot2part[pred_slot[b]]

    correct = (pred_part == gt_part) & visible & (pred_part >= 0)
    return {
        "pred_part": pred_part,
        "gt_part":   gt_part,
        "visible":   visible,
        "correct":   correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-scene metric computation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_one(model: ArtVGGT, batch: dict, device, run_ablation: bool,
                 force_disable_tracks: bool = False, use_bf16: bool = False):
    images        = batch["images"].to(device)
    extrinsics    = batch["extrinsics"].to(device)
    intrinsics    = batch["intrinsics"].to(device)
    timestamps    = batch["timestamps"].to(device)
    part_masks    = batch["part_masks"].to(device).float()                # [B, S, P_gt, H, W]
    tracks_2d_b   = batch["tracks_2d"].to(device)
    tracks_vis_b  = batch["tracks_vis"].to(device)
    tpl_b         = batch["track_part_label"].to(device)
    gt_motion     = batch["gt_motion_type"].to(device)
    gt_axis       = batch["gt_axis"].to(device)
    gt_pivot      = batch["gt_pivot"].to(device)
    gt_scalars    = batch["gt_scalars"].to(device)

    B, S, P_gt, H, W = part_masks.shape

    has_kin_gt   = bool(batch.get("has_kin_gt",   True))
    has_motion   = bool(batch.get("has_motion_data", True))
    if isinstance(has_kin_gt, torch.Tensor): has_kin_gt = bool(has_kin_gt.item())
    if isinstance(has_motion, torch.Tensor): has_motion = bool(has_motion.item())

    images_, extrinsics_, intrinsics_, timestamps_ = images, extrinsics, intrinsics, timestamps
    tracks_2d_, tracks_vis_ = tracks_2d_b, tracks_vis_b

    if use_bf16:
        # autocast mixed-precision forward (saves activation memory)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds_full = model(images_, extrinsics_, intrinsics_, timestamps_,
                               tracks_2d=tracks_2d_, tracks_vis=tracks_vis_,
                               disable_tracks=force_disable_tracks)
        preds_full = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                      for k, v in preds_full.items()}
    else:
        preds_full = model(images_, extrinsics_, intrinsics_, timestamps_,
                           tracks_2d=tracks_2d_, tracks_vis=tracks_vis_,
                           disable_tracks=force_disable_tracks)

    am_full = preds_full["assign_maps"]                                    # [B, P, H_p, W_p]
    pred_up = F.interpolate(am_full, (H, W), mode="bilinear", align_corners=False)
    gt_f0   = (part_masks[:, 0] > 0.5).float()
    matches = batch_hungarian_match(pred_up, gt_f0)
    is_dead = detect_dead_slots(am_full)                                   # [B, P]

    # ── (1) assign_map IoU @ frame 0 — pure slot routing, no GS, no motion ─
    # Hard argmax of the upsampled assign_maps, per-pixel slot id.
    am_argmax = pred_up.argmax(dim=1)                                      # [B, H, W]
    assign_iou_pairs = []
    for b in range(B):
        pred_idx, gt_idx = matches[b]
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            if is_dead[b, pi]:
                continue
            gt_m = (part_masks[b, 0, gi] > 0.5).float()
            if gt_m.sum() < 1: continue
            pr_m = (am_argmax[b] == pi).float()
            inter = (pr_m * gt_m).sum()
            union = (pr_m + gt_m).clamp(0, 1).sum()
            assign_iou_pairs.append((int(gi), float((inter / (union + 1e-6)).item())))
    assign_iou = float(np.mean([x[1] for x in assign_iou_pairs])) if assign_iou_pairs else 0.0

    # ── (1.5) Foreground-only IoU @ frame 0 — IGNORES slot/part identity ────
    # union(pred slots 1..P-1) vs union(GT parts 1..P_gt-1).  Answers
    # "did the model find the SET of moving regions?"  Robust to SAM2 over-seg
    # (real data) and to slot-vs-part count mismatch.
    fg_iou_list = []
    for b in range(B):
        # Pred foreground = any non-background, non-dead slot wins argmax
        alive_mask = (~is_dead[b]).float().to(am_full.device)        # [P]
        # Mask out dead slots before argmax → if any live slot's score >
        # background, this pixel is foreground
        am_b = pred_up[b]                                              # [P, H, W]
        am_b_masked = am_b.clone()
        for pi in range(am_b.shape[0]):
            if is_dead[b, pi]:
                am_b_masked[pi] = -1e9
        pred_arg = am_b_masked.argmax(dim=0)
        pred_fg  = ((pred_arg >= 1) & (~is_dead[b][pred_arg])).float()  # [H, W]

        gt_fg = (part_masks[b, 0, 1:] > 0.5).any(dim=0).float()          # [H, W]
        if gt_fg.sum() < 1: continue
        inter = (pred_fg * gt_fg).sum()
        union = (pred_fg + gt_fg).clamp(0, 1).sum()
        fg_iou_list.append(float((inter / (union + 1e-6)).item()))
    fg_iou = float(np.mean(fg_iou_list)) if fg_iou_list else 0.0

    # ── (2/3) Per-frame alpha-projection IoU — full GS + motion pipeline ──
    alpha = alpha_projection_per_part(preds_full, extrinsics, intrinsics, H, W)
    pred_bin = (alpha > 0.5).float()                                       # [B, S, P, H, W]

    iou_per_part = []                                                       # list of (gi, iou)
    iou_t0_pairs = []                                                       # frame-0 only
    iou_tpos_pairs = []                                                     # t > 0 only
    for b in range(B):
        pred_idx, gt_idx = matches[b]
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            if is_dead[b, pi]:
                continue
            ious = []
            ious_pos = []
            for t in range(S):
                gt_m = (part_masks[b, t, gi] > 0.5).float()
                if gt_m.sum() < 1: continue
                pr_m = pred_bin[b, t, pi]
                inter = (pr_m * gt_m).sum()
                union = (pr_m + gt_m).clamp(0, 1).sum()
                v = (inter / (union + 1e-6)).item()
                ious.append(v)
                if t == 0:
                    iou_t0_pairs.append((int(gi), float(v)))
                else:
                    ious_pos.append(v)
            if ious_pos:
                iou_tpos_pairs.append((int(gi), float(np.mean(ious_pos))))
            if ious:
                iou_per_part.append((int(gi), float(np.mean(ious))))

    # Mean IoUs (the three flavors)
    mIoU       = float(np.mean([x[1] for x in iou_per_part]))   if iou_per_part   else 0.0
    iou_t0     = float(np.mean([x[1] for x in iou_t0_pairs]))   if iou_t0_pairs   else 0.0
    iou_tpos   = float(np.mean([x[1] for x in iou_tpos_pairs])) if iou_tpos_pairs else 0.0

    # Track classification (only when GT track-part-label is available)
    if has_motion:
        track_cls = track_classification(am_full, tracks_2d_b, tracks_vis_b, tpl_b,
                                         matches, H, W)
        n_vis = int(track_cls["visible"].sum().item())
        n_cor = int(track_cls["correct"].sum().item())
        track_acc = n_cor / max(n_vis, 1)
    else:
        track_cls = None
        n_vis = n_cor = 0
        track_acc = float("nan")

    # Kinematic metrics: only when has_kin_gt
    kin = {"axis_cos": [], "pivot_l2": [], "scalar_l1": [], "type_acc": []}
    if has_kin_gt:
        for b in range(B):
            pred_idx, gt_idx = matches[b]
            for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
                if is_dead[b, pi] or gi == 0:
                    continue
                ax_p = preds_full["axis"][b, pi]
                ax_g = gt_axis[b, gi]
                kin["axis_cos"].append(float((ax_p * ax_g).sum().abs().item()))

                pv_p = preds_full["pivot"][b, pi]
                pv_g = gt_pivot[b, gi]
                kin["pivot_l2"].append(float((pv_p - pv_g).norm().item()))

                sc_p = preds_full["scalars"][b, pi]
                sc_g = gt_scalars[b, gi]
                sign = 1.0 if (ax_p * ax_g).sum() >= 0 else -1.0
                kin["scalar_l1"].append(float((sign * sc_p - sc_g).abs().mean().item()))

                type_pred = int(preds_full["motion_type_logits"][b, pi].argmax().item()) + 1
                type_gt   = int(gt_motion[b, gi].item())
                kin["type_acc"].append(1.0 if type_pred == type_gt else 0.0)

    # Ablation
    mIoU_ablate       = None        # alpha-projection IoU without tracks
    assign_iou_ablate = None        # assign_map  IoU without tracks  ← NEW
    fg_iou_ablate     = None        # foreground IoU without tracks
    pred_bin_ablate   = None
    if run_ablation:
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds_ablate = model(images_, extrinsics_, intrinsics_, timestamps_,
                                     tracks_2d=tracks_2d_, tracks_vis=tracks_vis_,
                                     disable_tracks=True)
            preds_ablate = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                            for k, v in preds_ablate.items()}
        else:
            preds_ablate = model(images_, extrinsics_, intrinsics_, timestamps_,
                                 tracks_2d=tracks_2d_, tracks_vis=tracks_vis_,
                                 disable_tracks=True)
        am_a = preds_ablate["assign_maps"]
        is_dead_a = detect_dead_slots(am_a)
        pred_up_a = F.interpolate(am_a, (H, W), mode="bilinear", align_corners=False)
        matches_a = batch_hungarian_match(pred_up_a, gt_f0)

        # Assign-map IoU @ frame 0 (no GS, no motion)
        am_a_argmax = pred_up_a.argmax(dim=1)
        assign_a_pairs = []
        for b in range(B):
            pred_idx, gt_idx = matches_a[b]
            for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
                if is_dead_a[b, pi]: continue
                gt_m = (part_masks[b, 0, gi] > 0.5).float()
                if gt_m.sum() < 1: continue
                pr_m = (am_a_argmax[b] == pi).float()
                inter = (pr_m * gt_m).sum()
                union = (pr_m + gt_m).clamp(0, 1).sum()
                assign_a_pairs.append(float((inter / (union + 1e-6)).item()))
        assign_iou_ablate = float(np.mean(assign_a_pairs)) if assign_a_pairs else 0.0

        # Foreground IoU (no slot identity) under ablation
        fg_a_list = []
        for b in range(B):
            am_b_a = pred_up_a[b].clone()
            for pi in range(am_b_a.shape[0]):
                if is_dead_a[b, pi]:
                    am_b_a[pi] = -1e9
            pred_arg_a = am_b_a.argmax(dim=0)
            pred_fg_a  = ((pred_arg_a >= 1) & (~is_dead_a[b][pred_arg_a])).float()
            gt_fg = (part_masks[b, 0, 1:] > 0.5).any(dim=0).float()
            if gt_fg.sum() < 1: continue
            inter = (pred_fg_a * gt_fg).sum()
            union = (pred_fg_a + gt_fg).clamp(0, 1).sum()
            fg_a_list.append(float((inter / (union + 1e-6)).item()))
        fg_iou_ablate = float(np.mean(fg_a_list)) if fg_a_list else 0.0

        alpha_a = alpha_projection_per_part(preds_ablate, extrinsics, intrinsics, H, W)
        pred_bin_a = (alpha_a > 0.5).float()
        ious_a = []
        for b in range(B):
            pred_idx, gt_idx = matches_a[b]
            for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
                if is_dead_a[b, pi]: continue
                for t in range(S):
                    gt_m = (part_masks[b, t, gi] > 0.5).float()
                    if gt_m.sum() < 1: continue
                    pr_m = pred_bin_a[b, t, pi]
                    inter = (pr_m * gt_m).sum()
                    union = (pr_m + gt_m).clamp(0, 1).sum()
                    ious_a.append((inter / (union + 1e-6)).item())
        mIoU_ablate     = float(np.mean(ious_a)) if ious_a else 0.0
        pred_bin_ablate = pred_bin_a

    return {
        "scene_id":          batch["scene_id"][0] if isinstance(batch["scene_id"], list) else str(batch["scene_id"]),
        "assign_iou":        assign_iou,            # slot routing (Hungarian-aligned per-part)
        "assign_iou_ablate": assign_iou_ablate,
        "fg_iou":            fg_iou,                # ★ identity-free foreground IoU
        "fg_iou_ablate":     fg_iou_ablate,
        "mIoU_alpha":        mIoU,                  # alpha-projection (auxiliary)
        "mIoU_alpha_ablate": mIoU_ablate,
        "iou_t0":            iou_t0,                # GS at rest pose
        "iou_tpos":          iou_tpos,              # GS + motion (t>0)
        "iou_per_part":      iou_per_part,
        "track_acc":    track_acc,
        "n_vis_tracks": n_vis,
        "n_correct_tracks": n_cor,
        "track_cls":    ({k: v.cpu() for k, v in track_cls.items()} if track_cls is not None else None),
        "kin":          {k: float(np.mean(v)) if v else float("nan") for k, v in kin.items()},
        "matches":      [(pi.cpu().numpy(), gi.cpu().numpy()) for pi, gi in matches],
        "is_dead":      is_dead.cpu().numpy(),
        "pred_bin":     pred_bin.cpu(),
        "pred_bin_ablate": (pred_bin_ablate.cpu() if pred_bin_ablate is not None else None),
        "preds_full":   {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                         for k, v in preds_full.items()
                         if k in ["axis", "pivot", "motion_type_logits", "scalars"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_numpy_img(img_chw: torch.Tensor) -> np.ndarray:
    """[3, H, W] in [0, 1] → [H, W, 3] uint8."""
    x = img_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def viz_frames(images: torch.Tensor, out_path: Path):
    """images: [S, 3, H, W]"""
    S = images.shape[0]
    fig, axes = plt.subplots(1, S, figsize=(2.0 * S, 2.0))
    if S == 1: axes = [axes]
    for t in range(S):
        axes[t].imshow(_to_numpy_img(images[t]))
        axes[t].set_title(f"t={t}", fontsize=8)
        axes[t].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def viz_tracks_colored(
    image_f0: torch.Tensor,
    tracks_2d: torch.Tensor,
    tracks_vis: torch.Tensor,
    labels: np.ndarray,
    out_path: Path,
    title: str = "",
):
    """
    image_f0:  [3, H, W]
    tracks_2d: [S, N, 2] pixel
    tracks_vis:[S, N]
    labels:    [N] int  (-1 = unknown / not predicted; 0..P_gt-1 = part)
    """
    S, N, _ = tracks_2d.shape
    img = _to_numpy_img(image_f0)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100))
    ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=10)

    vis_any = tracks_vis.sum(dim=0) > 0                     # [N]
    pts = tracks_2d.cpu().numpy()
    vis = tracks_vis.cpu().numpy()
    for n in range(N):
        if not bool(vis_any[n]): continue
        lab = int(labels[n]) if 0 <= labels[n] < len(PALETTE) else 0
        color = PALETTE[lab] / 255.0

        # Trajectory polyline (visible frames only)
        xs = [pts[t, n, 0] for t in range(S) if vis[t, n] > 0]
        ys = [pts[t, n, 1] for t in range(S) if vis[t, n] > 0]
        if len(xs) >= 2:
            ax.plot(xs, ys, "-", color=color, linewidth=0.6, alpha=0.5)
        if vis[0, n] > 0:
            ax.plot(pts[0, n, 0], pts[0, n, 1], "o", color=color,
                    markersize=2.5, markeredgecolor="white", markeredgewidth=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def viz_tracks_diff(
    image_f0: torch.Tensor,
    tracks_2d: torch.Tensor,
    tracks_vis: torch.Tensor,
    correct: np.ndarray,
    visible: np.ndarray,
    out_path: Path,
):
    """correct[N] bool; visible[N] bool; mis-classified visible tracks → red."""
    img = _to_numpy_img(image_f0)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100))
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"track diff  ({int(visible.sum() - correct.sum())} wrong / "
                 f"{int(visible.sum())} visible)", fontsize=10)

    pts = tracks_2d[0].cpu().numpy()
    for n in range(pts.shape[0]):
        if not visible[n]: continue
        x, y = pts[n]
        if correct[n]:
            ax.plot(x, y, "o", color=(0.6, 0.6, 0.6), markersize=2.0, alpha=0.7)
        else:
            ax.plot(x, y, "o", color=(1.0, 0.1, 0.1), markersize=3.0,
                    markeredgecolor="white", markeredgewidth=0.4)
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def _overlay_masks(img: np.ndarray, mask_per_part: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """img: [H, W, 3] uint8; mask_per_part: [P, H, W] float in [0, 1]. Output uint8."""
    out = img.astype(np.float32)
    P, H, W = mask_per_part.shape
    arg = mask_per_part.argmax(axis=0)                   # [H, W]
    val = mask_per_part.max(axis=0)
    for p in range(1, min(P, len(PALETTE))):
        m = (arg == p) & (val > 0.3)
        if not m.any(): continue
        c = PALETTE[p].astype(np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
    return out.clip(0, 255).astype(np.uint8)


def viz_masks_grid(
    images: torch.Tensor,                # [S, 3, H, W]
    gt_masks_per_frame: torch.Tensor,    # [S, P_gt, H, W]
    pred_masks_per_frame: torch.Tensor,  # [S, P, H, W]   (in pred-slot order)
    matches_b,                            # (pred_idx, gt_idx)
    is_dead_b,                            # [P] bool
    out_path: Path,
    frames=(0, None, -1),
):
    S = images.shape[0]
    f_list = [t if t is not None else S // 2 for t in frames]
    f_list = [t if t >= 0 else S + t for t in f_list]

    # Re-order pred mask channels to GT-part order so colors match
    pred_idx, gt_idx = matches_b
    P_gt = gt_masks_per_frame.shape[1]
    pm = torch.zeros_like(gt_masks_per_frame)
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        if is_dead_b[pi]:
            continue
        pm[:, gi] = pred_masks_per_frame[:, pi]

    fig, axes = plt.subplots(2, len(f_list), figsize=(3.0 * len(f_list), 6.0))
    if len(f_list) == 1: axes = axes[:, None]
    for col, t in enumerate(f_list):
        rgb = _to_numpy_img(images[t])
        gt_overlay = _overlay_masks(rgb, gt_masks_per_frame[t].cpu().numpy())
        pr_overlay = _overlay_masks(rgb, pm[t].cpu().numpy())
        axes[0, col].imshow(gt_overlay); axes[0, col].set_title(f"GT  t={t}", fontsize=10); axes[0, col].axis("off")
        axes[1, col].imshow(pr_overlay); axes[1, col].set_title(f"Pred t={t}", fontsize=10); axes[1, col].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def viz_joints(
    image_f0: torch.Tensor,
    pred_axis: torch.Tensor,             # [P_gt, 3]   (re-ordered)
    pred_pivot: torch.Tensor,            # [P_gt, 3]
    pred_motion_logits: torch.Tensor,    # [P_gt, 2]
    intrinsics: torch.Tensor,            # [3, 3]
    extrinsics: torch.Tensor,            # [4, 4] cam-to-world (frame 0)
    is_active_part: np.ndarray,          # [P_gt] bool
    out_path: Path,
):
    img = _to_numpy_img(image_f0)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100))
    ax.imshow(img); ax.axis("off"); ax.set_title("predicted joints (axis + pivot)", fontsize=10)

    R_w2c = extrinsics[:3, :3].T
    t_w2c = -R_w2c @ extrinsics[:3, 3]

    def project(p3):
        cam = R_w2c @ p3 + t_w2c
        if cam[2].item() >= 0: return None
        depth = -cam[2]
        u = (intrinsics[0, 0] * cam[0] / depth + intrinsics[0, 2]).item()
        v = (intrinsics[1, 1] * cam[1] / depth + intrinsics[1, 2]).item()
        return float(u), float(v)

    L = 0.25     # axis arrow half-length in world units
    P_gt = pred_axis.shape[0]
    for p in range(1, P_gt):                     # skip slot 0 (background)
        if not is_active_part[p]:
            continue
        ax3 = pred_axis[p].cpu()
        pv3 = pred_pivot[p].cpu()
        m_t = int(pred_motion_logits[p].argmax().item())   # 0=prismatic, 1=revolute
        a = ax3 / (ax3.norm() + 1e-6)
        p1 = project(pv3 - L * a)
        p2 = project(pv3 + L * a)
        pp = project(pv3)
        if pp is None: continue
        color = PALETTE[p] / 255.0
        ax.plot(*pp, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.0)
        if p1 is not None and p2 is not None:
            linestyle = "--" if m_t == 0 else "-"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    linestyle=linestyle, color=color, linewidth=2.5, alpha=0.85)
            arrow = FancyArrowPatch(p1, p2, arrowstyle="->", color=color,
                                    mutation_scale=15, linewidth=0.8)
            ax.add_patch(arrow)

    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def viz_ablation(
    image_f0: torch.Tensor,
    pred_full_f0: torch.Tensor,          # [P_gt, H, W]   (re-ordered, GT-part-aligned)
    pred_ablate_f0: torch.Tensor,        # [P_gt, H, W]   (re-ordered)
    iou_full: float,
    iou_ablate: float,
    out_path: Path,
):
    img = _to_numpy_img(image_f0)
    overlay_full   = _overlay_masks(img, pred_full_f0.cpu().numpy())
    overlay_ablate = _overlay_masks(img, pred_ablate_f0.cpu().numpy())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(overlay_full);   axes[0].set_title(f"with tracks   IoU={iou_full:.3f}", fontsize=10);   axes[0].axis("off")
    axes[1].imshow(overlay_ablate); axes[1].set_title(f"no tracks      IoU={iou_ablate:.3f}", fontsize=10); axes[1].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Per-scene viz orchestration
# ─────────────────────────────────────────────────────────────────────────────
def visualise_scene(batch: dict, result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    images       = batch["images"][0]                     # [S, 3, H, W]
    part_masks   = batch["part_masks"][0]                 # [S, P_gt, H, W]
    tracks_2d    = batch["tracks_2d"][0]                  # [S, N, 2]
    tracks_vis   = batch["tracks_vis"][0]                 # [S, N]
    extrinsics   = batch["extrinsics"][0]
    intrinsics   = batch["intrinsics"][0]

    matches_b   = result["matches"][0]
    is_dead_b   = result["is_dead"][0]
    pred_bin    = result["pred_bin"][0]                   # [S, P, H, W]
    track_cls   = result.get("track_cls", None)

    # 00 frames
    viz_frames(images, out_dir / "00_frames.png")

    # 01–03: only when GT track labels exist
    if track_cls is not None:
        gt_part   = track_cls["gt_part"][0].cpu().numpy()
        pred_part = track_cls["pred_part"][0].cpu().numpy()
        visible   = track_cls["visible"][0].cpu().numpy()
        correct   = track_cls["correct"][0].cpu().numpy()
        viz_tracks_colored(images[0], tracks_2d, tracks_vis, gt_part,
                           out_dir / "01_tracks_gt.png", title="tracks colored by GT part")
        viz_tracks_colored(images[0], tracks_2d, tracks_vis,
                           np.where(pred_part >= 0, pred_part, 0).astype(np.int64),
                           out_dir / "02_tracks_pred.png", title="tracks colored by predicted part")
        viz_tracks_diff(images[0], tracks_2d, tracks_vis, correct, visible,
                        out_dir / "03_tracks_diff.png")

    # 04 masks grid (GT vs Pred at 3 frames)
    viz_masks_grid(images, part_masks, pred_bin, matches_b, is_dead_b,
                   out_dir / "04_masks_grid.png")

    # 06 joints
    P_gt = part_masks.shape[1]
    pred_idx, gt_idx = matches_b
    pred_axis_gt = torch.zeros(P_gt, 3)
    pred_pivot_gt = torch.zeros(P_gt, 3)
    pred_mtl_gt = torch.zeros(P_gt, 2)
    is_active = np.zeros(P_gt, dtype=bool)
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        if is_dead_b[pi]: continue
        pred_axis_gt[gi]  = result["preds_full"]["axis"][0, pi]
        pred_pivot_gt[gi] = result["preds_full"]["pivot"][0, pi]
        pred_mtl_gt[gi]   = result["preds_full"]["motion_type_logits"][0, pi]
        is_active[gi]     = True
    viz_joints(images[0], pred_axis_gt, pred_pivot_gt, pred_mtl_gt,
               intrinsics, extrinsics[0], is_active,
               out_dir / "06_joints.png")

    # 08 ablation (frame-0 mask comparison)
    if result["pred_bin_ablate"] is not None:
        # Ordering: ablate uses its own matches; re-order to GT order using THE SAME pred_bin
        # but matched independently. Since we don't store ablate matches, reuse 'matches_b'
        # which is for full pass; the ablation re-uses its own assign->mask.  For viz,
        # just put both in pred-slot-0 → GT-part-0 etc. by re-projecting via GT mask 0
        # ordering. Cheap proxy: do a second Hungarian for ablate at viz time.
        # part_masks here is already [S, P_gt, H, W] (batch dim squeezed in caller).
        # We want frame-0 GT masks shaped [1, P_gt, H, W] for batch_hungarian.
        gt_f0 = (part_masks[0].unsqueeze(0) > 0.5).float().to(pred_bin.device)
        pred_a = result["pred_bin_ablate"][0]                # [S, P, H, W]
        pred_a_f0 = pred_a[0].unsqueeze(0)                   # [1, P, H, W]
        m_a = batch_hungarian_match(pred_a_f0, gt_f0)
        pa_full = torch.zeros(P_gt, *pred_a.shape[-2:])
        pa_abl  = torch.zeros(P_gt, *pred_a.shape[-2:])
        # full
        for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
            if is_dead_b[pi]: continue
            pa_full[gi] = pred_bin[0, pi]
        # ablate
        for pi, gi in zip(m_a[0][0].tolist(), m_a[0][1].tolist()):
            pa_abl[gi] = pred_a[0, pi]
        viz_ablation(images[0], pa_full, pa_abl,
                     result["assign_iou"], result["assign_iou_ablate"],
                     out_dir / "08_ablation.png")

    # metrics dump
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "scene_id":          result["scene_id"],
            "assign_iou":        result["assign_iou"],
            "assign_iou_ablate": result["assign_iou_ablate"],
            "fg_iou":            result["fg_iou"],
            "fg_iou_ablate":     result["fg_iou_ablate"],
            "mIoU_alpha":        result["mIoU_alpha"],
            "mIoU_alpha_ablate": result["mIoU_alpha_ablate"],
            "iou_t0":            result["iou_t0"],
            "iou_tpos":          result["iou_tpos"],
            "iou_per_part":      result["iou_per_part"],
            "track_acc":         result["track_acc"],
            "n_vis_tracks":      result["n_vis_tracks"],
            "n_correct_tracks":  result["n_correct_tracks"],
            "kin":               result["kin"],
        }, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary(rows: list[dict], out_dir: Path):
    if not rows:
        print("[plot_summary] no rows; skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # PRIMARY: assign_iou (slot routing only).  alpha-IoU stays as auxiliary.
    full = np.array([r["assign_iou"] for r in rows])
    has_abl = all(r.get("assign_iou_ablate") is not None for r in rows)
    abl  = np.array([r["assign_iou_ablate"] for r in rows]) if has_abl else None

    # Primary histogram (assign_iou)
    plt.figure(figsize=(6, 4))
    plt.hist(full, bins=30, alpha=0.7, label="with tracks", color="C0")
    if has_abl: plt.hist(abl, bins=30, alpha=0.5, label="no tracks", color="C3")
    plt.xlabel("per-scene assign_iou"); plt.ylabel("count"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"assign_iou distribution (N={len(rows)})")
    plt.tight_layout(); plt.savefig(out_dir / "iou_histogram.png", dpi=120); plt.close()

    # Auxiliary alpha histogram
    full_a = np.array([r["mIoU_alpha"] for r in rows])
    abl_a  = np.array([r["mIoU_alpha_ablate"] for r in rows]) if has_abl else None
    plt.figure(figsize=(6, 4))
    plt.hist(full_a, bins=30, alpha=0.7, label="with tracks", color="C0")
    if has_abl: plt.hist(abl_a, bins=30, alpha=0.5, label="no tracks", color="C3")
    plt.xlabel("per-scene alpha IoU (end-to-end)"); plt.ylabel("count"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"alpha IoU distribution (auxiliary, N={len(rows)})")
    plt.tight_layout(); plt.savefig(out_dir / "iou_histogram_alpha.png", dpi=120); plt.close()

    # Primary scatter (assign_iou)
    if has_abl:
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(abl, full, s=14, alpha=0.6)
        lo, hi = 0, max(full.max(), abl.max()) * 1.05
        plt.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, label="y=x")
        plt.xlabel("no-tracks assign_iou"); plt.ylabel("with-tracks assign_iou")
        gap = full - abl
        plt.title(f"Tracks help slot routing?  Δ={gap.mean():+.4f}  ({(gap > 0).mean() * 100:.0f}% above y=x)")
        plt.xlim(lo, hi); plt.ylim(lo, hi); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / "tracks_help_scatter.png", dpi=120); plt.close()

    # Track classification by part
    n_parts = max((max(p for p, _ in r["iou_per_part"]) + 1 if r["iou_per_part"] else 1) for r in rows)
    by_part_correct = np.zeros(n_parts); by_part_total = np.zeros(n_parts)
    for r in rows:
        # lazy approach: re-aggregate from per-scene track_acc/visible (simpler: skip part-wise)
        pass
    # We don't have per-part track stats stored; skip detailed per-part bar (would need re-run).

    # Track acc histogram (skip if all NaN, e.g. real_data without GT track labels)
    accs = np.array([r["track_acc"] for r in rows])
    valid = ~np.isnan(accs)
    if valid.any():
        accs_v = accs[valid]
        plt.figure(figsize=(6, 4))
        plt.hist(accs_v, bins=30, color="C2", alpha=0.8)
        plt.axvline(accs_v.mean(), color="red", linestyle="--", label=f"mean={accs_v.mean():.3f}")
        plt.xlabel("per-scene track classification acc"); plt.ylabel("count")
        plt.legend(); plt.grid(alpha=0.3); plt.title("Track classification accuracy")
        plt.tight_layout(); plt.savefig(out_dir / "track_acc_histogram.png", dpi=120); plt.close()

    # ── IoU decomposition: assign_iou → iou_t0 → iou_tpos → alpha_all ─────
    if all("assign_iou" in r for r in rows):
        ai = np.array([r["assign_iou"] for r in rows])
        i0 = np.array([r["iou_t0"]     for r in rows])
        ip = np.array([r["iou_tpos"]   for r in rows])
        ia = np.array([r["mIoU_alpha"] for r in rows])     # end-to-end alpha

        # 1) Per-scene scatter showing the cascade
        plt.figure(figsize=(7.2, 4.5))
        x = np.arange(len(rows))
        order = np.argsort(-ai)              # sort by best slot routing
        plt.plot(x, ai[order], "o-", label="assign_iou (slot routing) ★",  color="C0", markersize=3, linewidth=0.8)
        plt.plot(x, i0[order], "s-", label="iou_t0 (GS @ rest pose)",      color="C2", markersize=3, linewidth=0.8)
        plt.plot(x, ip[order], "^-", label="iou_tpos (GS + motion, t>0)",  color="C1", markersize=3, linewidth=0.8)
        plt.plot(x, ia[order], "x-", label="alpha_all (end-to-end)",        color="C3", markersize=3, linewidth=0.8)
        plt.xlabel("scene rank (sorted by assign_iou)")
        plt.ylabel("IoU")
        plt.legend(loc="lower left", fontsize=8)
        plt.grid(alpha=0.3)
        plt.title("IoU decomposition per scene")
        plt.tight_layout(); plt.savefig(out_dir / "iou_decomposition.png", dpi=120); plt.close()

        # 2) Bar chart of mean drops at each stage
        means = [ai.mean(), i0.mean(), ip.mean(), ia.mean()]
        labels = ["assign_iou\n(routing) ★", "iou_t0\n(rest GS)", "iou_tpos\n(GS+motion)", "alpha_all\n(end-to-end)"]
        drops = [None] + [means[i] - means[i-1] for i in range(1, 4)]

        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        bars = ax.bar(labels, means, color=["C0", "C2", "C1", "C3"], alpha=0.85)
        for i, b in enumerate(bars):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10)
            if drops[i] is not None:
                ax.text(b.get_x() + b.get_width()/2, h - 0.04, f"Δ={drops[i]:+.3f}",
                        ha="center", va="top", color="white", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(means) * 1.15)
        ax.set_ylabel("mean IoU")
        ax.set_title("Where is the IoU lost?  (cascade decomposition)")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(out_dir / "iou_cascade.png", dpi=120); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_type", choices=["sim", "real"], default="sim",
                   help="sim = ArticulatedDataset (PartNet-Mobility);  "
                        "real = iTACORealDataset (no GT kin, no tracks).")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--img_size",   type=int, default=518)
    p.add_argument("--num_vis_scenes", type=int, default=30)
    p.add_argument("--no_ablation", action="store_true")
    p.add_argument("--motion_cache_name", default="motion_cache_gt.npz")
    p.add_argument("--max_tracks_per_sample", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--exclude_cams", nargs="*", default=["cam_00"])
    p.add_argument("--limit_scenes", type=int, default=-1,
                   help="If >0, evaluate only the first N scenes (for fast smoke test).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use_bf16", action="store_true",
                   help="bf16 autocast in forward (halves activation memory)")
    return p.parse_args()


def main():
    cfg = parse_args()
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    model, step = load_model(cfg.ckpt, cfg, device)
    # NOTE: full model.bfloat16() breaks because some forward paths create fp32
    # intermediates (Plucker rays, position grids) that mismatch bf16 weights.
    # Use --num_frames 4 instead to halve memory if OOM.

    if cfg.dataset_type == "real":
        from datasets.itaco_real_dataset import iTACORealDataset
        val_ds = iTACORealDataset(
            data_root      = cfg.data_root,
            target_size    = cfg.img_size,
            num_frames     = cfg.num_frames,
            max_parts      = 8,
            max_tracks     = cfg.max_tracks_per_sample,
            split          = "all",
            random_start   = False,    # deterministic eval
        )
    else:
        val_ds = ArticulatedDataset(
            data_root         = cfg.data_root,
            target_size       = cfg.img_size,
            num_frames        = cfg.num_frames,
            max_parts         = 8,
            phase             = "1",
            split             = "val",
            val_ratio         = cfg.val_ratio,
            exclude_cams      = set(cfg.exclude_cams) if cfg.exclude_cams else set(),
            motion_cache_name = cfg.motion_cache_name,
            max_tracks        = cfg.max_tracks_per_sample,
        )
    if cfg.limit_scenes > 0:
        val_ds.entries = val_ds.entries[:cfg.limit_scenes]
    print(f"[data] val scenes = {len(val_ds)}  (dataset_type={cfg.dataset_type})")

    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # On real_data: model was trained on sim tracks; we have no real tracks,
    # and ablation isn't meaningful. Always disable tracks at forward.
    force_disable = (cfg.dataset_type == "real")
    do_ablation   = (not cfg.no_ablation) and (not force_disable)

    rows = []
    for i, batch in enumerate(loader):
        try:
            res = evaluate_one(model, batch, device,
                               run_ablation=do_ablation,
                               force_disable_tracks=force_disable,
                               use_bf16=cfg.use_bf16)
        except Exception as e:
            import traceback
            print(f"[scene {i}] error: {e}")
            traceback.print_exc()
            continue
        rows.append({
            "scene_id":          res["scene_id"],
            "assign_iou":        res["assign_iou"],
            "assign_iou_ablate": res["assign_iou_ablate"],
            "fg_iou":            res["fg_iou"],
            "fg_iou_ablate":     res["fg_iou_ablate"],
            "mIoU_alpha":        res["mIoU_alpha"],
            "mIoU_alpha_ablate": res["mIoU_alpha_ablate"],
            "iou_t0":            res["iou_t0"],
            "iou_tpos":          res["iou_tpos"],
            "track_acc":         res["track_acc"],
            "n_vis_tracks":      res["n_vis_tracks"],
            "n_correct_tracks":  res["n_correct_tracks"],
            "kin":               res["kin"],
            "iou_per_part":      res["iou_per_part"],
        })
        if i < cfg.num_vis_scenes:
            scene_dir = out_root / "scenes" / res["scene_id"].replace("/", "_")
            try:
                visualise_scene(batch, res, scene_dir)
            except Exception as e:
                import traceback
                print(f"[viz {res['scene_id']}] error: {e}")
                traceback.print_exc()
        if (i + 1) % 10 == 0:
            ai = np.mean([r["assign_iou"] for r in rows])
            ma = np.mean([r["mIoU_alpha"] for r in rows])
            print(f"  [{i+1}/{len(loader)}] assign_iou={ai:.4f}  alpha_iou={ma:.4f}", flush=True)
        torch.cuda.empty_cache()

    # Save raw per-scene rows
    with open(out_root / "all_scenes_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Aggregate summary
    ai      = np.array([r["assign_iou"]        for r in rows]) if rows else None
    ai_abl  = np.array([r["assign_iou_ablate"] for r in rows]) if not cfg.no_ablation and rows else None
    fg      = np.array([r["fg_iou"]            for r in rows]) if rows else None
    fg_abl  = np.array([r["fg_iou_ablate"]     for r in rows]) if not cfg.no_ablation and rows else None
    al      = np.array([r["mIoU_alpha"]        for r in rows]) if rows else None
    al_abl  = np.array([r["mIoU_alpha_ablate"] for r in rows]) if not cfg.no_ablation and rows else None
    track_acc = np.array([r["track_acc"] for r in rows])

    i0_arr = np.array([r["iou_t0"]   for r in rows]) if rows else None
    ip_arr = np.array([r["iou_tpos"] for r in rows]) if rows else None

    def _m(a):  return float(a.mean())     if a is not None and len(a) else float("nan")
    def _md(a): return float(np.median(a)) if a is not None and len(a) else float("nan")

    summary = {
        "ckpt":           cfg.ckpt,
        "step":           int(step),
        "n_scenes":       len(rows),

        # ── PRIMARY: identity-free foreground IoU (cross-domain robust) ────
        "fg_iou_mean":           _m(fg),
        "fg_iou_median":         _md(fg),
        "fg_iou_ablate_mean":    _m(fg_abl),
        "track_gap_fg_mean":     _m(fg - fg_abl) if fg is not None and fg_abl is not None else None,

        # ── slot routing (Hungarian aligned per-part) ─────────────────────
        "assign_iou_mean":        _m(ai),
        "assign_iou_median":      _md(ai),
        "assign_iou_ablate_mean": _m(ai_abl),
        "track_gap_assign_mean":  _m(ai - ai_abl) if ai is not None and ai_abl is not None else None,

        # ── AUXILIARY: end-to-end alpha-projection (through GS pipeline) ──
        "mIoU_alpha_mean":         _m(al),
        "mIoU_alpha_ablate_mean":  _m(al_abl),
        "track_gap_alpha_mean":    _m(al - al_abl) if al is not None and al_abl is not None else None,

        "track_acc_mean": _m(track_acc),

        "iou_decomposition": {
            "assign_iou_mean":         _m(ai),
            "iou_t0_mean":             _m(i0_arr),
            "iou_tpos_mean":           _m(ip_arr),
            "alpha_all_mean":          _m(al),
            "drop_routing_to_GS_t0":   _m((ai - i0_arr)) if ai is not None and i0_arr is not None else float("nan"),
            "drop_GS_t0_to_motion":    _m((i0_arr - ip_arr)) if i0_arr is not None and ip_arr is not None else float("nan"),
        },
        "kin_mean": {
            "axis_cos":  float(np.nanmean([r["kin"]["axis_cos"]  for r in rows])) if rows else float("nan"),
            "pivot_l2":  float(np.nanmean([r["kin"]["pivot_l2"]  for r in rows])) if rows else float("nan"),
            "scalar_l1": float(np.nanmean([r["kin"]["scalar_l1"] for r in rows])) if rows else float("nan"),
            "type_acc":  float(np.nanmean([r["kin"]["type_acc"]  for r in rows])) if rows else float("nan"),
        },
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    plot_summary(rows, out_root / "summary")
    print(f"\n[done] outputs → {out_root}")


if __name__ == "__main__":
    main()
