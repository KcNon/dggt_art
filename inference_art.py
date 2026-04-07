"""
inference_art.py — 测试 ArtVGGT 训练权重

功能：
  1. 加载训练好的 checkpoint
  2. 在测试集上前向推断
  3. 计算定量指标：Mask IoU、Motion Type Acc、Axis Cosine Sim、Scalar MAE
  4. 保存可视化：分割图叠加、指标曲线、每场景详细报告

用法：
  python inference_art.py \
      --checkpoint /data2/cyt/checkpoints/art_phase1a/ckpt_001500.pth \
      --data_root  /data2/cyt/data_root \
      --output_dir ./results \
      [--num_scenes 20] [--num_frames 4] [--gpu 0]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from dggt.models.art_vggt import ArtVGGT
from datasets.articulated_dataset import ArticulatedDataset


# ============================================================================
# 颜色调色板：最多 8 个 slot
# ============================================================================
SLOT_COLORS = np.array([
    [0.9, 0.9, 0.9],   # Slot 0: 静态背景 — 浅灰
    [0.95, 0.2, 0.2],  # Slot 1: 红
    [0.2, 0.75, 0.2],  # Slot 2: 绿
    [0.2, 0.4, 0.95],  # Slot 3: 蓝
    [0.95, 0.7, 0.1],  # Slot 4: 橙
    [0.7, 0.2, 0.9],   # Slot 5: 紫
    [0.1, 0.85, 0.85], # Slot 6: 青
    [0.95, 0.4, 0.7],  # Slot 7: 粉
], dtype=np.float32)

MOTION_TYPE_NAMES = ["static", "prismatic", "revolute"]


# ============================================================================
# Matching helpers
# ============================================================================

@torch.no_grad()
def hungarian_match(pred_maps, gt_masks):
    """
    pred_maps: [P, H, W] float in [0,1]
    gt_masks:  [P_gt, H, W] binary
    Returns: pred_idx [K], gt_idx [K]
    """
    P     = pred_maps.shape[0]
    P_gt  = gt_masks.shape[0]
    pred_bin = (pred_maps > 0.5).float()
    cost = np.zeros((P, P_gt), dtype=np.float32)
    for i in range(P):
        for j in range(P_gt):
            inter = (pred_bin[i] * gt_masks[j]).sum().item()
            union = (pred_bin[i] + gt_masks[j]).clamp(0, 1).sum().item()
            cost[i, j] = 1.0 - inter / (union + 1e-6)
    row, col = linear_sum_assignment(cost)
    return row, col, cost


def compute_iou(pred_bin, gt_mask):
    """pred_bin, gt_mask: [H, W] tensors."""
    inter = (pred_bin * gt_mask).sum().item()
    union = (pred_bin + gt_mask).clamp(0, 1).sum().item()
    return inter / (union + 1e-6)


# ============================================================================
# Visualization helpers
# ============================================================================

def make_colored_mask(assign_maps_up, alpha=0.55):
    """
    assign_maps_up: [P, H, W]  (upsampled, soft assignments)
    Returns: [H, W, 3] RGB float32 composite overlay
    """
    P, H, W = assign_maps_up.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for p in range(min(P, len(SLOT_COLORS))):
        m = assign_maps_up[p].cpu().numpy()              # [H, W]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)  # normalize
        color = SLOT_COLORS[p]
        canvas += m[:, :, None] * color[None, None, :]
    canvas = np.clip(canvas, 0, 1)
    return canvas


def blend_image_mask(image_np, mask_colored, alpha=0.5):
    """
    image_np:     [H, W, 3] float32 [0,1]
    mask_colored: [H, W, 3] float32 [0,1]
    """
    return np.clip((1 - alpha) * image_np + alpha * mask_colored, 0, 1)


def save_scene_figure(
    scene_id, images, assign_maps_up, gt_masks_avg,
    frame_ids_to_show, pred_idx, gt_idx, iou_per_slot,
    motion_pred, motion_gt, axis_pred, axis_gt,
    scalar_pred, scalar_gt, n_active,
    out_path
):
    """Save a multi-panel figure for one scene."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return   # matplotlib not available

    n_frames_show = len(frame_ids_to_show)
    n_rows = 3 + n_active   # image row + pred-mask row + GT-mask row + per-part rows
    fig = plt.figure(figsize=(4 * n_frames_show, 3 * n_rows), dpi=80)
    gs  = gridspec.GridSpec(n_rows, n_frames_show, figure=fig,
                             hspace=0.35, wspace=0.05)

    H, W = images.shape[-2:]

    # ── Row 0: input images ───────────────────────────────────────────────
    for fi, f in enumerate(frame_ids_to_show):
        ax = fig.add_subplot(gs[0, fi])
        img_np = images[f].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(f"frame {f}", fontsize=8)
        ax.axis("off")

    # ── Row 1: predicted slot coloring ────────────────────────────────────
    for fi, f in enumerate(frame_ids_to_show):
        ax = fig.add_subplot(gs[1, fi])
        img_np = images[f].permute(1, 2, 0).cpu().numpy()
        mask_c = make_colored_mask(assign_maps_up)
        blended = blend_image_mask(img_np, mask_c)
        ax.imshow(blended)
        if fi == 0:
            ax.set_ylabel("Pred slots", fontsize=8)
        ax.axis("off")

    # ── Row 2: GT mask coloring ───────────────────────────────────────────
    for fi, f in enumerate(frame_ids_to_show):
        ax = fig.add_subplot(gs[2, fi])
        img_np = images[f].permute(1, 2, 0).cpu().numpy()
        # Build colored GT mask
        P_gt_show = gt_masks_avg.shape[0]
        gt_c = np.zeros((H, W, 3), dtype=np.float32)
        for g in range(min(P_gt_show, len(SLOT_COLORS))):
            m = gt_masks_avg[g].cpu().numpy()
            gt_c += m[:, :, None] * SLOT_COLORS[g][None, None, :]
        gt_c = np.clip(gt_c, 0, 1)
        blended = blend_image_mask(img_np, gt_c)
        ax.imshow(blended)
        if fi == 0:
            ax.set_ylabel("GT masks", fontsize=8)
        ax.axis("off")

    # ── Rows 3+: per-active-part scalar trajectory + kinematics info ─────
    for k in range(n_active):
        ax = fig.add_subplot(gs[3 + k, :])
        S = scalar_pred.shape[-1]
        xs = np.arange(S)

        p_idx = pred_idx[k] if k < len(pred_idx) else -1
        g_idx = gt_idx[k] if k < len(gt_idx) else k

        if p_idx >= 0:
            sp = scalar_pred[p_idx].cpu().numpy()
        else:
            sp = np.zeros(S)
        sg = scalar_gt[g_idx].cpu().numpy()

        ax.plot(xs, sg, "b-o", label="GT scalar", markersize=3, linewidth=1.5)
        ax.plot(xs, sp, "r--s", label="Pred scalar", markersize=3, linewidth=1.5)
        ax.set_ylim(-1.2, 1.2)

        # Kinematics annotation
        m_pred_str = MOTION_TYPE_NAMES[motion_pred[p_idx]] if p_idx >= 0 else "?"
        m_gt_str   = MOTION_TYPE_NAMES[motion_gt[g_idx]]
        ax_cos = float(F.cosine_similarity(
            axis_pred[p_idx].unsqueeze(0), axis_gt[g_idx].unsqueeze(0)
        ).abs()) if p_idx >= 0 else 0.0
        iou_val = iou_per_slot.get(k, 0.0)

        title = (f"Part {k} | IoU={iou_val:.3f} | "
                 f"Type GT={m_gt_str} Pred={m_pred_str} | "
                 f"|cos(axis)|={ax_cos:.3f}")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("frame", fontsize=7)
        ax.set_ylabel("scalar", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Scene: {scene_id}", fontsize=10, fontweight="bold")
    plt.savefig(out_path, bbox_inches="tight", dpi=80)
    plt.close(fig)


# ============================================================================
# Per-scene evaluation
# ============================================================================

@torch.no_grad()
def evaluate_scene(model, batch, device, num_frames_show=4):
    """
    Run model on one scene, compute metrics, return result dict.

    batch keys (single item, no extra batch dim from collate):
        images      [S, 3, H, W]
        extrinsics  [S, 4, 4]
        intrinsics  [3, 3]
        timestamps  [S]
        part_masks  [S, P, H, W]
        gt_motion_type [P]
        gt_axis     [P, 3]
        gt_pivot    [P, 3]
        gt_scalars  [P, S]
        n_active_parts int
        scene_id    str
    """
    images     = batch["images"].unsqueeze(0).to(device)       # [1, S, 3, H, W]
    extrinsics = batch["extrinsics"].unsqueeze(0).to(device)   # [1, S, 4, 4]
    intrinsics = batch["intrinsics"].unsqueeze(0).to(device)   # [1, 3, 3]
    timestamps = batch["timestamps"].unsqueeze(0).to(device)   # [1, S]
    part_masks = batch["part_masks"].to(device)                # [S, P, H, W]
    gt_motion  = batch["gt_motion_type"].to(device)            # [P]
    gt_axis    = batch["gt_axis"].to(device)                   # [P, 3]
    gt_scalars = batch["gt_scalars"].to(device)                # [P, S]
    n_active   = int(batch["n_active_parts"])
    scene_id   = batch["scene_id"]

    S, P_gt, H, W = part_masks.shape

    # Model forward
    preds = model(images, extrinsics, intrinsics, timestamps)

    assign_maps = preds["assign_maps"][0]    # [P, H_p, W_p]
    motion_logits = preds["motion_type_logits"][0]   # [P, 3]
    axis_pred = preds["axis"][0]             # [P, 3]
    scalars   = preds["scalars"][0]          # [P, S]

    # Upsample assign_maps to image resolution
    assign_up = F.interpolate(
        assign_maps.unsqueeze(0), (H, W),
        mode="bilinear", align_corners=False
    ).squeeze(0)   # [P, H, W]

    # GT masks averaged over frames
    gt_masks_avg = part_masks.float().mean(dim=0)   # [P_gt, H, W]

    # ── Hungarian matching ────────────────────────────────────────────────
    pred_idx, gt_idx, cost_mat = hungarian_match(assign_up, gt_masks_avg)

    # ── Per-slot IoU ──────────────────────────────────────────────────────
    iou_per_slot = {}
    ious = []
    for k, (pi, gi) in enumerate(zip(pred_idx, gt_idx)):
        pred_bin = (assign_up[pi] > 0.5).float()
        iou = compute_iou(pred_bin, gt_masks_avg[gi])
        iou_per_slot[k] = iou
        ious.append(iou)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    # ── Motion type accuracy ──────────────────────────────────────────────
    pred_types = motion_logits.argmax(dim=-1).cpu()   # [P]
    type_correct, type_total = 0, 0
    for pi, gi in zip(pred_idx, gt_idx):
        if gi < n_active:
            type_correct += int(pred_types[pi] == gt_motion[gi].cpu())
            type_total += 1
    type_acc = type_correct / max(type_total, 1)

    # ── Axis cosine similarity (active slots only) ────────────────────────
    cos_sims = []
    for pi, gi in zip(pred_idx, gt_idx):
        if gi < n_active and gt_motion[gi] > 0:  # skip static
            cos = float(F.cosine_similarity(
                axis_pred[pi].unsqueeze(0),
                gt_axis[gi].unsqueeze(0)
            ).abs())
            cos_sims.append(cos)
    mean_axis_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

    # ── Scalar MAE ────────────────────────────────────────────────────────
    maes = []
    for pi, gi in zip(pred_idx, gt_idx):
        if gi < n_active and gt_motion[gi] > 0:
            # Sign fix: if dot(pred_axis, gt_axis) < 0, flip scalar
            dot_val = float((axis_pred[pi] * gt_axis[gi]).sum())
            s_pred = scalars[pi].cpu()
            s_gt   = gt_scalars[gi].cpu()
            if dot_val < 0:
                s_pred = -s_pred
            mae = float((s_pred - s_gt).abs().mean())
            maes.append(mae)
    mean_scalar_mae = float(np.mean(maes)) if maes else 0.0

    return {
        "scene_id":       scene_id,
        "n_active":       n_active,
        "mean_iou":       mean_iou,
        "type_acc":       type_acc,
        "axis_cos":       mean_axis_cos,
        "scalar_mae":     mean_scalar_mae,
        # tensors for visualization
        "_images":        batch["images"],      # [S, 3, H, W]
        "_assign_up":     assign_up.cpu(),      # [P, H, W]
        "_gt_masks_avg":  gt_masks_avg.cpu(),   # [P_gt, H, W]
        "_pred_idx":      pred_idx,
        "_gt_idx":        gt_idx,
        "_iou_per_slot":  iou_per_slot,
        "_motion_pred":   pred_types.tolist(),
        "_motion_gt":     gt_motion.cpu().tolist(),
        "_axis_pred":     axis_pred.cpu(),
        "_axis_gt":       gt_axis.cpu(),
        "_scalar_pred":   scalars.cpu(),
        "_scalar_gt":     gt_scalars.cpu(),
    }


# ============================================================================
# Main
# ============================================================================

def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {cfg.checkpoint}")
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg", {})

    img_size    = saved_cfg.get("img_size", 518)
    n_gaussians = saved_cfg.get("n_gaussians", 256)
    scene_radius = saved_cfg.get("scene_radius", 1.0)
    step        = ckpt.get("step", 0)
    print(f"  Checkpoint step: {step}")

    # ── Build model ───────────────────────────────────────────────────────
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

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = ArticulatedDataset(
        data_root   = cfg.data_root,
        target_size = img_size,
        num_frames  = cfg.num_frames,
        max_parts   = 8,
    )
    n_scenes = min(cfg.num_scenes, len(dataset))
    print(f"  Dataset: {len(dataset)} scenes total, evaluating {n_scenes}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_results = []
    scene_indices = list(range(n_scenes))

    for i, idx in enumerate(scene_indices):
        batch = dataset[idx]
        print(f"[{i+1:3d}/{n_scenes}] scene={batch['scene_id']} ", end="", flush=True)

        try:
            result = evaluate_scene(model, batch, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(
            f"IoU={result['mean_iou']:.3f}  "
            f"TypeAcc={result['type_acc']:.3f}  "
            f"|cos(axis)|={result['axis_cos']:.3f}  "
            f"ScalarMAE={result['scalar_mae']:.3f}"
        )

        all_results.append({
            k: v for k, v in result.items() if not k.startswith("_")
        })

        # ── Save visualization ─────────────────────────────────────────
        if i < cfg.max_vis:
            n_frames = result["_images"].shape[0]
            frames_to_show = np.linspace(
                0, n_frames - 1,
                min(cfg.vis_frames, n_frames), dtype=int
            ).tolist()
            vis_path = vis_dir / f"{batch['scene_id']}.png"
            save_scene_figure(
                scene_id     = result["scene_id"],
                images       = result["_images"],
                assign_maps_up = result["_assign_up"],
                gt_masks_avg = result["_gt_masks_avg"],
                frame_ids_to_show = frames_to_show,
                pred_idx     = result["_pred_idx"],
                gt_idx       = result["_gt_idx"],
                iou_per_slot = result["_iou_per_slot"],
                motion_pred  = result["_motion_pred"],
                motion_gt    = result["_motion_gt"],
                axis_pred    = result["_axis_pred"],
                axis_gt      = result["_axis_gt"],
                scalar_pred  = result["_scalar_pred"],
                scalar_gt    = result["_scalar_gt"],
                n_active     = result["n_active"],
                out_path     = str(vis_path),
            )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    if not all_results:
        print("No results collected.")
        return

    mean_iou     = np.mean([r["mean_iou"]     for r in all_results])
    mean_type    = np.mean([r["type_acc"]      for r in all_results])
    mean_axis    = np.mean([r["axis_cos"]      for r in all_results])
    mean_scalar  = np.mean([r["scalar_mae"]    for r in all_results])

    print("\n" + "=" * 60)
    print(f"{'Metric':<30}  {'Value':>8}")
    print("-" * 40)
    print(f"{'Scenes evaluated':<30}  {len(all_results):>8d}")
    print(f"{'Mean Mask IoU':<30}  {mean_iou:>8.4f}")
    print(f"{'Motion Type Accuracy':<30}  {mean_type:>8.4f}")
    print(f"{'Mean |cos(axis)|':<30}  {mean_axis:>8.4f}")
    print(f"{'Scalar MAE':<30}  {mean_scalar:>8.4f}")
    print("=" * 60)

    # ── Save JSON report ──────────────────────────────────────────────────
    report = {
        "checkpoint":    cfg.checkpoint,
        "step":          step,
        "n_scenes":      len(all_results),
        "summary": {
            "mean_iou":    mean_iou,
            "type_acc":    mean_type,
            "axis_cos":    mean_axis,
            "scalar_mae":  mean_scalar,
        },
        "per_scene":     all_results,
    }
    report_path = out_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
    if cfg.max_vis > 0:
        print(f"Visualizations: {vis_dir}/")

    # ── Per-scene IoU histogram (if matplotlib available) ─────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ious = [r["mean_iou"] for r in all_results]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].hist(ious, bins=20, color="steelblue", edgecolor="white")
        axes[0].set_title("Mask IoU distribution")
        axes[0].set_xlabel("IoU")
        axes[0].axvline(mean_iou, color="red", linestyle="--",
                         label=f"mean={mean_iou:.3f}")
        axes[0].legend()

        type_accs = [r["type_acc"] for r in all_results]
        axes[1].hist(type_accs, bins=10, color="seagreen", edgecolor="white")
        axes[1].set_title("Motion Type Accuracy")
        axes[1].set_xlabel("Accuracy")
        axes[1].axvline(mean_type, color="red", linestyle="--",
                          label=f"mean={mean_type:.3f}")
        axes[1].legend()

        cos_vals = [r["axis_cos"] for r in all_results]
        axes[2].hist(cos_vals, bins=20, color="darkorange", edgecolor="white")
        axes[2].set_title("|cos(axis)| distribution")
        axes[2].set_xlabel("|cos similarity|")
        axes[2].axvline(mean_axis, color="red", linestyle="--",
                          label=f"mean={mean_axis:.3f}")
        axes[2].legend()

        maes = [r["scalar_mae"] for r in all_results]
        axes[3].hist(maes, bins=20, color="mediumpurple", edgecolor="white")
        axes[3].set_title("Scalar MAE distribution")
        axes[3].set_xlabel("MAE")
        axes[3].axvline(mean_scalar, color="red", linestyle="--",
                          label=f"mean={mean_scalar:.3f}")
        axes[3].legend()

        fig.suptitle(f"Eval @ step {step}  ({len(all_results)} scenes)", fontsize=12)
        plt.tight_layout()
        summary_path = out_dir / "summary.png"
        plt.savefig(summary_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Summary plot: {summary_path}")
    except ImportError:
        pass


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ArtVGGT inference / evaluation")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to .pth checkpoint")
    p.add_argument("--data_root",   required=True,
                   help="Root directory of test scenes")
    p.add_argument("--output_dir",  default="./results",
                   help="Where to save results and visualizations")
    p.add_argument("--num_scenes",  type=int, default=50,
                   help="Number of scenes to evaluate")
    p.add_argument("--num_frames",  type=int, default=4,
                   help="Frames per scene (should match training)")
    p.add_argument("--gpu",         type=int, default=0,
                   help="CUDA device index")
    p.add_argument("--max_vis",     type=int, default=20,
                   help="Max number of scenes to save visualizations for")
    p.add_argument("--vis_frames",  type=int, default=4,
                   help="How many frames to show per scene in visualization")
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
