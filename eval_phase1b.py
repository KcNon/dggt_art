"""
eval_phase1b.py — Phase 1b evaluation for ArtVGGT

Metrics:
  - Mask IoU          : per matched (pred, GT) pair, argmax-binarised
  - Motion Type Acc   : 2-class (prismatic / revolute) on active foreground slots
  - |cos(axis)|       : absolute cosine similarity after 180° sign fix
  - Axis Angle Error  : arccos(|cos|) in degrees
  - Scalar MAE        : mean abs error of normalised motion scalars
  - Pivot L2          : L2 distance between predicted and GT pivot points

Key correctness fixes vs inference_art.py:
  1. Argmax binarisation for matching (not threshold=0.5, which breaks softmax outputs)
  2. Model outputs 2-class {0=prismatic, 1=revolute}; GT has 3-class {0=static,
     1=prismatic, 2=revolute}. Only active foreground parts (GT index ≥ 1, mask
     sum > 0) are evaluated for type accuracy.
  3. Val split uses same seed / val_ratio as training so scenes are not seen during train.

Usage:
  python eval_phase1b.py \
      --checkpoint /data2/cyt/checkpoints/art_v11_phase1/ckpt_002000.pth \
      --data_root  /data2/cyt/data_root \
      --output_dir ./eval_results/v11_step2000 \
      [--val_ratio 0.15] [--num_frames 4] [--gpu 0] [--max_vis 20]
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

from dggt.models.art_vggt import ArtVGGT
from datasets.articulated_dataset import ArticulatedDataset


# ============================================================================
# Slot colour palette (up to 8 slots)
# ============================================================================
SLOT_COLORS = np.array([
    [0.85, 0.85, 0.85],  # slot 0: static background — light grey
    [0.95, 0.20, 0.20],  # slot 1: red
    [0.20, 0.75, 0.20],  # slot 2: green
    [0.20, 0.40, 0.95],  # slot 3: blue
    [0.95, 0.70, 0.10],  # slot 4: orange
    [0.70, 0.20, 0.90],  # slot 5: purple
    [0.10, 0.85, 0.85],  # slot 6: cyan
    [0.95, 0.40, 0.70],  # slot 7: pink
], dtype=np.float32)

MOTION_NAMES = {0: "static", 1: "prismatic", 2: "revolute"}


# ============================================================================
# Hungarian matching (argmax-based, consistent with training)
# ============================================================================

@torch.no_grad()
def argmax_hungarian_match(
    pred_maps: torch.Tensor,   # [P, H, W]  soft assignment (softmax)
    gt_masks:  torch.Tensor,   # [P_gt, H, W]  binary
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match predicted slots → GT parts using argmax binarisation.

    Returns pred_idx [K], gt_idx [K] (K = number of active GT foreground parts).
    Slot 0 and zero-padded GT masks are excluded from foreground matching.
    """
    P     = pred_maps.shape[0]
    P_gt  = gt_masks.shape[0]
    device = pred_maps.device

    # Argmax: each pixel goes to its highest-probability slot
    pred_argmax = pred_maps.argmax(dim=0)                          # [H, W]
    pred_bin = (
        torch.arange(P, device=device).view(P, 1, 1) == pred_argmax.unsqueeze(0)
    ).float()                                                      # [P, H, W]

    # Build cost matrix (1 - IoU) for ALL slots × ALL GT masks
    cost_mat = torch.zeros(P, P_gt, device=device)
    for i in range(P):
        for j in range(P_gt):
            inter = (pred_bin[i] * gt_masks[j]).sum()
            union = (pred_bin[i] + gt_masks[j]).clamp(0, 1).sum()
            cost_mat[i, j] = 1.0 - inter / (union + 1e-6)

    # Active foreground GT indices: skip index 0 (background) and zero-padded
    gt_active = [j for j in range(1, P_gt) if gt_masks[j].sum() > 0]
    if not gt_active:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Only match slots 1..P-1 to active GT foreground masks
    cost_sub = cost_mat[1:, :][:, gt_active].cpu().numpy()
    row_sub, col_sub = linear_sum_assignment(cost_sub)
    pred_idx = row_sub + 1                              # back to 1-indexed
    gt_idx   = np.array(gt_active)[col_sub]
    return pred_idx, gt_idx


# ============================================================================
# Per-scene evaluation
# ============================================================================

@torch.no_grad()
def evaluate_scene(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
) -> dict:
    """
    Evaluate one scene. batch is a single un-batched sample from the dataset.

    Returns a dict of scalar metrics plus raw tensors (prefixed with '_').
    """
    images     = batch["images"].unsqueeze(0).to(device)      # [1, S, 3, H, W]
    extrinsics = batch["extrinsics"].unsqueeze(0).to(device)  # [1, S, 4, 4]
    intrinsics = batch["intrinsics"].unsqueeze(0).to(device)  # [1, 3, 3]
    timestamps = batch["timestamps"].unsqueeze(0).to(device)  # [1, S]
    part_masks = batch["part_masks"].to(device)               # [S, P_gt, H, W]
    gt_motion  = batch["gt_motion_type"].to(device)           # [P_gt]  {0,1,2}
    gt_axis    = batch["gt_axis"].to(device)                  # [P_gt, 3]
    gt_pivot   = batch["gt_pivot"].to(device)                 # [P_gt, 3]
    gt_scalars = batch["gt_scalars"].to(device)               # [P_gt, S]

    S, P_gt, H, W = part_masks.shape

    # Forward pass
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        preds = model(images, extrinsics, intrinsics, timestamps)

    assign_maps   = preds["assign_maps"][0]           # [P, H_p, W_p]
    motion_logits = preds["motion_type_logits"][0]    # [P, 2]
    axis_pred     = preds["axis"][0]                  # [P, 3]
    pivot_pred    = preds["pivot"][0]                 # [P, 3]
    scalars_pred  = preds["scalars"][0]               # [P, S_frames]

    # Upsample assign_maps → image resolution
    assign_up = F.interpolate(
        assign_maps.unsqueeze(0), (H, W), mode="bilinear", align_corners=False
    ).squeeze(0)                                      # [P, H, W]

    # Frame-0 GT masks (canonical rest state) — consistent with assign_maps.
    # assign_maps is defined at the rest pose, so matching against the rest-state
    # GT gives the correct slot→part correspondence without temporal ambiguity.
    gt_masks_bin = (part_masks[0] > 0.5).float()                 # [P_gt, H, W]

    # ── Hungarian matching (argmax, active foreground only) ────────────────
    pred_idx, gt_idx = argmax_hungarian_match(assign_up, gt_masks_bin)

    # ── Argmax binary prediction masks (for IoU) ──────────────────────────
    P = assign_maps.shape[0]
    pred_argmax = assign_up.argmax(dim=0)                          # [H, W]
    pred_bin = (
        torch.arange(P, device=device).view(P, 1, 1) == pred_argmax.unsqueeze(0)
    ).float()                                                      # [P, H, W]

    # ── Mask IoU ──────────────────────────────────────────────────────────
    ious = []
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        inter = (pred_bin[pi] * gt_masks_bin[gi]).sum().item()
        union = (pred_bin[pi] + gt_masks_bin[gi]).clamp(0, 1).sum().item()
        ious.append(inter / (union + 1e-6))
    mean_iou = float(np.mean(ious)) if ious else 0.0

    # ── Motion type accuracy ───────────────────────────────────────────────
    # Model: 2-class logits {0=prismatic, 1=revolute}
    # GT:    3-class        {0=static,    1=prismatic, 2=revolute}
    # Mapping: pred_class + 1 → GT class (for non-static slots)
    pred_class = motion_logits.argmax(dim=-1).cpu()   # [P]  values {0, 1}
    type_correct, type_total = 0, 0
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        gt_type = int(gt_motion[gi].item())
        if gt_type == 0:
            # Active GT foreground should not be static; skip if mislabeled
            continue
        pred_type_mapped = int(pred_class[pi].item()) + 1  # {0→1, 1→2}
        type_correct += int(pred_type_mapped == gt_type)
        type_total   += 1
    type_acc = type_correct / max(type_total, 1)

    # ── Axis metrics (skip static GT parts) ───────────────────────────────
    cos_sims, angle_errs = [], []
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        gt_type = int(gt_motion[gi].item())
        if gt_type == 0:
            continue  # static part: no axis defined
        # Absolute cosine (180° ambiguity handled by sign fix in training)
        cos = float(F.cosine_similarity(
            axis_pred[pi].unsqueeze(0),
            gt_axis[gi].unsqueeze(0),
        ).abs().clamp(0, 1))
        cos_sims.append(cos)
        angle_errs.append(math.degrees(math.acos(cos)))
    mean_axis_cos   = float(np.mean(cos_sims))   if cos_sims   else 0.0
    mean_angle_deg  = float(np.mean(angle_errs)) if angle_errs else 90.0

    # ── Scalar MAE (with sign fix) ─────────────────────────────────────────
    scalar_maes = []
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        gt_type = int(gt_motion[gi].item())
        if gt_type == 0:
            continue
        s_pred = scalars_pred[pi].cpu()               # [S_frames]
        s_gt   = gt_scalars[gi].cpu()                 # [S_frames]
        # Apply same sign fix as training
        dot = float((axis_pred[pi] * gt_axis[gi]).sum().item())
        if dot < 0:
            s_pred = -s_pred
        scalar_maes.append(float((s_pred - s_gt).abs().mean()))
    mean_scalar_mae = float(np.mean(scalar_maes)) if scalar_maes else 0.0

    # ── Pivot L2 ──────────────────────────────────────────────────────────
    pivot_l2s = []
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        gt_type = int(gt_motion[gi].item())
        if gt_type != 2:
            continue  # pivot only meaningful for revolute joints
        l2 = float((pivot_pred[pi] - gt_pivot[gi]).norm().item())
        pivot_l2s.append(l2)
    mean_pivot_l2 = float(np.mean(pivot_l2s)) if pivot_l2s else float("nan")

    # ── Pack tensors for optional visualisation ────────────────────────────
    n_matched = len(pred_idx)

    return {
        "scene_id":       batch["scene_id"],
        "n_matched":      n_matched,
        "mean_iou":       mean_iou,
        "type_acc":       type_acc,
        "axis_cos":       mean_axis_cos,
        "axis_angle_deg": mean_angle_deg,
        "scalar_mae":     mean_scalar_mae,
        "pivot_l2":       mean_pivot_l2,
        # raw tensors (excluded from JSON, used for vis)
        "_images":       batch["images"],               # [S, 3, H, W]
        "_assign_up":    assign_up.cpu(),               # [P, H, W]
        "_gt_masks_bin": gt_masks_bin.cpu(),            # [P_gt, H, W]
        "_pred_idx":     pred_idx,
        "_gt_idx":       gt_idx,
        "_pred_class":   pred_class.tolist(),
        "_gt_motion":    gt_motion.cpu().tolist(),
        "_axis_pred":    axis_pred.cpu(),               # [P, 3]
        "_axis_gt":      gt_axis.cpu(),                 # [P_gt, 3]
        "_scalars_pred": scalars_pred.cpu(),            # [P, S]
        "_scalars_gt":   gt_scalars.cpu(),              # [P_gt, S]
    }


# ============================================================================
# Visualisation helpers
# ============================================================================

def _make_colored_seg(assign_up: torch.Tensor) -> np.ndarray:
    """assign_up [P, H, W] → [H, W, 3] RGB float32 via argmax colouring."""
    P, H, W = assign_up.shape
    argmax = assign_up.argmax(dim=0).numpy()           # [H, W]
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for p in range(min(P, len(SLOT_COLORS))):
        mask = (argmax == p).astype(np.float32)
        canvas += mask[:, :, None] * SLOT_COLORS[p][None, None, :]
    return np.clip(canvas, 0, 1)


def _make_colored_gt(gt_masks_bin: torch.Tensor) -> np.ndarray:
    """gt_masks_bin [P_gt, H, W] → [H, W, 3] via per-part colouring."""
    P_gt, H, W = gt_masks_bin.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for g in range(min(P_gt, len(SLOT_COLORS))):
        m = gt_masks_bin[g].numpy()
        canvas += m[:, :, None] * SLOT_COLORS[g][None, None, :]
    return np.clip(canvas, 0, 1)


def save_scene_vis(result: dict, out_path: str, n_frames_show: int = 4):
    """Multi-panel figure: input frames | pred seg | GT seg | scalar trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return

    images       = result["_images"]                    # [S, 3, H, W]
    assign_up    = result["_assign_up"]                 # [P, H, W]
    gt_masks_bin = result["_gt_masks_bin"]              # [P_gt, H, W]
    pred_idx     = result["_pred_idx"]
    gt_idx       = result["_gt_idx"]
    scalars_pred = result["_scalars_pred"]              # [P, S]
    scalars_gt   = result["_scalars_gt"]                # [P_gt, S]
    pred_class   = result["_pred_class"]
    gt_motion    = result["_gt_motion"]
    axis_pred    = result["_axis_pred"]
    axis_gt      = result["_axis_gt"]

    S = images.shape[0]
    frame_ids = np.linspace(0, S - 1, min(n_frames_show, S), dtype=int).tolist()
    n_frames_show = len(frame_ids)
    n_active = len(pred_idx)

    n_rows = 3 + max(n_active, 1)
    fig = plt.figure(figsize=(4 * n_frames_show, 3 * n_rows), dpi=80)
    gs  = gridspec.GridSpec(n_rows, n_frames_show, figure=fig,
                            hspace=0.4, wspace=0.05)

    H, W = images.shape[-2:]
    seg_pred = _make_colored_seg(assign_up)
    seg_gt   = _make_colored_gt(gt_masks_bin)

    for fi, f in enumerate(frame_ids):
        img = images[f].permute(1, 2, 0).numpy()

        # Row 0: input image
        ax = fig.add_subplot(gs[0, fi])
        ax.imshow(img)
        ax.set_title(f"frame {f}", fontsize=8)
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("Input", fontsize=7)

        # Row 1: predicted segmentation
        ax = fig.add_subplot(gs[1, fi])
        ax.imshow(np.clip(0.5 * img + 0.5 * seg_pred, 0, 1))
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("Pred seg", fontsize=7)

        # Row 2: GT segmentation
        ax = fig.add_subplot(gs[2, fi])
        ax.imshow(np.clip(0.5 * img + 0.5 * seg_gt, 0, 1))
        ax.axis("off")
        if fi == 0:
            ax.set_ylabel("GT seg", fontsize=7)

    # Rows 3+: scalar trajectories per matched part
    S_scalars = scalars_pred.shape[-1]
    xs = np.arange(S_scalars)
    for k in range(n_active):
        pi, gi = int(pred_idx[k]), int(gt_idx[k])
        ax = fig.add_subplot(gs[3 + k, :])

        sp = scalars_pred[pi].numpy()
        sg = scalars_gt[gi].numpy()
        ax.plot(xs, sg, "b-o",  label="GT",   markersize=3, linewidth=1.5)
        ax.plot(xs, sp, "r--s", label="Pred", markersize=3, linewidth=1.5)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")

        gt_t    = MOTION_NAMES.get(gt_motion[gi], "?")
        pred_t  = MOTION_NAMES.get(pred_class[pi] + 1, "?")  # 0→prismatic, 1→revolute
        cos_val = float(F.cosine_similarity(
            axis_pred[pi].unsqueeze(0), axis_gt[gi].unsqueeze(0)
        ).abs())
        iou_val = result.get("mean_iou", 0.0)  # per-slot not tracked here

        ax.set_title(
            f"Part {k} (pred slot {pi} → GT {gi}) | "
            f"Type GT={gt_t} Pred={pred_t} | "
            f"|cos(axis)|={cos_val:.3f}",
            fontsize=8,
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("frame index", fontsize=7)
        ax.set_ylabel("scalar", fontsize=7)
        ax.grid(True, alpha=0.3)

    step_str = f"step {result.get('_step', '?')}"
    fig.suptitle(
        f"Scene: {result['scene_id']}  |  {step_str}  |  "
        f"IoU={result['mean_iou']:.3f}  TypeAcc={result['type_acc']:.3f}  "
        f"|cos|={result['axis_cos']:.3f}  ScalarMAE={result['scalar_mae']:.3f}",
        fontsize=9, fontweight="bold",
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=80)
    plt.close(fig)


# ============================================================================
# Summary histogram
# ============================================================================

def save_summary_plot(all_results: list, step: int, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = {
        "Mask IoU":           [r["mean_iou"]       for r in all_results],
        "Motion Type Acc":    [r["type_acc"]        for r in all_results],
        "|cos(axis)|":        [r["axis_cos"]        for r in all_results],
        "Axis Angle (deg)":   [r["axis_angle_deg"]  for r in all_results],
        "Scalar MAE":         [r["scalar_mae"]      for r in all_results],
    }
    colors = ["steelblue", "seagreen", "darkorange", "tomato", "mediumpurple"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (title, vals), color in zip(axes, metrics.items(), colors):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            ax.set_title(title)
            continue
        mean_v = float(np.mean(vals))
        ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(mean_v, color="red", linestyle="--",
                   label=f"mean={mean_v:.3f}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Phase 1b Eval @ step {step}  ({len(all_results)} val scenes)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"Loading: {cfg.checkpoint}")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")
    saved_cfg   = ckpt.get("cfg", {})
    step        = ckpt.get("step", 0)
    n_gaussians = saved_cfg.get("n_gaussians", 256)
    img_size    = saved_cfg.get("img_size", 518)
    scene_radius = saved_cfg.get("scene_radius", 1.0)
    print(f"  Checkpoint step: {step}")

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
        print(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Dataset (val split only, same split as training) ──────────────────
    full_ds = ArticulatedDataset(
        data_root   = cfg.data_root,
        target_size = img_size,
        num_frames  = cfg.num_frames,
        max_parts   = 8,
    )
    n_total = len(full_ds)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total).tolist()
    n_val = max(1, int(n_total * cfg.val_ratio))
    val_indices = indices[:n_val]

    if cfg.num_scenes > 0:
        val_indices = val_indices[:cfg.num_scenes]
    val_ds = Subset(full_ds, val_indices)
    print(f"  Val scenes: {len(val_ds)} / {n_total} total")

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_results = []
    for i, batch in enumerate(val_ds):
        scene_id = batch["scene_id"]
        print(f"[{i+1:3d}/{len(val_ds)}] {scene_id}  ", end="", flush=True)

        try:
            result = evaluate_scene(model, batch, device)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        result["_step"] = step
        print(
            f"IoU={result['mean_iou']:.3f}  "
            f"TypeAcc={result['type_acc']:.3f}  "
            f"|cos|={result['axis_cos']:.3f}  "
            f"AxisErr={result['axis_angle_deg']:.1f}°  "
            f"ScalarMAE={result['scalar_mae']:.3f}  "
            f"PivotL2={result['pivot_l2']:.3f}"
        )

        all_results.append({k: v for k, v in result.items() if not k.startswith("_")})

        # Visualise first max_vis scenes
        if i < cfg.max_vis:
            safe_id = scene_id.replace("/", "_").replace(os.sep, "_")
            vis_path = str(vis_dir / f"{safe_id}.png")
            save_scene_vis(result, vis_path)

    if not all_results:
        print("No results collected.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────
    def _mean(key):
        vals = [r[key] for r in all_results if not math.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "mean_iou":       _mean("mean_iou"),
        "type_acc":       _mean("type_acc"),
        "axis_cos":       _mean("axis_cos"),
        "axis_angle_deg": _mean("axis_angle_deg"),
        "scalar_mae":     _mean("scalar_mae"),
        "pivot_l2":       _mean("pivot_l2"),
    }

    print("\n" + "=" * 60)
    print(f"  Phase 1b Eval @ step {step}  ({len(all_results)} scenes)")
    print("-" * 60)
    print(f"  {'Mask IoU':<28}  {summary['mean_iou']:>7.4f}")
    print(f"  {'Motion Type Accuracy':<28}  {summary['type_acc']:>7.4f}")
    print(f"  {'|cos(axis)|':<28}  {summary['axis_cos']:>7.4f}")
    print(f"  {'Axis Angle Error (deg)':<28}  {summary['axis_angle_deg']:>7.2f}")
    print(f"  {'Scalar MAE':<28}  {summary['scalar_mae']:>7.4f}")
    print(f"  {'Pivot L2 (revolute only)':<28}  {summary['pivot_l2']:>7.4f}")
    print("=" * 60)

    # ── Save JSON ──────────────────────────────────────────────────────────
    report = {
        "checkpoint":  cfg.checkpoint,
        "step":        step,
        "n_scenes":    len(all_results),
        "summary":     summary,
        "per_scene":   all_results,
    }
    report_path = out_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if math.isnan(x) else x)
    print(f"\nReport: {report_path}")

    # ── Summary histogram ──────────────────────────────────────────────────
    save_summary_plot(all_results, step, str(out_dir / "summary.png"))
    if cfg.max_vis > 0:
        print(f"Visualisations: {vis_dir}/")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ArtVGGT Phase 1b evaluation")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to .pth checkpoint")
    p.add_argument("--data_root",   required=True,
                   help="Dataset root (same as --data_root in train_art.py)")
    p.add_argument("--output_dir",  required=True,
                   help="Directory to write results.json, summary.png, vis/")
    p.add_argument("--val_ratio",   type=float, default=0.15,
                   help="Fraction of scenes used for validation (must match training)")
    p.add_argument("--num_frames",  type=int,   default=4,
                   help="Frames per scene (should match checkpoint cfg)")
    p.add_argument("--num_scenes",  type=int,   default=0,
                   help="Limit number of val scenes (0 = all)")
    p.add_argument("--gpu",         type=int,   default=0)
    p.add_argument("--max_vis",     type=int,   default=20,
                   help="Save visualisations for first N scenes (0 = none)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
