"""
Hungarian matching (bipartite assignment) for slot ↔ GT-part alignment.

Called ONCE per training step, BEFORE any loss is computed, to resolve the
permutation ambiguity inherent in slot-based models.

Cost matrix:   C[p, g] = 1 - IoU(pred_mask_p, gt_mask_g)

After matching, also fixes the 180° sign ambiguity on predicted axes:
if dot(pred_axis_p, gt_axis_g) < 0  →  flip both pred_axis and pred_scalar.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match_masks(
    pred_maps: torch.Tensor,   # [P, H, W]  soft assignment (probs)
    gt_masks: torch.Tensor,    # [P_gt, H, W]  binary GT part masks
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimal slot → GT assignment for one sample.

    Args:
        pred_maps: [P, H, W]   predicted assignment maps (values in [0,1])
        gt_masks:  [P_gt, H, W] binary GT masks
    Returns:
        pred_idx: [K]  matched prediction indices  (K = min(P, P_gt))
        gt_idx:   [K]  matched GT indices
        cost_mat: [P, P_gt]  full IoU cost matrix
    """
    P     = pred_maps.shape[0]
    P_gt  = gt_masks.shape[0]
    device = pred_maps.device

    # Binarise predictions via argmax (each pixel → one slot).
    # threshold=0.5 breaks for softmax outputs where values average ~1/P.
    pred_argmax = pred_maps.argmax(dim=0)   # [H, W]  slot index per pixel
    pred_bin = (
        torch.arange(P, device=device).view(P, 1, 1) == pred_argmax.unsqueeze(0)
    ).float()   # [P, H, W]

    cost_mat = torch.zeros(P, P_gt, device=device)
    for i in range(P):
        for j in range(P_gt):
            inter = (pred_bin[i] * gt_masks[j]).sum()
            union = (pred_bin[i] + gt_masks[j]).clamp(0, 1).sum()
            iou   = inter / (union + 1e-6)
            cost_mat[i, j] = 1.0 - iou

    # Only match slots 1..P-1 (foreground) to ACTIVE FOREGROUND GT masks:
    #   - Skip GT index 0 (background/static-root; paired with pred slot 0 by design)
    #   - Skip zero GT masks (padding for objects with fewer joints than max_parts)
    gt_active = torch.tensor(
        [j for j in range(1, P_gt) if gt_masks[j].sum() > 0],
        dtype=torch.long, device=device,
    )

    if len(gt_active) == 0:
        # No active foreground GT masks — return empty matching
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, cost_mat

    cost_sub = cost_mat[1:, :][:, gt_active]   # [P-1, K_gt]  only active GT cols
    row_sub, col_sub = linear_sum_assignment(cost_sub.cpu().numpy())
    row_idx = row_sub + 1                       # 0-indexed in cost_sub → 1..P-1
    col_idx = gt_active[col_sub].cpu()          # map back to original GT indices

    return (
        torch.tensor(row_idx, dtype=torch.long, device=device),
        col_idx.to(device),
        cost_mat,
    )


@torch.no_grad()
def batch_hungarian_match(
    pred_maps: torch.Tensor,   # [B, P, H, W]
    gt_masks: torch.Tensor,    # [B, P_gt, H, W]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Per-sample Hungarian matching over a batch.

    Returns:
        List of (pred_idx, gt_idx) for each batch element.
    """
    B = pred_maps.shape[0]
    return [
        hungarian_match_masks(pred_maps[b], gt_masks[b])[:2]
        for b in range(B)
    ]


# ---------------------------------------------------------------------------
# Prediction reordering
# ---------------------------------------------------------------------------

def reorder_by_match(
    tensor: torch.Tensor,      # [B, P, ...]
    matches: list,             # list of (pred_idx, gt_idx) per sample
    P_gt: int,
) -> torch.Tensor:
    """
    Reorder slot-indexed predictions to align with GT part ordering.

    Unmatched slots are zero-filled.

    Args:
        tensor:  [B, P, ...]   any slot-indexed prediction
        matches: batch of (pred_idx, gt_idx) pairs
        P_gt:    number of GT parts
    Returns:
        [B, P_gt, ...]
    """
    B = tensor.shape[0]
    extra = tensor.shape[2:]
    out = tensor.new_zeros(B, P_gt, *extra)
    for b, (pred_idx, gt_idx) in enumerate(matches):
        out[b, gt_idx] = tensor[b, pred_idx]
    return out


# ---------------------------------------------------------------------------
# Axis sign-ambiguity fix
# ---------------------------------------------------------------------------

def fix_axis_sign_ambiguity(
    pred_axis: torch.Tensor,    # [B, P, 3]
    pred_scalar: torch.Tensor,  # [B, P, S]
    gt_axis: torch.Tensor,      # [B, P, 3]  (already aligned via match)
    confidence_threshold: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resolve 180° axis-direction ambiguity AFTER Hungarian matching.

    If predicted axis points into the opposite hemisphere from GT, flip both
    the axis and the scalar so that the loss is always computed in a consistent
    half-space, eliminating the bimodal gradient oscillation problem.

    A confidence threshold is applied: flipping is only performed when the dot
    product is sufficiently negative (|dot| > threshold AND dot < 0). When the
    dot product is near zero, the two axes are nearly perpendicular and the
    Hungarian match is unreliable — flipping in this case would inject
    random-direction gradients into scalar_mlp, causing the loss spikes
    observed in training (scalar MSE → 4.0, i.e. 4× worse than predicting zero).

    Args:
        pred_axis:            [B, P, 3]  predicted unit axes
        pred_scalar:          [B, P, S]  predicted motion scalars
        gt_axis:              [B, P, 3]  GT unit axes (aligned to pred ordering)
        confidence_threshold: only flip when dot < -threshold (default 0.3)
    Returns:
        fixed_axis:   [B, P, 3]
        fixed_scalar: [B, P, S]
    """
    dot  = (pred_axis * gt_axis).sum(dim=-1, keepdim=True)  # [B, P, 1]
    # Flip only when we are confident the axis is in the wrong hemisphere.
    flip = (dot < -confidence_threshold).float()             # 1 where confident flip needed

    fixed_axis   = pred_axis   * (1.0 - 2.0 * flip)         # [B, P, 3]
    fixed_scalar = pred_scalar * (1.0 - 2.0 * flip)         # [B, P, S]

    return fixed_axis, fixed_scalar


# ---------------------------------------------------------------------------
# Convenience: apply match + fix ambiguity in one call
# ---------------------------------------------------------------------------

def match_and_fix(
    pred_maps: torch.Tensor,    # [B, P, H, W]
    gt_masks: torch.Tensor,     # [B, P_gt, H, W]
    pred_axis: torch.Tensor,    # [B, P, 3]
    pred_scalar: torch.Tensor,  # [B, P, S]
    gt_axis: torch.Tensor,      # [B, P_gt, 3]
) -> dict:
    """
    Full matching pipeline:
      1. Hungarian match predict → GT
      2. Reorder all prediction tensors to GT ordering
      3. Fix axis sign ambiguity

    Returns dict with:
        matches:       list of (pred_idx, gt_idx)
        pred_axis:     [B, P_gt, 3]  reordered + sign-fixed
        pred_scalar:   [B, P_gt, S]  reordered + sign-fixed
    """
    P_gt = gt_masks.shape[1]

    # Step 1: Hungarian match
    upsample_pred = F.interpolate(
        pred_maps, (gt_masks.shape[-2], gt_masks.shape[-1]),
        mode="bilinear", align_corners=False,
    )
    matches = batch_hungarian_match(upsample_pred, gt_masks)

    # Step 2: reorder axis + scalar
    axis_reordered   = reorder_by_match(pred_axis,   matches, P_gt)  # [B, P_gt, 3]
    scalar_reordered = reorder_by_match(pred_scalar, matches, P_gt)  # [B, P_gt, S]
    gt_axis_reordered = gt_axis                                        # already in GT order

    # Step 3: fix sign
    axis_fixed, scalar_fixed = fix_axis_sign_ambiguity(
        axis_reordered, scalar_reordered, gt_axis_reordered
    )

    return {
        "matches":      matches,
        "pred_axis":    axis_fixed,
        "pred_scalar":  scalar_fixed,
    }
