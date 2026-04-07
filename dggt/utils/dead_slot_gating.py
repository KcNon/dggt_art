"""
Dead Slot Gating — identifies and penalises empty/inactive slots.

Design rationale (finalised after discussion):
  - NO hard pixel-fraction threshold for gradient cutting.
  - L1 sparsity loss on Slot 1..7 (dynamic slots) is always active.
    Weight starts low (0.001 during warmup) and rises (0.1 after warmup).
  - Slot 0 (static background) is EXEMPT from sparsity: it should absorb
    all stationary pixels and is allowed unlimited coverage.
  - Dead-slot detection (boolean mask) is used ONLY for:
    (a) masking out kinematic / rendering losses of truly empty slots
    (b) forcing dead-slot Gaussian opacity to zero via a separate L1 term.
"""

import torch
import torch.nn.functional as F


def compute_slot_mass(assign_maps: torch.Tensor) -> torch.Tensor:
    """
    Total probability mass assigned to each slot.

    Args:
        assign_maps: [B, P, H_p, W_p]  softmax-normalised assignment maps
    Returns:
        slot_mass: [B, P]
    """
    return assign_maps.sum(dim=[-1, -2])  # [B, P]


def detect_dead_slots(
    assign_maps: torch.Tensor,
    threshold_fraction: float = 0.005,
) -> torch.Tensor:
    """
    Boolean dead-slot detector (used for loss masking only, NOT gradient gating).

    A slot is "dead" when its total probability mass is below
    `threshold_fraction` × (H_p × W_p).

    Args:
        assign_maps: [B, P, H_p, W_p]
        threshold_fraction: fraction below which a slot is considered dead
    Returns:
        is_dead: [B, P]  bool
    """
    B, P, H_p, W_p = assign_maps.shape
    slot_mass = compute_slot_mass(assign_maps)          # [B, P]
    threshold = H_p * W_p * threshold_fraction
    return slot_mass < threshold                        # [B, P]


def slot_sparsity_loss(
    assign_maps: torch.Tensor,
    l1_weight: float = 0.1,
) -> torch.Tensor:
    """
    L1 sparsity regularisation on *dynamic* slot assignment maps (Slot 1..7).

    Slot 0 (static background) is exempt — it should grow freely.

    The L1 acts on per-slot total mass so that truly unused slots collapse
    to zero while slots with legitimate parts keep their coverage because
    their rendering-loss gradient outweighs the L1 penalty.

    Args:
        assign_maps: [B, P, H_p, W_p]
        l1_weight:   regularisation coefficient
    Returns:
        scalar loss
    """
    dynamic_maps = assign_maps[:, 1:, :, :]       # [B, P-1, H_p, W_p]
    slot_mass = dynamic_maps.sum(dim=[-1, -2])     # [B, P-1]
    return l1_weight * slot_mass.mean()


def dead_slot_opacity_loss(
    opacities: torch.Tensor,    # [B, P, N_gaussians]
    is_dead: torch.Tensor,      # [B, P]  bool
    l1_weight: float = 0.1,
) -> torch.Tensor:
    """
    Force dead-slot Gaussians to near-zero opacity.

    Args:
        opacities: [B, P, N_gaussians]  Gaussian opacities after sigmoid
        is_dead:   [B, P]  bool dead-slot indicator
        l1_weight: coefficient
    Returns:
        scalar loss
    """
    if not is_dead.any():
        return opacities.new_zeros(1).squeeze()

    dead_mask = is_dead.float().unsqueeze(-1)      # [B, P, 1]
    return l1_weight * (opacities * dead_mask).mean()
