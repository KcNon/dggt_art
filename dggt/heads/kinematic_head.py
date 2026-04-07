"""
KinematicHead — predicts per-slot joint parameters.

Per slot p, outputs:
  motion_type_logits [B, P, 3]   — raw logits for [static, prismatic, revolute]
  axis               [B, P, 3]   — unit rotation/translation axis (L2-normalised)
  pivot              [B, P, 3]   — pivot point (sigmoid → mapped to scene bbox)

Constraints:
  • Slot 0 is FORCED to "static" by zeroing its logits before the CE loss.
    The logits are still produced (no gradient blocking) so that the network
    does not learn a degenerate mapping; the loss simply has no gradient for
    Slot 0's type head.
  • Dead slots have their kinematic losses masked out (handled externally
    by the training loop using is_dead flags).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KinematicHead(nn.Module):
    """
    Three independent linear decoders on top of slot features.

    Args:
        dim_in:    slot feature dimension (D_slot)
        scene_radius: half-side of the normalised scene bounding box
                      pivot outputs are in [-scene_radius, scene_radius]
    """

    def __init__(self, dim_in: int = 1024, scene_radius: float = 1.0):
        super().__init__()
        self.scene_radius = scene_radius

        # Motion type head: raw logits, 3 classes
        self.type_head = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_in // 4),
            nn.GELU(),
            nn.Linear(dim_in // 4, 3),
        )

        # Axis head: raw 3-vector → L2-normalise
        self.axis_head = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_in // 4),
            nn.GELU(),
            nn.Linear(dim_in // 4, 3),
        )

        # Pivot head: raw 3-vector → sigmoid → scale to bbox
        self.pivot_head = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, dim_in // 4),
            nn.GELU(),
            nn.Linear(dim_in // 4, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        slot_features: torch.Tensor,   # [B, P, D]
    ) -> dict:
        """
        Returns:
            motion_type_logits: [B, P, 3]   (CE loss; Slot 0 forced to 0 externally)
            axis:               [B, P, 3]   (unit vectors)
            pivot:              [B, P, 3]   (in [-scene_radius, scene_radius])
        """
        B, P, D = slot_features.shape

        # Motion type
        motion_type_logits = self.type_head(slot_features)     # [B, P, 3]

        # Force Slot 0 to static: override its logit so softmax → [1, 0, 0]
        # We do NOT stop gradient — we just manually set static logit >> others.
        static_override = torch.full(
            (B, 1, 3), fill_value=-1e4,
            device=slot_features.device, dtype=slot_features.dtype,
        )
        static_override[:, :, 0] = 1e4
        motion_type_logits = torch.cat(
            [static_override, motion_type_logits[:, 1:, :]], dim=1
        )   # [B, P, 3]

        # Axis (L2-normalised, eps prevents zero-vector collapse)
        axis_raw = self.axis_head(slot_features)               # [B, P, 3]
        axis = F.normalize(axis_raw, dim=-1, eps=1e-6)         # [B, P, 3]

        # Pivot (sigmoid maps to [0,1] then scale to [-r, r])
        pivot_raw = self.pivot_head(slot_features)             # [B, P, 3]
        pivot = 2.0 * self.scene_radius * torch.sigmoid(pivot_raw) - self.scene_radius
        # [B, P, 3]  ∈ [-scene_radius, scene_radius]

        return {
            "motion_type_logits": motion_type_logits,   # [B, P, 3]
            "axis":               axis,                  # [B, P, 3]
            "pivot":              pivot,                 # [B, P, 3]
        }
