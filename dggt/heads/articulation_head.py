"""
ArticulationHead — unified joint parameter decoder.

Replaces separate KinematicHead + DynamicsHead.

Per slot p, predicts one coherent articulation vector:
  bbox_center  [3]  : part bounding-box centre     ∈ [-r, r]³
  bbox_size    [3]  : part bounding-box half-extent ∈ (0, r]³   (half-size, not full)
  type_logits  [2]  : raw logits for [prismatic, revolute]
  axis         [3]  : unit rotation / translation direction
  pivot        [3]  : pivot point                  ∈ [-r, r]³
  scalars      [S]  : per-frame motion amount       ∈ [-1, 1]
                      (predicted by a timestamp-conditioned sub-MLP)

Total static output per slot: 3+3+2+3+3 = 14  (+ S via dynamics sub-MLP)

Slot 0 convention (static base):
  type_logits are still predicted but excluded from CE loss externally.
  bbox IS supervised (constrains static-part Gaussians).
  scalars output but forced to 0 in loss.

2-class vs 3-class change:
  Old code: 3 classes (static=0, prismatic=1, revolute=2)
  New code: 2 classes (prismatic=0, revolute=1)
  GT label mapping at loss time: GT=1 (prismatic) → 0, GT=2 (revolute) → 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dim_in: int, hidden: int, dim_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim_in),
        nn.Linear(dim_in, hidden),
        nn.GELU(),
        nn.Linear(hidden, dim_out),
    )


class ArticulationHead(nn.Module):
    """
    Unified articulation decoder.

    Args:
        dim_in:       slot feature dimension
        num_slots:    P (total slots, including static Slot 0)
        scene_radius: half-side of normalised scene bbox
        hidden_dim:   MLP hidden width for static outputs
        scalar_hidden: MLP hidden width for dynamics sub-MLP
    """

    def __init__(
        self,
        dim_in:        int   = 1024,
        num_slots:     int   = 8,
        scene_radius:  float = 1.0,
        hidden_dim:    int   = 256,
        scalar_hidden: int   = 256,
    ):
        super().__init__()
        self.scene_radius = scene_radius
        self.num_slots    = num_slots

        # Per-slot shared backbone (all slots share this)
        self.backbone = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Static output heads (operate on backbone output)
        self.bbox_head  = nn.Linear(hidden_dim, 6)          # center(3) + size(3)
        self.type_head  = nn.Linear(hidden_dim, 2)          # prismatic / revolute
        self.axis_head  = nn.Linear(hidden_dim, 3)
        self.pivot_head = nn.Linear(hidden_dim, 3)

        # Dynamics: scalar per (slot, frame) via timestamp conditioning
        # Input: slot backbone feature (hidden_dim) + timestamp scalar (1)
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, scalar_hidden),
            nn.GELU(),
            nn.Linear(scalar_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialise bbox_size bias so sigmoid(0)*2r = r → size starts at r/2
        # (neutral: half the scene radius as initial half-extent)
        # No special init needed since sigmoid(0) = 0.5 → size = 0.5*2r = r

    def forward(
        self,
        slot_features: torch.Tensor,   # [B, P, D]
        timestamps:    torch.Tensor,   # [B, S]  normalised to [0, 1]
    ) -> dict:
        """
        Returns:
            motion_type_logits : [B, P, 2]   raw logits [prismatic, revolute]
            axis               : [B, P, 3]   L2-normalised unit vectors
            pivot              : [B, P, 3]   ∈ [-r, r]³
            scalars            : [B, P, S]   ∈ [-1, 1]
            bbox_center        : [B, P, 3]   ∈ [-r, r]³
            bbox_size          : [B, P, 3]   ∈ (0, r]³  (half-extent)
        """
        B, P, D = slot_features.shape
        S = timestamps.shape[1]
        r = self.scene_radius

        # ── Shared backbone ───────────────────────────────────────────────
        # Reshape to [B*P, D] for batch-efficient MLP
        feat_flat = slot_features.reshape(B * P, D)
        h = self.backbone(feat_flat)                     # [B*P, hidden_dim]
        h_bp = h.reshape(B, P, -1)                      # [B, P, hidden_dim]

        # ── Static outputs ────────────────────────────────────────────────
        # BBox
        raw_bbox = self.bbox_head(h)                     # [B*P, 6]
        raw_bbox = raw_bbox.reshape(B, P, 6)
        bbox_center = 2.0 * r * torch.sigmoid(raw_bbox[..., :3]) - r   # ∈ [-r, r]³
        bbox_size   = r * torch.sigmoid(raw_bbox[..., 3:])               # ∈ (0, r]  (half-extent)

        # Motion type logits (2-class)
        motion_type_logits = self.type_head(h).reshape(B, P, 2)          # [B, P, 2]

        # Axis (L2 normalised)
        axis_raw = self.axis_head(h).reshape(B, P, 3)
        axis = F.normalize(axis_raw, dim=-1, eps=1e-4)                   # [B, P, 3]  eps=1e-4 safe for float16

        # Pivot
        pivot_raw = self.pivot_head(h).reshape(B, P, 3)
        pivot = 2.0 * r * torch.sigmoid(pivot_raw) - r                  # [B, P, 3]

        # ── Dynamic scalars (timestamp-conditioned) ───────────────────────
        # h_bp: [B, P, hidden_dim]
        # timestamps: [B, S]  → [B, 1, S, 1] → [B, P, S, 1]
        h_exp = h_bp.unsqueeze(2).expand(-1, -1, S, -1)                 # [B, P, S, hidden]
        ts_exp = timestamps.unsqueeze(1).unsqueeze(-1).expand(B, P, S, 1)
        inp_sc = torch.cat([h_exp, ts_exp], dim=-1)                     # [B, P, S, hidden+1]
        inp_sc_flat = inp_sc.reshape(B * P * S, -1)
        scalars = self.scalar_mlp(inp_sc_flat).reshape(B, P, S)         # [B, P, S]

        return {
            "motion_type_logits": motion_type_logits,   # [B, P, 2]
            "axis":               axis,                  # [B, P, 3]
            "pivot":              pivot,                 # [B, P, 3]
            "scalars":            scalars,               # [B, P, S]
            "bbox_center":        bbox_center,           # [B, P, 3]
            "bbox_size":          bbox_size,             # [B, P, 3]  half-extent
        }
