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

        # Single articulation vector head (paper Fig.2 / Eq.3-6):
        #   Â_p ∈ R^14 = [bbox_center(3) | bbox_size(3) | axis(3) | pivot(3) | type(2)]
        # partitioned along the channel axis and remapped in forward().
        # (The +T scalar part is produced by the timestamp-conditioned sub-MLP below.)
        self.art_head = nn.Linear(hidden_dim, 14)

        # Channel partition of the 14-dim articulation vector
        self._SL_BBOX_C = slice(0, 3)
        self._SL_BBOX_S = slice(3, 6)
        self._SL_AXIS   = slice(6, 9)
        self._SL_PIVOT  = slice(9, 12)
        self._SL_TYPE   = slice(12, 14)

        # Dynamics: scalar per (slot, frame) via timestamp conditioning.
        # Final 2*sigmoid-1 maps to [-1,1] (paper Eq.6: S_p = 2*ψ(Ŝ_p) - 1).
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, scalar_hidden),
            nn.GELU(),
            nn.Linear(scalar_hidden, 1),
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
                                 (Slot 0 is the static base; its logits are forced
                                  to static (prismatic ← -inf) so argmax/softmax keep
                                  it immobile — paper: 2-way class only on movable slots)
            axis               : [B, P, 3]   L2-normalised unit vectors      (Eq.4)
            pivot              : [B, P, 3]   ∈ [-r, r]³                       (Eq.5)
            scalars            : [B, P, S]   ∈ [-1, 1]  per-frame            (Eq.6)
            bbox_center        : [B, P, 3]   ∈ [-r, r]³                       (Eq.3)
            bbox_size          : [B, P, 3]   ∈ (0, 2r]³  (half-extent)        (Eq.3)
        """
        B, P, D = slot_features.shape
        S = timestamps.shape[1]
        r = self.scene_radius

        # ── Shared backbone ───────────────────────────────────────────────
        # Reshape to [B*P, D] for batch-efficient MLP
        feat_flat = slot_features.reshape(B * P, D)
        h = self.backbone(feat_flat)                     # [B*P, hidden_dim]
        h_bp = h.reshape(B, P, -1)                      # [B, P, hidden_dim]

        # ── Single articulation vector Â_p ∈ R^14, partition + remap ───────
        a = self.art_head(h).reshape(B, P, 14)           # [B, P, 14]

        bbox_center = 2.0 * r * torch.sigmoid(a[..., self._SL_BBOX_C]) - r   # ∈ [-r, r]³  (Eq.3)
        bbox_size   = 2.0 * r * torch.sigmoid(a[..., self._SL_BBOX_S])       # ∈ (0, 2r]³  (Eq.3)
        axis        = F.normalize(a[..., self._SL_AXIS], dim=-1, eps=1e-4)   # unit       (Eq.4)
        pivot       = 2.0 * r * torch.sigmoid(a[..., self._SL_PIVOT]) - r    # ∈ [-r, r]³  (Eq.5)
        motion_type_logits = a[..., self._SL_TYPE].contiguous()             # [B, P, 2]

        # Slot 0 = static base (paper convention): 2-way movable classification
        # applies only to Slots 1..P-1. Slot 0's logits are detached to a neutral
        # constant; static-ness is enforced where motion is applied (loss sets
        # motion_probs[:,0]=[1,0,0]; SDF renderer never transforms Slot 0).
        motion_type_logits = motion_type_logits.clone()
        motion_type_logits[:, 0, :] = 0.0

        # ── Dynamic scalars (timestamp-conditioned, Eq.6: S_p = 2ψ-1) ──────
        # h_bp: [B, P, hidden_dim]; timestamps: [B, S] → [B, P, S, 1]
        h_exp = h_bp.unsqueeze(2).expand(-1, -1, S, -1)                 # [B, P, S, hidden]
        ts_exp = timestamps.unsqueeze(1).unsqueeze(-1).expand(B, P, S, 1)
        inp_sc = torch.cat([h_exp, ts_exp], dim=-1)                     # [B, P, S, hidden+1]
        inp_sc_flat = inp_sc.reshape(B * P * S, -1)
        scalars = 2.0 * torch.sigmoid(self.scalar_mlp(inp_sc_flat)) - 1.0
        scalars = scalars.reshape(B, P, S)                              # [B, P, S]

        return {
            "motion_type_logits": motion_type_logits,   # [B, P, 2]
            "axis":               axis,                  # [B, P, 3]
            "pivot":              pivot,                 # [B, P, 3]
            "scalars":            scalars,               # [B, P, S]
            "bbox_center":        bbox_center,           # [B, P, 3]
            "bbox_size":          bbox_size,             # [B, P, 3]  half-extent
        }
