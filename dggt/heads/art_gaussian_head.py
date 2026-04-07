"""
ArtGaussianHead — decodes canonical-space 3D Gaussians from slot features.

Each slot gets an independent MLP that produces N_gaussians_per_slot
Gaussians in the canonical (object-centric rest-pose) coordinate frame.

Gaussian attributes per point:
  mu     [3]:  centre position
  rot    [4]:  rotation quaternion (w, x, y, z), L2-normalised
  scale  [3]:  log-scale → exp → isotropic-ish scale
  color  [3]:  RGB ∈ [0, 1] via sigmoid
  opacity[1]:  ∈ (0, 1) via sigmoid

GS_DIM = 3 + 4 + 3 + 3 + 1 = 14

Design note (per discussion):
  Pure slot-feature MLP decoding. Pixel-aligned upsampling (PixelSplat-style)
  is a future upgrade path but not implemented here for Phase 1 simplicity.
  N_gaussians_per_slot defaults to 256; the density adapts implicitly through
  opacity — dead Gaussians from small/absent parts converge to near-zero opacity
  under the dead-slot opacity loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


GS_DIM = 14    # mu(3) + rot(4) + scale(3) + color(3) + opacity(1)

# Indices into the raw GS output vector
_IDX_MU      = slice(0, 3)
_IDX_ROT     = slice(3, 7)
_IDX_SCALE   = slice(7, 10)
_IDX_COLOR   = slice(10, 13)
_IDX_OPACITY = slice(13, 14)


class SlotGaussianMLP(nn.Module):
    """Independent MLP for one slot."""

    def __init__(self, dim_in: int, hidden_dim: int, n_gaussians: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, n_gaussians * GS_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]  one slot's features
        Returns:
            [B, N_g, GS_DIM]  raw GS attributes (pre-activation)
        """
        B = x.shape[0]
        raw = self.mlp(x)                        # [B, N_g * GS_DIM]
        return raw.reshape(B, self.n_gaussians, GS_DIM)


class ArtGaussianHead(nn.Module):
    """
    Canonical 3D Gaussian decoder.

    One independent MLP per slot — each slot specialises independently.
    NOTE: weights are NOT shared across slots by design.

    Args:
        dim_in:          slot feature dimension
        num_slots:       number of slots (P)
        n_gaussians:     Gaussians per slot
        hidden_dim:      MLP hidden width
        scale_init_log:  initial log-scale bias (exp(scale_init_log) ≈ initial GS size)
        scene_radius:    canonical bbox half-side (positions in [-r, r])
    """

    def __init__(
        self,
        dim_in: int = 1024,
        num_slots: int = 8,
        n_gaussians: int = 256,
        hidden_dim: int = 512,
        scale_init_log: float = -4.0,
        scene_radius: float = 1.0,
    ):
        super().__init__()
        self.dim_in       = dim_in
        self.num_slots    = num_slots
        self.n_gaussians  = n_gaussians
        self.scene_radius = scene_radius
        self.scale_init_log = scale_init_log

        self.slot_mlps = nn.ModuleList([
            SlotGaussianMLP(dim_in, hidden_dim, n_gaussians)
            for _ in range(num_slots)
        ])

        # Bias the final linear layer to encourage small initial scale
        for mlp in self.slot_mlps:
            last_linear = mlp.mlp[-1]
            with torch.no_grad():
                # Scale bias: initialise to scale_init_log so exp(bias) ≈ small
                last_linear.bias[_IDX_SCALE].fill_(scale_init_log)
                # Opacity bias: initialise to -2 → sigmoid ≈ 0.12 (slightly transparent)
                last_linear.bias[_IDX_OPACITY].fill_(-2.0)

    def forward(
        self,
        slot_features: torch.Tensor,   # [B, P, D]
        bbox_center: torch.Tensor | None = None,   # [B, P, 3]
        bbox_size:   torch.Tensor | None = None,   # [B, P, 3]  half-extent
    ) -> dict:
        """
        Decode canonical-space Gaussians for all slots.

        Returns dict with keys:
            mu      [B, P, N_g, 3]   position ∈ [-scene_radius, scene_radius]
            rot     [B, P, N_g, 4]   unit quaternion (w,x,y,z)
            scale   [B, P, N_g, 3]   positive scale (via exp)
            color   [B, P, N_g, 3]   RGB ∈ [0, 1]
            opacity [B, P, N_g, 1]   ∈ (0, 1)
        """
        B, P, D = slot_features.shape
        assert P == self.num_slots, (
            f"Expected {self.num_slots} slots, got {P}"
        )

        raw_list = []
        for p in range(P):
            raw_p = self.slot_mlps[p](slot_features[:, p, :])  # [B, N_g, GS_DIM]
            raw_list.append(raw_p)

        raw = torch.stack(raw_list, dim=1)   # [B, P, N_g, GS_DIM]

        # Activations
        mu_raw = torch.tanh(raw[..., _IDX_MU])
        if bbox_center is not None and bbox_size is not None:
            # Constrain Gaussian centers within predicted bounding box for each slot
            # bbox_center: [B, P, 3], bbox_size: [B, P, 3] (half-extent)
            mu = bbox_center.unsqueeze(2) + mu_raw * bbox_size.unsqueeze(2)
            # mu ∈ [bbox_center - bbox_size, bbox_center + bbox_size]
        else:
            mu = mu_raw * self.scene_radius
        rot     = F.normalize(raw[..., _IDX_ROT], dim=-1, eps=1e-4)   # [B,P,N_g,4]  eps safe for float16
        scale   = torch.exp(raw[..., _IDX_SCALE]).clamp(min=1e-6)     # [B,P,N_g,3]
        color   = torch.sigmoid(raw[..., _IDX_COLOR])                  # [B,P,N_g,3]
        opacity = torch.sigmoid(raw[..., _IDX_OPACITY])                # [B,P,N_g,1]

        return {
            "mu":      mu,       # [B, P, N_g, 3]
            "rot":     rot,      # [B, P, N_g, 4]
            "scale":   scale,    # [B, P, N_g, 3]
            "color":   color,    # [B, P, N_g, 3]
            "opacity": opacity,  # [B, P, N_g, 1]
        }
