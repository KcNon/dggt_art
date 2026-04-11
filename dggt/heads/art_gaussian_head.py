"""
ArtGaussianHead — decodes canonical-space 3D Gaussians from slot features.

Each slot gets two independent MLPs:
  SlotGeometryMLP   — predicts mu, rot, scale from slot features only
  SlotAppearanceMLP — predicts color, opacity from slot features + first-frame
                      image features (patch_proj fused at input)

Rationale for the split:
  - Position (mu) is already bbox-constrained, so image features add little
    signal and may destabilise geometry learning.
  - Color and opacity directly correspond to first-frame pixel appearance, so
    fusing first-frame patch features only into the appearance branch is more
    semantically correct and avoids cross-contamination.

Gaussian attributes per point:
  mu     [3]:  centre position        — from geometry MLP
  rot    [4]:  rotation quaternion    — from geometry MLP
  scale  [3]:  positive scale         — from geometry MLP
  color  [3]:  RGB ∈ [0, 1]           — from appearance MLP
  opacity[1]:  ∈ (0, 1)              — from appearance MLP

Per-slot first-frame image feature fusion (appearance branch only):
  patch_feats_frame0 [B, N_p, dim_patch] is the concatenation of
    - Aggregator last-layer output for frame 0 (frame+global concat, 2C=2048)
    - DINOv2 last-layer output for frame 0 (C=1024)
  giving dim_patch = 3C = 3072.

  For each slot p, assign_maps[:, p] (detached) is used as a soft spatial
  weight to pool patch_feats_frame0 → [B, dim_patch], then projected via a
  shared patch_proj Linear to [B, dim_proj] and concatenated with
  slot_features[:, p] before the appearance MLP only.

  assign_maps is detached so that GS-head appearance gradients do not
  interfere with the PartSlotRouter's mask-routing objective.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Output dims per branch
GS_DIM_GEO = 10   # mu(3) + rot(4) + scale(3)
GS_DIM_APP = 4    # color(3) + opacity(1)

# Indices into geometry raw output
_IDX_MU    = slice(0, 3)
_IDX_ROT   = slice(3, 7)
_IDX_SCALE = slice(7, 10)

# Indices into appearance raw output
_IDX_COLOR   = slice(0, 3)
_IDX_OPACITY = slice(3, 4)


class SlotGeometryMLP(nn.Module):
    """
    Geometry branch for one slot.
    Input:  slot_features [B, dim_in]
    Output: [B, N_g, GS_DIM_GEO]  (mu, rot, scale — pre-activation)
    """

    def __init__(self, dim_in: int, hidden_dim: int, n_gaussians: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, n_gaussians * GS_DIM_GEO),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.mlp(x).reshape(B, self.n_gaussians, GS_DIM_GEO)


class SlotAppearanceMLP(nn.Module):
    """
    Appearance branch for one slot.
    Input:  cat([slot_features, patch_proj_feat])  [B, dim_in + dim_proj]
    Output: [B, N_g, GS_DIM_APP]  (color, opacity — pre-activation)
    """

    def __init__(self, dim_in: int, hidden_dim: int, n_gaussians: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_gaussians * GS_DIM_APP),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.mlp(x).reshape(B, self.n_gaussians, GS_DIM_APP)


class ArtGaussianHead(nn.Module):
    """
    Canonical 3D Gaussian decoder with split geometry / appearance branches.

    One independent geometry MLP and one independent appearance MLP per slot.
    Image features (first-frame patch features) feed ONLY into the appearance
    branch so that geometry learning is not contaminated by photometric signal.

    Args:
        dim_in:          slot feature dimension (D)
        num_slots:       number of slots (P)
        n_gaussians:     Gaussians per slot
        hidden_dim:      MLP hidden width
        scale_init_log:  initial log-scale bias (exp(scale_init_log) ≈ initial GS size)
        scene_radius:    canonical bbox half-side (positions in [-r, r])
        dim_patch:       dimension of concatenated first-frame patch features
                         (agg_last 2C + dino_last C = 3C = 3072 by default).
                         Set to 0 to disable image feature fusion.
        dim_proj:        projection dimension for patch features before concat (default 256)
    """

    def __init__(
        self,
        dim_in: int = 1024,
        num_slots: int = 8,
        n_gaussians: int = 256,
        hidden_dim: int = 512,
        scale_init_log: float = -4.0,
        scene_radius: float = 1.0,
        dim_patch: int = 3072,
        dim_proj: int = 256,
    ):
        super().__init__()
        self.dim_in       = dim_in
        self.num_slots    = num_slots
        self.n_gaussians  = n_gaussians
        self.scene_radius = scene_radius
        self.scale_init_log = scale_init_log
        self.dim_patch    = dim_patch
        self.dim_proj     = dim_proj

        # ── Shared patch feature projector (appearance branch only) ───────────
        self.use_patch_feats = (dim_patch > 0 and dim_proj > 0)
        if self.use_patch_feats:
            self.patch_proj = nn.Sequential(
                nn.LayerNorm(dim_patch),
                nn.Linear(dim_patch, dim_proj),
                nn.GELU(),
            )
            app_dim_in = dim_in + dim_proj
        else:
            self.patch_proj = None
            app_dim_in = dim_in

        # ── Per-slot geometry MLPs (no image features) ────────────────────────
        self.slot_geo_mlps = nn.ModuleList([
            SlotGeometryMLP(dim_in, hidden_dim, n_gaussians)
            for _ in range(num_slots)
        ])

        # ── Per-slot appearance MLPs (slot + image features) ─────────────────
        self.slot_app_mlps = nn.ModuleList([
            SlotAppearanceMLP(app_dim_in, hidden_dim, n_gaussians)
            for _ in range(num_slots)
        ])

        # ── Bias initialisation ───────────────────────────────────────────────
        for mlp in self.slot_geo_mlps:
            last_linear = mlp.mlp[-1]
            with torch.no_grad():
                last_linear.bias[_IDX_SCALE].fill_(scale_init_log)

        for mlp in self.slot_app_mlps:
            last_linear = mlp.mlp[-1]
            with torch.no_grad():
                last_linear.bias[_IDX_OPACITY].fill_(-2.0)

    def forward(
        self,
        slot_features: torch.Tensor,                     # [B, P, D]
        bbox_center: Optional[torch.Tensor] = None,      # [B, P, 3]
        bbox_size:   Optional[torch.Tensor] = None,      # [B, P, 3]  half-extent
        patch_feats_frame0: Optional[torch.Tensor] = None,  # [B, N_p, dim_patch]
        assign_maps: Optional[torch.Tensor] = None,      # [B, P, H_p, W_p]
    ) -> dict:
        """
        Decode canonical-space Gaussians for all slots.

        Geometry branch (mu, rot, scale) uses slot_features only.
        Appearance branch (color, opacity) uses slot_features + pooled
        first-frame image features (when patch_feats_frame0 and assign_maps
        are provided and use_patch_feats is True).

        Returns dict with keys:
            mu      [B, P, N_g, 3]   position ∈ bbox or [-scene_radius, scene_radius]
            rot     [B, P, N_g, 4]   unit quaternion (w,x,y,z)
            scale   [B, P, N_g, 3]   positive scale (via exp)
            color   [B, P, N_g, 3]   RGB ∈ [0, 1]
            opacity [B, P, N_g, 1]   ∈ (0, 1)
        """
        B, P, D = slot_features.shape
        assert P == self.num_slots, (
            f"Expected {self.num_slots} slots, got {P}"
        )

        # ── Per-slot masked pooling of first-frame patch features ──────────
        use_img = (
            self.use_patch_feats
            and patch_feats_frame0 is not None
            and assign_maps is not None
        )
        if use_img:
            # assign_maps: [B, P, H_p, W_p] — detach so GS gradients don't
            # flow back into PartSlotRouter's routing objective
            masks = assign_maps.detach()                     # [B, P, H_p, W_p]
            B2, P2, H_p, W_p = masks.shape
            N_p = H_p * W_p
            masks_flat = masks.view(B2, P2, N_p)             # [B, P, N_p]
            # Normalise each slot mask to sum-to-1 (soft attention weights)
            weights = masks_flat / masks_flat.sum(-1, keepdim=True).clamp(min=1e-6)

            # patch_feats_frame0: [B, N_p, dim_patch]
            # Weighted pool per slot: einsum(bpn, bnd -> bpd)
            pooled = torch.einsum("bpn,bnd->bpd", weights, patch_feats_frame0)
            # [B, P, dim_patch]

            # Project to dim_proj (shared across slots)
            pooled_proj = self.patch_proj(pooled)            # [B, P, dim_proj]

        # ── Per-slot forward ───────────────────────────────────────────────
        geo_list = []
        app_list = []
        for p in range(P):
            sf = slot_features[:, p, :]                       # [B, D]

            # Geometry: slot features only
            geo_list.append(self.slot_geo_mlps[p](sf))        # [B, N_g, GS_DIM_GEO]

            # Appearance: slot features + image features
            if use_img:
                app_in = torch.cat([sf, pooled_proj[:, p, :]], dim=-1)  # [B, D+dim_proj]
            else:
                app_in = sf if self.patch_proj is None else torch.cat(
                    [sf, torch.zeros(B, self.dim_proj, device=sf.device, dtype=sf.dtype)],
                    dim=-1,
                )
            app_list.append(self.slot_app_mlps[p](app_in))    # [B, N_g, GS_DIM_APP]

        geo_raw = torch.stack(geo_list, dim=1)  # [B, P, N_g, GS_DIM_GEO]
        app_raw = torch.stack(app_list, dim=1)  # [B, P, N_g, GS_DIM_APP]

        # ── Activations ───────────────────────────────────────────────────
        mu_raw = torch.tanh(geo_raw[..., _IDX_MU])
        if bbox_center is not None and bbox_size is not None:
            mu = bbox_center.unsqueeze(2) + mu_raw * bbox_size.unsqueeze(2)
        else:
            mu = mu_raw * self.scene_radius

        rot     = F.normalize(geo_raw[..., _IDX_ROT], dim=-1, eps=1e-4)
        scale   = torch.exp(geo_raw[..., _IDX_SCALE]).clamp(min=1e-6)
        color   = torch.sigmoid(app_raw[..., _IDX_COLOR])
        opacity = torch.sigmoid(app_raw[..., _IDX_OPACITY])

        return {
            "mu":      mu,       # [B, P, N_g, 3]
            "rot":     rot,      # [B, P, N_g, 4]
            "scale":   scale,    # [B, P, N_g, 3]
            "color":   color,    # [B, P, N_g, 3]
            "opacity": opacity,  # [B, P, N_g, 1]
        }
