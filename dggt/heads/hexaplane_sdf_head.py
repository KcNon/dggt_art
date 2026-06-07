"""
HexaPlaneSDFHead — per-part hexa-plane feature decoder + shared SDF/RGB MLPs.

Replaces ArtGaussianHead. This is the *geometry/texture* branch of the decoder
(paper Fig.2 branch (i)): each part slot is decoded into a hexa-plane feature
representation T_p, which is queried at 3D sample points during SDF volume
rendering to produce per-point SDF value and RGB colour.

Hexa-plane (paper Eq.S1): three axis-aligned feature planes (xy, yz, xz), each
split into two halves by the sign of the orthogonal axis → 6 planes total:
    {xy+, xy-, yz+, yz-, xz+, xz-},  each [Cf, R, R].
A 3D point x̂ = (x, y, z) ∈ [-1, 1]³ (normalised part-local coords) is queried as:
    f_xy = bilinear(z≥0 ? T_xy+ : T_xy-, (x, y))
    f_yz = bilinear(x≥0 ? T_yz+ : T_yz-, (y, z))
    f_xz = bilinear(y≥0 ? T_xz+ : T_xz-, (x, z))
    f    = concat(f_xy, f_yz, f_xz)  ∈ ℝ^{3Cf}
Two small shared MLPs map f → SDF s and RGB c (paper Eq.S2):
    s = MLP_sdf(f) + s_bias(x̂),   s_bias = ‖x̂‖ − sphere_radius   (sphere init)
    c = sigmoid(MLP_rgb(f))        ∈ [0, 1]³

First-frame image features (same masked-pooling fusion as the old GS head) are
concatenated to the slot vector before plane decoding to give the appearance/
texture branch a photometric signal.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexaPlaneSDFHead(nn.Module):
    """
    Args:
        dim_in:        slot feature dimension (D)
        num_slots:     number of part slots (P)
        plane_res:     hexa-plane spatial resolution R (default 64)
        plane_ch:      hexa-plane feature channels Cf (default 32)
        mlp_hidden:    width of SDF / RGB MLPs
        sphere_radius: sphere-init prior radius in normalised local space (default 0.5)
        scene_radius:  canonical scene half-extent r (kept for API symmetry)
        dim_patch:     first-frame patch feature dim (agg 2C + dino C = 3C = 3072);
                       set 0 to disable image-feature fusion
        dim_proj:      projection dim for pooled patch features before concat
    """

    def __init__(
        self,
        dim_in: int = 1024,
        num_slots: int = 8,
        plane_res: int = 64,
        plane_ch: int = 32,
        mlp_hidden: int = 64,
        sphere_radius: float = 0.5,
        scene_radius: float = 1.0,
        dim_patch: int = 3072,
        dim_proj: int = 256,
    ):
        super().__init__()
        self.num_slots     = num_slots
        self.R             = plane_res
        self.Cf            = plane_ch
        self.sphere_radius = sphere_radius
        self.scene_radius  = scene_radius
        self.r0            = 8   # decoder seed resolution (8 → 16 → 32 → 64)
        assert plane_res == self.r0 * (2 ** 3), (
            f"plane_res must be {self.r0 * 8} for the 3-stage upsampler (got {plane_res})"
        )

        # ── First-frame image-feature fusion (appearance/texture) ──────────
        self.use_patch_feats = (dim_patch > 0 and dim_proj > 0)
        if self.use_patch_feats:
            self.patch_proj = nn.Sequential(
                nn.LayerNorm(dim_patch),
                nn.Linear(dim_patch, dim_proj),
                nn.GELU(),
            )
            dec_in = dim_in + dim_proj
        else:
            self.patch_proj = None
            dec_in = dim_in

        # ── Plane seed: slot vector → [6, Cf, r0, r0] ──────────────────────
        self.seed = nn.Sequential(
            nn.LayerNorm(dec_in),
            nn.Linear(dec_in, 6 * self.Cf * self.r0 * self.r0),
        )

        # ── Shared conv upsampler r0 → R (applied per plane) ───────────────
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(self.Cf, self.Cf, 4, stride=2, padding=1),  # 8→16
            nn.GELU(),
            nn.ConvTranspose2d(self.Cf, self.Cf, 4, stride=2, padding=1),  # 16→32
            nn.GELU(),
            nn.ConvTranspose2d(self.Cf, self.Cf, 4, stride=2, padding=1),  # 32→64
        )

        # ── Shared SDF / RGB MLPs (query hexa features 3Cf → s, c) ─────────
        self.sdf_mlp = nn.Sequential(
            nn.Linear(3 * self.Cf, mlp_hidden),
            nn.Softplus(beta=100),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Softplus(beta=100),
            nn.Linear(mlp_hidden, 1),
        )
        self.rgb_mlp = nn.Sequential(
            nn.Linear(3 * self.Cf, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 3),
        )

        # Zero-init SDF MLP last layer so initial SDF ≈ s_bias (a sphere),
        # which stabilises VolSDF training (paper / VolSDF sphere init).
        nn.init.zeros_(self.sdf_mlp[-1].weight)
        nn.init.zeros_(self.sdf_mlp[-1].bias)

    # ----------------------------------------------------------------------
    # Plane decoding
    # ----------------------------------------------------------------------
    def decode_planes(
        self,
        slot_features: torch.Tensor,                        # [B, P, D]
        patch_feats_frame0: Optional[torch.Tensor] = None,  # [B, N_p, dim_patch]
        assign_maps: Optional[torch.Tensor] = None,         # [B, P, H_p, W_p]
    ) -> torch.Tensor:
        """Return hexa-plane features [B, P, 6, Cf, R, R]."""
        B, P, D = slot_features.shape
        assert P == self.num_slots

        dec_in = slot_features
        if (self.use_patch_feats and patch_feats_frame0 is not None
                and assign_maps is not None):
            masks = assign_maps.detach()                    # [B, P, H_p, W_p]
            B2, P2, H_p, W_p = masks.shape
            masks_flat = masks.view(B2, P2, H_p * W_p)
            weights = masks_flat / masks_flat.sum(-1, keepdim=True).clamp(min=1e-6)
            pooled = torch.einsum("bpn,bnd->bpd", weights, patch_feats_frame0)
            pooled_proj = self.patch_proj(pooled)           # [B, P, dim_proj]
            dec_in = torch.cat([slot_features, pooled_proj], dim=-1)
        elif self.use_patch_feats:
            pad = slot_features.new_zeros(B, P, self.patch_proj[-2].out_features)
            dec_in = torch.cat([slot_features, pad], dim=-1)

        seed = self.seed(dec_in.reshape(B * P, -1))         # [B*P, 6*Cf*r0*r0]
        seed = seed.reshape(B * P * 6, self.Cf, self.r0, self.r0)
        planes = self.upsampler(seed)                       # [B*P*6, Cf, R, R]
        return planes.reshape(B, P, 6, self.Cf, self.R, self.R)

    # ----------------------------------------------------------------------
    # Hexa-plane query
    # ----------------------------------------------------------------------
    def query_features(
        self,
        planes6: torch.Tensor,   # [..., 6, Cf, R, R]   (one part; arbitrary lead dims)
        pts: torch.Tensor,       # [M, 3]  normalised local coords ∈ [-1, 1]
    ) -> torch.Tensor:
        """Bilinearly sample the 6 sign-split planes → [M, 3Cf]."""
        Cf, R = self.Cf, self.R
        planes6 = planes6.reshape(6, Cf, R, R)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        def samp(plane_idx, ca, cb):
            # grid_sample: input [1,Cf,R,R], grid [1,1,M,2] (last dim = (x=w, y=h))
            grid = torch.stack([ca, cb], dim=-1).view(1, 1, -1, 2)
            out = F.grid_sample(
                planes6[plane_idx].unsqueeze(0), grid,
                mode="bilinear", align_corners=True, padding_mode="border",
            )                                                # [1, Cf, 1, M]
            return out.view(Cf, -1).transpose(0, 1)          # [M, Cf]

        # Sign-split selection (paper Eq.S1)
        zpos = (z >= 0).unsqueeze(-1)
        xpos = (x >= 0).unsqueeze(-1)
        ypos = (y >= 0).unsqueeze(-1)
        f_xy = torch.where(zpos, samp(0, x, y), samp(1, x, y))
        f_yz = torch.where(xpos, samp(2, y, z), samp(3, y, z))
        f_xz = torch.where(ypos, samp(4, x, z), samp(5, x, z))
        return torch.cat([f_xy, f_yz, f_xz], dim=-1)         # [M, 3Cf]

    def query(self, planes6: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Query SDF and RGB at points (one part).
        Returns: sdf [M, 1], rgb [M, 3] ∈ [0,1].
        """
        feats = self.query_features(planes6, pts)            # [M, 3Cf]
        s_bias = pts.norm(dim=-1, keepdim=True) - self.sphere_radius   # [M, 1]
        sdf = self.sdf_mlp(feats) + s_bias                   # [M, 1]
        rgb = torch.sigmoid(self.rgb_mlp(feats))             # [M, 3]
        return sdf, rgb

    # Convenience: keep a forward() that just decodes planes.
    def forward(self, slot_features, patch_feats_frame0=None, assign_maps=None):
        return self.decode_planes(slot_features, patch_feats_frame0, assign_maps)
