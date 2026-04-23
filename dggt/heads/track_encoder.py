"""
TrackEncoder — Encodes 2D point tracks into per-track tokens consumable by
PartSlotRouter as an additional KV / self-attn stream.

Design (parameter-free descriptor + residual MLP):

  Input:  tracks [B, S, N, 2]  (pixel coords at (W, H))
          vis    [B, S, N]     (0/1 visibility)

  Step 1 — Dynamics descriptor (parameter-free):
      • normalise coords to [-1, 1] via (W, H)
      • 1st difference  dxy         [S-1]   → pad → [S]
      • 2nd difference  ddxy        [S-2]   → pad → [S]
      • Similarity-aligned residual: per frame t, fit a 2D similarity from
        frame-0 → frame-t on all visible tracks (weighted LS on points that
        are visible in BOTH frames), apply inverse to subtract camera / rigid
        scene motion. What remains is part-relative motion — camera-invariant.
      • Global statistics: mean vis, std of residual, track length.

  Step 2 — Projection: desc [B, N, F_desc] → [B, N, D_inner=512]
  Step 3 — Residual MLP: x = x + MLP(x)   (acts as lightweight per-track mixer)
  Step 4 — Output projection: [B, N, D_inner] → [B, N, D_out=1024]

The similarity residual removes the dominant rigid global transform so the
encoder can focus on *part-relative* motion even with a moving camera.
For a static camera this reduces to original (xy - xy_0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weighted_similarity_fit(
    src: torch.Tensor,      # [B, N, 2]
    dst: torch.Tensor,      # [B, N, 2]
    w:   torch.Tensor,      # [B, N]
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted 2D similarity (rotation + uniform scale + translation) fitting
    src → dst. Returns (A [B,2,2], t [B,2]) such that   dst ≈ A @ src + t.

    Closed-form via weighted Umeyama. Stable for N≥2 with non-zero weights.
    """
    B = src.shape[0]
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)                 # [B,1]

    mu_s = (w.unsqueeze(-1) * src).sum(dim=1) / w_sum                 # [B,2]
    mu_d = (w.unsqueeze(-1) * dst).sum(dim=1) / w_sum

    src_c = src - mu_s.unsqueeze(1)                                   # [B,N,2]
    dst_c = dst - mu_d.unsqueeze(1)

    # Weighted covariance 2×2
    w_e   = w.unsqueeze(-1)                                           # [B,N,1]
    sigma = (w_e.unsqueeze(-1) * dst_c.unsqueeze(-1) * src_c.unsqueeze(-2)).sum(dim=1)
    # sigma: [B, 2, 2]

    # Weighted variance of src
    var_s = (w * (src_c ** 2).sum(dim=-1)).sum(dim=1) / w_sum.squeeze(-1)  # [B]

    # SVD on 2×2
    try:
        U, S, Vh = torch.linalg.svd(sigma.float())
    except Exception:
        # Fallback to identity when SVD fails (degenerate batch)
        A = torch.eye(2, device=src.device, dtype=src.dtype).expand(B, 2, 2).clone()
        t = mu_d - mu_s
        return A, t

    # Reflection correction
    det_UV = torch.linalg.det(U) * torch.linalg.det(Vh)               # [B]
    D = torch.eye(2, device=src.device).unsqueeze(0).expand(B, 2, 2).clone()
    D[:, 1, 1] = torch.sign(det_UV).clamp(min=-1.0, max=1.0)

    R = U @ D @ Vh                                                    # [B,2,2]
    s = (S * D.diagonal(dim1=-2, dim2=-1)).sum(dim=-1) / var_s.clamp(min=eps)
    A = (s.view(B, 1, 1) * R).to(src.dtype)
    t = mu_d - (A @ mu_s.unsqueeze(-1)).squeeze(-1)
    return A, t


def compute_dynamics_descriptor(
    tracks: torch.Tensor,   # [B, S, N, 2]  pixel coords
    vis:    torch.Tensor,   # [B, S, N]     0/1
    img_size: tuple[int, int],   # (H, W)
) -> torch.Tensor:
    """
    Returns descriptor [B, N, F]  where F = 2S + 2S + 2S + 2S + 3
      = [norm_xy | dxy | ddxy | residual_xy | stats]
    """
    B, S, N, _ = tracks.shape
    H, W = img_size
    device = tracks.device
    dtype  = tracks.dtype

    # Normalise to [-1, 1]
    xy = tracks.clone()
    xy[..., 0] = xy[..., 0] / max(W, 1) * 2.0 - 1.0
    xy[..., 1] = xy[..., 1] / max(H, 1) * 2.0 - 1.0                   # [B,S,N,2]

    vis_f = vis.to(dtype)                                             # [B,S,N]

    # 1st / 2nd differences (pad with zeros at the start)
    dxy  = torch.zeros_like(xy)
    ddxy = torch.zeros_like(xy)
    if S >= 2:
        dxy[:,  1:]  = xy[:, 1:]  - xy[:, :-1]
    if S >= 3:
        ddxy[:, 2:]  = dxy[:, 2:] - dxy[:, 1:-1]

    # Per-frame similarity-aligned residual (camera motion removed)
    residual = torch.zeros_like(xy)
    if S >= 2:
        src = xy[:, 0]                                                # [B,N,2]
        vis0 = vis_f[:, 0]                                            # [B,N]
        for t in range(1, S):
            dst  = xy[:, t]
            vis_t = vis_f[:, t]
            w = vis0 * vis_t                                          # [B,N]
            # If fewer than 3 joint visible points, fall back to identity.
            enough = (w.sum(dim=-1) >= 3.0).view(B, 1, 1)             # [B,1,1]
            A, trans = _weighted_similarity_fit(src, dst, w)
            pred = (A @ src.transpose(1, 2)).transpose(1, 2) + trans.unsqueeze(1)
            res  = dst - pred                                         # [B,N,2]
            # When not enough visible pairs → residual = dst - src (no alignment)
            res_fb = dst - src
            res  = torch.where(enough, res, res_fb)
            residual[:, t] = res

    # Stats per track
    vis_mean = vis_f.mean(dim=1)                                      # [B,N]
    res_std  = residual.std(dim=1).mean(dim=-1)                       # [B,N]
    track_len = vis_f.sum(dim=1) / max(S, 1)                          # [B,N] (redundant-ish)

    # Rearrange time → features:  [B, S, N, 2] → [B, N, 2S]
    def to_bnf(x):
        return x.permute(0, 2, 1, 3).reshape(B, N, 2 * S)

    feat = torch.cat([
        to_bnf(xy),          # 2S
        to_bnf(dxy),         # 2S
        to_bnf(ddxy),        # 2S
        to_bnf(residual),    # 2S
        vis_mean.unsqueeze(-1),
        res_std.unsqueeze(-1),
        track_len.unsqueeze(-1),
    ], dim=-1)               # [B, N, 8S + 3]
    return feat


class TrackEncoder(nn.Module):
    """
    Encode per-track dynamics into D_out-dim tokens.

    Args:
        num_frames:   S (used to size input Linear)
        img_size:     (H, W) for coord normalisation
        dim_inner:    internal dim (default 512)
        dim_out:      output dim, must match PSR dim_slot (default 1024)
        mlp_ratio:    residual MLP expansion (default 4)
    """

    def __init__(
        self,
        num_frames: int,
        img_size: tuple[int, int] | int,
        dim_inner: int = 512,
        dim_out:   int = 1024,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = tuple(img_size)

        F_in = 8 * num_frames + 3
        self.proj_in = nn.Sequential(
            nn.Linear(F_in, dim_inner),
            nn.LayerNorm(dim_inner),
        )
        hidden = dim_inner * mlp_ratio
        self.encoder = nn.Sequential(
            nn.LayerNorm(dim_inner),
            nn.Linear(dim_inner, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_inner),
        )
        self.proj_out = nn.Linear(dim_inner, dim_out)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        tracks: torch.Tensor,   # [B, S, N, 2]
        vis:    torch.Tensor,   # [B, S, N]
    ) -> torch.Tensor:
        B, S, N, _ = tracks.shape
        assert S == self.num_frames, (
            f"TrackEncoder built for S={self.num_frames} but got S={S}"
        )
        feat = compute_dynamics_descriptor(tracks, vis, self.img_size)  # [B,N,F_in]
        x = self.proj_in(feat)                    # [B,N,D_inner]
        x = x + self.encoder(x)                    # residual MLP
        x = self.proj_out(x)                       # [B,N,D_out]
        return x
