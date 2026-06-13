"""
sdf_volume.py — SDF volume renderer for articulated objects (VolSDF + hexa-plane).

Phase 1 (this file): STATIC multi-part compositing at the rest frame.
  Each part p has a predicted AABB B_p (bbox_center ± bbox_size) in scene
  (== world) canonical coords. Camera rays are intersected with every part AABB,
  points are sampled along the valid segments, queried on the part's hexa-plane
  (HexaPlaneSDFHead) for SDF + RGB, converted to density via VolSDF, then ALL
  parts' samples are merged, sorted by ray distance, and alpha-composited.

Dynamic parts (ray inverse-transform, paper Eq.S4–S8) come in Phase 2.

Acceleration via nerfacc CUDA primitives:
  ray_aabb_intersect, render_weight_from_density, accumulate_along_rays.
(No occupancy grid: geometry is feed-forward and changes every step, so an
 occupancy cache does not apply.)
"""

from typing import Optional

import torch
import torch.nn.functional as F
import nerfacc

from dggt.utils.ray_transform import inverse_transform_rays


# ---------------------------------------------------------------------------
# VolSDF: SDF → density  (Laplace CDF, ref [69] VolSDF; paper Eq.S3)
# ---------------------------------------------------------------------------
def volsdf_density(sdf: torch.Tensor, beta: float) -> torch.Tensor:
    """
    σ = α · Ψ_β(−s),  α = 1/β,  Ψ_β = zero-mean Laplace CDF with scale β.
        s ≥ 0 (outside): σ = (1/(2β)) · exp(−s/β)
        s < 0 (inside) : σ = (1/β) · (1 − ½ exp(s/β))
    1/β is annealed up over training to sharpen surfaces.
    """
    alpha = 1.0 / beta
    s = sdf
    # Numerically stable: both branches use exp(-|s|/β) ∈ (0,1], avoiding the
    # inf/NaN that torch.where produces when the *unselected* branch overflows
    # (exp(+large)). For s≥0: 0.5·exp(-s/β); for s<0: 1-0.5·exp(s/β).
    half = 0.5 * torch.exp(-s.abs() / beta)
    out = torch.where(s >= 0, half, 1.0 - half)
    return alpha * out


# ---------------------------------------------------------------------------
# Camera rays
# ---------------------------------------------------------------------------
def generate_rays(
    c2w: torch.Tensor,          # [4, 4]  cam-to-world
    K: torch.Tensor,            # [3, 3]  intrinsics (pixel units at render res)
    pix_xy: torch.Tensor,       # [Nr, 2] pixel coords (x=col, y=row), float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (rays_o [Nr,3], rays_d [Nr,3] unit) in world coords.

    Cameras use the OpenGL/SAPIEN convention (forward = -z, +y up, image row v
    increases downward) — verified from the dataset: process.py renders with
    `get_model_matrix()` (cam-to-world) and depth = -position_z. Unprojecting GT
    depth confirms objects land in ~[-0.85, 0.85]³ only under this convention.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = pix_xy[:, 0]
    v = pix_xy[:, 1]
    d_cam = torch.stack(
        [(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(u)], dim=-1
    )   # [Nr,3]  OpenGL camera frame
    R = c2w[:3, :3]
    d_world = d_cam @ R.transpose(0, 1)                         # [Nr,3]
    d_world = F.normalize(d_world, dim=-1)
    o_world = c2w[:3, 3].unsqueeze(0).expand_as(d_world)        # [Nr,3]
    return o_world.contiguous(), d_world.contiguous()


# ---------------------------------------------------------------------------
# Static multi-part composite rendering (one image)
# ---------------------------------------------------------------------------
def render_rays_static(
    head,                              # HexaPlaneSDFHead (for .query)
    planes: torch.Tensor,             # [P, 6, Cf, R, R]
    bbox_center: torch.Tensor,        # [P, 3]
    bbox_size: torch.Tensor,          # [P, 3] half-extent
    alive: torch.Tensor,              # [P] bool
    rays_o: torch.Tensor,             # [Nr, 3]
    rays_d: torch.Tensor,             # [Nr, 3] unit
    beta: float = 0.1,
    n_samples: int = 64,
    bg_color: float = 1.0,
    motion_probs: torch.Tensor = None,  # [P, 3] (static, prismatic, revolute); None → all static
    axis: torch.Tensor = None,          # [P, 3]
    pivot: torch.Tensor = None,         # [P, 3]
    scalar: torch.Tensor = None,        # [P]   this frame's normalised motion
    scene_radius: float = 1.0,
) -> dict:
    """
    Multi-part SDF volume rendering with optional per-part articulation.

    If motion_probs is given, each dynamic part inverse-transforms the rays into
    its rest frame (paper Eq.S4–S8) before AABB intersection / hexa query; rigid
    transforms preserve ray distance t, so cross-part compositing stays valid.

    Returns dict:
        rgb          [Nr, 3]   composited colour (over bg_color)
        opacity      [Nr, 1]   accumulated foreground alpha (silhouette)
        depth        [Nr, 1]   expected ray distance
        part_opacity [Nr, P]   per-part accumulated alpha (per-part mask)
    """
    P = planes.shape[0]
    Nr = rays_o.shape[0]
    cd = collect_samples(
        head, planes, bbox_center, bbox_size, alive, rays_o, rays_d,
        n_samples=n_samples, motion_probs=motion_probs, axis=axis,
        pivot=pivot, scalar=scalar, scene_radius=scene_radius,
    )
    if cd is None:
        bg = rays_o.new_full((Nr, 3), bg_color)
        return {
            "rgb": bg,
            "opacity": rays_o.new_zeros(Nr, 1),
            "depth": rays_o.new_zeros(Nr, 1),
            "part_opacity": rays_o.new_zeros(Nr, P),
        }
    # Shared SDF/RGB MLPs (single call). For the multi-frame training loss, prefer
    # collect_samples → ONE global MLP call across all frames → composite_samples,
    # so each shared param is used exactly once per backward (DDP-safe). This
    # wrapper keeps the single-image API (overfit / inference) unchanged.
    sdf   = head.sdf_mlp(cd["feats"]) + cd["s_bias"]
    rgb   = torch.sigmoid(head.rgb_mlp(cd["feats"]))
    sigma = volsdf_density(sdf[:, 0], beta)
    return composite_samples(
        sigma, rgb, cd["ray_idx"], cd["t_starts"], cd["t_ends"], cd["part"],
        Nr, P, bg_color=bg_color,
    )


def collect_samples(
    head,                              # HexaPlaneSDFHead (for .query_features)
    planes: torch.Tensor,             # [P, 6, Cf, R, R]
    bbox_center: torch.Tensor,        # [P, 3]
    bbox_size: torch.Tensor,          # [P, 3] half-extent
    alive: torch.Tensor,              # [P] bool
    rays_o: torch.Tensor,             # [Nr, 3]
    rays_d: torch.Tensor,             # [Nr, 3] unit
    n_samples: int = 64,
    motion_probs: torch.Tensor = None,
    axis: torch.Tensor = None,
    pivot: torch.Tensor = None,
    scalar: torch.Tensor = None,
    scene_radius: float = 1.0,
) -> dict | None:
    """
    Sample points along rays per alive part and query their per-part hexa features.
    Does NOT call the shared SDF/RGB MLPs — the caller batches those into a single
    call (so the shared params are used exactly once per backward → DDP-safe).
    Returns dict(feats[N,3Cf], s_bias[N,1], ray_idx[N], t_starts[N], t_ends[N],
    part[N]) or None if no part is hit by any ray.
    """
    P = planes.shape[0]
    device = rays_o.device
    ray_idx_all, ts_all, te_all, feat_all, sbias_all, part_all = [], [], [], [], [], []

    for p in range(P):
        if not bool(alive[p]):
            continue
        center = bbox_center[p]                                 # [3]
        size   = bbox_size[p].clamp(min=1e-4)                   # [3]

        # Inverse-transform rays into part p's rest frame (dynamic parts).
        if motion_probs is not None and bool((motion_probs[p, 1:] > 1e-3).any()):
            o_p, d_p = inverse_transform_rays(
                rays_o, rays_d, motion_probs[p], axis[p], pivot[p], scalar[p],
                scene_radius=scene_radius,
            )
        else:
            o_p, d_p = rays_o, rays_d

        aabb = torch.cat([center - size, center + size]).view(1, 6)
        t_min, t_max, hit = nerfacc.ray_aabb_intersect(o_p, d_p, aabb)
        t_min, t_max, hit = t_min[:, 0], t_max[:, 0], hit[:, 0]
        ray_sel = torch.nonzero(hit, as_tuple=False).squeeze(-1)   # [K]
        if ray_sel.numel() == 0:
            continue

        tn = t_min[ray_sel].clamp(min=0.0)                     # [K]
        tf = t_max[ray_sel]
        edges = torch.linspace(0, 1, n_samples + 1, device=device)
        t_e = tn[:, None] + (tf - tn)[:, None] * edges[None, :]    # [K, n+1]
        t_starts = t_e[:, :-1].reshape(-1)                     # [K*n]
        t_ends   = t_e[:, 1:].reshape(-1)
        t_mid    = 0.5 * (t_starts + t_ends)

        ray_of = ray_sel[:, None].expand(-1, n_samples).reshape(-1)   # [K*n]
        x_rest = o_p[ray_of] + t_mid[:, None] * d_p[ray_of]    # [K*n, 3] rest frame
        x_hat = (x_rest - center) / size                       # [-1,1]³ inside AABB

        # Per-part hexa features (uses per-part `planes[p]` activations, NOT shared
        # params). Shared SDF/RGB MLPs are deferred to a single batched call upstream.
        feat_all.append(head.query_features(planes[p], x_hat))  # [K*n, 3Cf]
        sbias_all.append(x_hat.norm(dim=-1, keepdim=True) - head.sphere_radius)
        ray_idx_all.append(ray_of)
        ts_all.append(t_starts)
        te_all.append(t_ends)
        part_all.append(torch.full_like(ray_of, p))

    if len(ray_idx_all) == 0:
        return None
    return {
        "feats":    torch.cat(feat_all),    # [N, 3Cf]
        "s_bias":   torch.cat(sbias_all),   # [N, 1]
        "ray_idx":  torch.cat(ray_idx_all), # [N]
        "t_starts": torch.cat(ts_all),      # [N]
        "t_ends":   torch.cat(te_all),      # [N]
        "part":     torch.cat(part_all),    # [N]
    }


def composite_samples(
    sigma: torch.Tensor,    # [N]    per-sample density
    rgb: torch.Tensor,      # [N, 3] per-sample colour
    ray_idx: torch.Tensor,  # [N]    ray index in [0, Nr)
    t_starts: torch.Tensor, # [N]
    t_ends: torch.Tensor,   # [N]
    part: torch.Tensor,     # [N]    part id per sample
    Nr: int,
    P: int,
    bg_color: float = 1.0,
) -> dict:
    """Sort merged samples per ray and volume-composite → rgb/opacity/depth/part_opacity."""
    bg = sigma.new_full((Nr, 3), bg_color)

    # Sort samples per ray by increasing distance (merge parts along each ray).
    key = ray_idx.double() * 1e6 + t_mid_global(t_starts, t_ends)
    order = torch.argsort(key)
    ray_idx, t_starts, t_ends = ray_idx[order], t_starts[order], t_ends[order]
    sigma, rgb, part = sigma[order], rgb[order], part[order]

    weights, _, _ = nerfacc.render_weight_from_density(
        t_starts, t_ends, sigma, ray_indices=ray_idx, n_rays=Nr
    )                                                          # [N]
    comp_rgb = nerfacc.accumulate_along_rays(weights, rgb, ray_idx, Nr)    # [Nr,3]
    opacity  = nerfacc.accumulate_along_rays(weights, None, ray_idx, Nr)   # [Nr,1]
    t_mid    = 0.5 * (t_starts + t_ends)
    depth    = nerfacc.accumulate_along_rays(weights, t_mid.unsqueeze(-1), ray_idx, Nr)

    # Per-part accumulated alpha (silhouette of each part)
    part_opacity = sigma.new_zeros(Nr, P)
    for p in range(P):
        m = (part == p)
        if m.any():
            part_opacity[:, p:p + 1] = nerfacc.accumulate_along_rays(
                weights[m], None, ray_idx[m], Nr
            )

    comp_rgb = comp_rgb + (1.0 - opacity) * bg
    return {
        "rgb": comp_rgb,
        "opacity": opacity,
        "depth": depth,
        "part_opacity": part_opacity,
    }


def t_mid_global(t_starts: torch.Tensor, t_ends: torch.Tensor) -> torch.Tensor:
    return (0.5 * (t_starts + t_ends)).double()
