"""
ray_transform.py — inverse ray transforms for dynamic-part SDF rendering.

For a dynamic part at stage t, instead of physically moving the part's volume we
INVERSELY transform the camera rays into the part's rest (canonical) frame, then
render the (unchanged) rest-pose hexa-plane field (paper Eq.S4–S8). Because all
transforms are rigid (rotation/translation), ray-parameter distances t are
preserved, so composited depth/sorting in world space stay valid: sample at t on
the ORIGINAL world ray for compositing, but QUERY the field at the transformed
(rest-frame) point.

Forward motion (point p, normalised scalar S∈[-1,1], r = scene_radius):
    Prismatic: p' = p + 2r·S·D
    Revolute : p' = O + Rot(D, 2π·S)(p − O)
Inverse ray (o, v):
    Prismatic: ô = o − 2r·S·D,                       v̂ = v
    Revolute : ô = O + Rot(D, −2π·S)(o − O),         v̂ = Rot(D, −2π·S) v
Static: identity.

Training uses a soft mixture over motion types (probs), inference picks argmax.
"""

import math
import torch
import torch.nn.functional as F

from dggt.utils.rigid_transform import rodrigues_rotation_matrix


def inverse_transform_rays(
    rays_o: torch.Tensor,        # [N, 3] world ray origins
    rays_d: torch.Tensor,        # [N, 3] world ray dirs (unit)
    motion_probs: torch.Tensor,  # [3] soft probs (static, prismatic, revolute)
    axis: torch.Tensor,          # [3] unit
    pivot: torch.Tensor,         # [3]
    scalar: torch.Tensor,        # [] normalised motion ∈ [-1,1] at this frame
    scene_radius: float = 1.0,
    max_angle: float = 2 * math.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-blended inverse ray transform → (ô [N,3], v̂ [N,3] unit)."""
    p_s, p_p, p_r = motion_probs[0], motion_probs[1], motion_probs[2]

    # Prismatic inverse
    trans = (2.0 * scene_radius) * scalar * axis            # [3]
    o_p = rays_o - trans.unsqueeze(0)
    d_p = rays_d

    # Revolute inverse: rotate by -θ about axis through pivot
    angle = -(max_angle) * scalar                          # []  (negative = inverse)
    R = rodrigues_rotation_matrix(axis.unsqueeze(0), angle.unsqueeze(0))[0]  # [3,3]
    o_r = (rays_o - pivot.unsqueeze(0)) @ R.transpose(0, 1) + pivot.unsqueeze(0)
    d_r = rays_d @ R.transpose(0, 1)

    # Static: identity
    o_s, d_s = rays_o, rays_d

    o_hat = p_s * o_s + p_p * o_p + p_r * o_r
    d_hat = p_s * d_s + p_p * d_p + p_r * d_r
    d_hat = F.normalize(d_hat, dim=-1, eps=1e-6)
    return o_hat, d_hat
