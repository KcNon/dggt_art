"""
Rigid body transforms for articulated object rendering.

Supports three joint types:
  0 = static   (identity)
  1 = prismatic (translation along axis)
  2 = revolute  (rotation around axis through pivot)

During training: differentiable soft mixture over motion types (via softmax weights).
During inference: hard argmax selection.

All operations are fully differentiable w.r.t. axis, pivot, and scalar inputs.
Numerically stabilised at θ ≈ 0 with eps clamping.
"""

import torch
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Low-level math
# ---------------------------------------------------------------------------

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Batched skew-symmetric (cross-product) matrix for vectors [..., 3].

    Returns [..., 3, 3] such that skew(v) @ u == cross(v, u).
    """
    *batch, _ = v.shape
    zeros = torch.zeros(*batch, device=v.device, dtype=v.dtype)
    rows = [
        torch.stack([ zeros,      -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([ v[..., 2],   zeros,      -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],  v[..., 0],   zeros     ], dim=-1),
    ]
    return torch.stack(rows, dim=-2)  # [..., 3, 3]


def rodrigues_rotation_matrix(
    axis: torch.Tensor,   # [..., 3] unit vectors
    angle: torch.Tensor,  # [...] radians
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Rodrigues' formula:  R = I + sin(θ)[D]× + (1-cos(θ))[D]×²

    Numerically stable at θ ≈ 0 (no divide-by-zero, gradients bounded).

    Args:
        axis:  [..., 3] unit rotation axes (assumed normalised by caller)
        angle: [...] rotation angles in radians
    Returns:
        R: [..., 3, 3] rotation matrices
    """
    *batch, _ = axis.shape
    # Ensure axis is unit-length (safe for float16: eps=1e-4 >> float16 min positive 6.1e-5)
    axis = F.normalize(axis, dim=-1, eps=1e-4)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(*batch, 3, 3)
    D = skew_symmetric(axis)                     # [..., 3, 3]
    D2 = torch.matmul(D, D)                       # [..., 3, 3]

    sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)   # [..., 1, 1]
    cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)   # [..., 1, 1]

    return I + sin_a * D + (1.0 - cos_a) * D2


def axis_angle_to_quaternion(
    axis: torch.Tensor,   # [..., 3] unit vectors
    angle: torch.Tensor,  # [...] radians
) -> torch.Tensor:
    """
    Convert axis-angle to quaternion (w, x, y, z).

    Directly differentiable; avoids numerical issues of matrix-to-quaternion.
    """
    half = angle / 2.0
    w = torch.cos(half)                    # [...]
    xyz = torch.sin(half).unsqueeze(-1) * axis  # [..., 3]
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # [..., 4]


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert batched rotation matrices [..., 3, 3] to quaternions [..., 4] (w,x,y,z).

    Uses Shepperd's numerically-stable vectorised method.
    """
    *batch, _, _ = R.shape
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Compute all four candidate quaternions
    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2       # 4w
    s1 = torch.sqrt(torch.clamp(1 + R[...,0,0] - R[...,1,1] - R[...,2,2], min=1e-10)) * 2  # 4x
    s2 = torch.sqrt(torch.clamp(1 - R[...,0,0] + R[...,1,1] - R[...,2,2], min=1e-10)) * 2  # 4y
    s3 = torch.sqrt(torch.clamp(1 - R[...,0,0] - R[...,1,1] + R[...,2,2], min=1e-10)) * 2  # 4z

    q0 = torch.stack([0.25*s0,
                      (R[...,2,1]-R[...,1,2])/s0.clamp(1e-10),
                      (R[...,0,2]-R[...,2,0])/s0.clamp(1e-10),
                      (R[...,1,0]-R[...,0,1])/s0.clamp(1e-10)], dim=-1)

    q1 = torch.stack([(R[...,2,1]-R[...,1,2])/s1.clamp(1e-10),
                      0.25*s1,
                      (R[...,0,1]+R[...,1,0])/s1.clamp(1e-10),
                      (R[...,0,2]+R[...,2,0])/s1.clamp(1e-10)], dim=-1)

    q2 = torch.stack([(R[...,0,2]-R[...,2,0])/s2.clamp(1e-10),
                      (R[...,0,1]+R[...,1,0])/s2.clamp(1e-10),
                      0.25*s2,
                      (R[...,1,2]+R[...,2,1])/s2.clamp(1e-10)], dim=-1)

    q3 = torch.stack([(R[...,1,0]-R[...,0,1])/s3.clamp(1e-10),
                      (R[...,0,2]+R[...,2,0])/s3.clamp(1e-10),
                      (R[...,1,2]+R[...,2,1])/s3.clamp(1e-10),
                      0.25*s3], dim=-1)

    # Select case by largest diagonal
    cond0 = (trace > 0)
    cond1 = (~cond0) & (R[...,0,0] > R[...,1,1]) & (R[...,0,0] > R[...,2,2])
    cond2 = (~cond0) & (~cond1) & (R[...,1,1] > R[...,2,2])
    cond3 = ~cond0 & ~cond1 & ~cond2

    def sel(c, qi):
        return c.unsqueeze(-1).float() * qi

    q = sel(cond0, q0) + sel(cond1, q1) + sel(cond2, q2) + sel(cond3, q3)
    return F.normalize(q, dim=-1, eps=1e-4)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Batched Hamilton product q1 ⊗ q2  (w, x, y, z convention).

    Args:
        q1, q2: [..., 4]
    Returns:
        [..., 4]
    """
    w1, x1, y1, z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    w2, x2, y2, z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ---------------------------------------------------------------------------
# Joint transforms
# ---------------------------------------------------------------------------

def apply_revolute(
    points: torch.Tensor,   # [N, 3]
    quats: torch.Tensor,    # [N, 4]  (w, x, y, z)
    axis: torch.Tensor,     # [3]
    pivot: torch.Tensor,    # [3]
    scalar: torch.Tensor,   # []  ∈ [-1, 1]
    max_angle: float = 2 * math.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a revolute (rotation) joint."""
    angle = scalar * max_angle                    # []
    R = rodrigues_rotation_matrix(               # [3, 3]
        axis.unsqueeze(0), angle.unsqueeze(0)
    )[0]

    # Rotate points around pivot
    pts_shifted = points - pivot.unsqueeze(0)    # [N, 3]
    world_pts = pts_shifted @ R.T + pivot        # [N, 3]

    # Synchronise Gaussian quaternions: q_world = R_quat ⊗ q_canonical
    R_quat = axis_angle_to_quaternion(axis, angle)               # [4]
    R_quat_exp = R_quat.unsqueeze(0).expand(quats.shape[0], -1) # [N, 4]
    world_quats = F.normalize(quaternion_multiply(R_quat_exp, quats), dim=-1, eps=1e-4)

    return world_pts, world_quats


def apply_prismatic(
    points: torch.Tensor,   # [N, 3]
    quats: torch.Tensor,    # [N, 4]
    axis: torch.Tensor,     # [3]
    scalar: torch.Tensor,   # []  ∈ [-1, 1]
    max_translation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a prismatic (translation) joint."""
    translation = scalar * max_translation * axis   # [3]
    world_pts = points + translation.unsqueeze(0)   # [N, 3]
    return world_pts, quats  # orientations unchanged


def apply_rigid_transform(
    points: torch.Tensor,              # [N, 3]
    quats: torch.Tensor,               # [N, 4]
    motion_type_probs: torch.Tensor,   # [3]  softmax probs (static, prismatic, revolute)
    axis: torch.Tensor,                # [3]  unit vector
    pivot: torch.Tensor,               # [3]
    scalar: torch.Tensor,              # []   ∈ [-1, 1]
    max_angle: float = 2 * math.pi,
    max_translation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable soft mixture of all motion types.

    Training: soft weighted sum keeps gradients flowing through all branches.
    Inference: caller should use argmax(motion_type_probs) for hard selection.

    Args:
        points: canonical Gaussian centres [N, 3]
        quats:  canonical Gaussian quaternions [N, 4]
        motion_type_probs: [3] softmax probabilities for [static, prismatic, revolute]
        axis:   unit rotation/translation axis [3]
        pivot:  pivot point for revolute joints [3]
        scalar: normalised motion scalar ∈ [-1, 1]
    Returns:
        (world_points [N, 3], world_quats [N, 4])
    """
    p_s, p_p, p_r = motion_type_probs[0], motion_type_probs[1], motion_type_probs[2]

    pts_static,    qts_static    = points, quats
    pts_prism,     qts_prism     = apply_prismatic(points, quats, axis, scalar, max_translation)
    pts_revolute,  qts_revolute  = apply_revolute(points, quats, axis, pivot, scalar, max_angle)

    world_points = (p_s * pts_static + p_p * pts_prism + p_r * pts_revolute)
    world_quats  = (p_s * qts_static + p_p * qts_prism + p_r * qts_revolute)
    world_quats  = F.normalize(world_quats, dim=-1, eps=1e-4)

    return world_points, world_quats


def apply_rigid_transform_hard(
    points: torch.Tensor,
    quats: torch.Tensor,
    motion_type: int,       # 0=static, 1=prismatic, 2=revolute
    axis: torch.Tensor,
    pivot: torch.Tensor,
    scalar: torch.Tensor,
    max_angle: float = 2 * math.pi,
    max_translation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-selection version for inference."""
    if motion_type == 0:
        return points, quats
    elif motion_type == 1:
        return apply_prismatic(points, quats, axis, scalar, max_translation)
    else:
        return apply_revolute(points, quats, axis, pivot, scalar, max_angle)
