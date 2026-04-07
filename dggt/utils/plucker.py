"""
Plücker ray computation for multi-view image sequences.

Plücker coordinates represent a line in 3D space as (d, m) where:
  d = unit ray direction in world coordinates
  m = o × d  (moment vector, o = camera origin in world)

This 6D representation is used as a spatial prior injected into PartSlotRouter,
replacing absolute positional encodings for geometry-grounded attention.

Convention: extrinsics are cam-to-world 4×4 matrices (R|t form).
"""

import torch
import torch.nn.functional as F


def compute_plucker_rays(
    extrinsics: torch.Tensor,   # [B, S, 4, 4] cam-to-world
    intrinsics: torch.Tensor,   # [B, 3, 3] or [B, S, 3, 3]
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Compute Plücker rays for every pixel in a multi-view sequence.

    Args:
        extrinsics: cam-to-world matrices [B, S, 4, 4]
        intrinsics: camera intrinsic matrices [B, 3, 3] or [B, S, 3, 3]
        H, W: image height and width

    Returns:
        plucker_rays [B, S, H*W, 6] — (direction d, moment o×d)
    """
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype

    if intrinsics.dim() == 3:
        intrinsics = intrinsics.unsqueeze(1).expand(B, S, 3, 3)

    # Pixel grid (u=col, v=row), pixel centers at half-integer coords
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # [H, W]

    ones = torch.ones(H * W, device=device, dtype=dtype)
    pixels = torch.stack(
        [u_coords.reshape(-1), v_coords.reshape(-1), ones], dim=-1
    )  # [H*W, 3]
    pixels = pixels.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)  # [B, S, H*W, 3]

    # K_inv: [B, S, 3, 3]
    K_inv = torch.linalg.inv(intrinsics.reshape(B * S, 3, 3)).reshape(B, S, 3, 3)

    # Ray directions in camera space, then rotate to world space.
    # OpenGL/Blender convention: camera looks along -Z, so negate d_cam
    # to get rays pointing into the scene (forward direction).
    d_cam = torch.einsum("bsij,bsnj->bsni", K_inv, pixels)       # [B, S, H*W, 3]
    d_cam = -d_cam                                                 # forward rays in OpenGL cam space
    R = extrinsics[:, :, :3, :3]                                   # [B, S, 3, 3]
    d_world = torch.einsum("bsij,bsnj->bsni", R, d_cam)           # [B, S, H*W, 3]
    d_world = F.normalize(d_world, dim=-1)

    # Camera origin in world space
    o = extrinsics[:, :, :3, 3]                                    # [B, S, 3]
    o = o.unsqueeze(2).expand_as(d_world)                          # [B, S, H*W, 3]

    # Moment: m = o × d
    moment = torch.linalg.cross(o, d_world)                        # [B, S, H*W, 3]

    return torch.cat([d_world, moment], dim=-1)                    # [B, S, H*W, 6]


def compute_plucker_rays_patch(
    extrinsics: torch.Tensor,   # [B, S, 4, 4]
    intrinsics: torch.Tensor,   # [B, 3, 3] or [B, S, 3, 3]
    H: int,
    W: int,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Compute one Plücker ray per patch (at the patch center pixel).
    Matches the spatial resolution of DINOv2 patch tokens.

    Returns:
        plucker_rays [B, S, N_patches, 6]
        where N_patches = (H // patch_size) * (W // patch_size)
    """
    H_p = H // patch_size
    W_p = W // patch_size
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype

    if intrinsics.dim() == 3:
        intrinsics = intrinsics.unsqueeze(1).expand(B, S, 3, 3)

    # Patch-center pixel coordinates (in pixel units)
    v_centers = (torch.arange(H_p, device=device, dtype=dtype) + 0.5) * patch_size
    u_centers = (torch.arange(W_p, device=device, dtype=dtype) + 0.5) * patch_size
    v_grid, u_grid = torch.meshgrid(v_centers, u_centers, indexing="ij")  # [H_p, W_p]

    ones = torch.ones(H_p * W_p, device=device, dtype=dtype)
    pixels = torch.stack(
        [u_grid.reshape(-1), v_grid.reshape(-1), ones], dim=-1
    )  # [N_patches, 3]
    pixels = pixels.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)  # [B, S, N_patches, 3]

    K_inv = torch.linalg.inv(intrinsics.reshape(B * S, 3, 3)).reshape(B, S, 3, 3)

    d_cam = torch.einsum("bsij,bsnj->bsni", K_inv, pixels)
    d_cam = -d_cam                                 # forward rays in OpenGL cam space
    R = extrinsics[:, :, :3, :3]
    d_world = torch.einsum("bsij,bsnj->bsni", R, d_cam)
    d_world = F.normalize(d_world, dim=-1)

    o = extrinsics[:, :, :3, 3].unsqueeze(2).expand_as(d_world)
    moment = torch.linalg.cross(o, d_world)

    return torch.cat([d_world, moment], dim=-1)   # [B, S, N_patches, 6]


def plucker_stop_gradient(plucker_rays: torch.Tensor) -> torch.Tensor:
    """
    Detach Plücker rays from the computation graph.

    Used in Phase 2 when Plücker rays are derived from CameraHead predictions:
    prevents noisy pose gradients from propagating into PartSlotRouter during
    early training when pose estimates are unreliable.
    """
    return plucker_rays.detach()
