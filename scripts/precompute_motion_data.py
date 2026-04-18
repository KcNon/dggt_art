#!/usr/bin/env python3
"""Precompute per-scene motion data for ArtVGGT motion-supervised training.

For every cam under every scene in --data_root, this script:
  1. runs CoWTracker on the RGB video (dense per-pixel 2D tracks)
  2. subsamples to ~stride²=4096 query points and filters by vis·conf
  3. unprojects to 3D via depth + intrinsics + extrinsics (world meters)
  4. initializes per-track soft part assignment with DINOv2 features + KMeans
  5. alternates weighted-Procrustes (R, t per part per frame) with Adam on
     soft weights → produces clean per-part rigid-residual partition
  6. projects per-pixel weights back to patch grid → motion_mask

Output written to: ``{cam_dir}/motion_cache.npz`` with keys
    frame_ids        list[str]
    tracks_2d_norm   [S, N, 2]   pixel coords / (W, H), values in [0, 1]
    tracks_3d        [S, N, 3]   world-coord meters
    tracks_vis       [S, N]      float, 1 = visible
    track_part_label [N, P]      soft assignment summing to 1 per row
    motion_mask      [P, H_p, W_p] soft per-patch label
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# third_party/cowtracker is not pip-installed; add to path.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))  # project root, for `dggt.*`
sys.path.insert(0, str(_THIS.parent / "third_party" / "cowtracker"))


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_video(img_paths, target_size):
    """Return uint8 video [S, 3, H, W] and the (W_orig, H_orig) of source."""
    imgs = []
    orig_size = None
    for p in img_paths:
        im = Image.open(p).convert("RGB")
        if orig_size is None:
            orig_size = im.size  # (W, H)
        im = im.resize((target_size, target_size), Image.BILINEAR)
        imgs.append(np.asarray(im))
    arr = np.stack(imgs).astype(np.uint8)             # [S, H, W, 3]
    arr = np.transpose(arr, (0, 3, 1, 2))             # [S, 3, H, W]
    return arr, orig_size


def load_depth(npy_path, target_size):
    d = np.load(npy_path).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_size, target_size),
                      mode="bilinear", align_corners=False)
    return t.squeeze().numpy()                         # [H, W]


def load_intrinsics(path, orig_w, orig_h, target_size):
    K = np.loadtxt(path).astype(np.float32)
    K[0, 0] *= target_size / orig_w   # fx
    K[1, 1] *= target_size / orig_h   # fy
    K[0, 2] *= target_size / orig_w   # cx
    K[1, 2] *= target_size / orig_h   # cy
    return K


def list_cam_dirs(scene_dir, exclude_cams):
    out = []
    for c in sorted(scene_dir.iterdir()):
        if (c.is_dir() and c.name.startswith("cam_")
                and (c / "images").exists()
                and c.name not in exclude_cams):
            out.append(c)
    return out


def get_frame_paths(cam_dir):
    img_dir = cam_dir / "images"
    paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    fids = [p.stem for p in paths]
    return fids, paths


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def unproject_to_world(pix_xy, depth_at_pix, K, cam_to_world):
    """Lift 2D pixel + depth → 3D world coord.

    pix_xy:        [N, 2]  pixel coords (x, y)
    depth_at_pix:  [N]     metric depth at those pixels (camera Z)
    K:             [3, 3]
    cam_to_world:  [4, 4]
    Returns:       [N, 3]  world coords
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pix_xy[:, 0] - cx) / fx * depth_at_pix
    y = (pix_xy[:, 1] - cy) / fy * depth_at_pix
    z = depth_at_pix
    cam = np.stack([x, y, z], axis=-1)                  # [N, 3]
    R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
    return cam @ R.T + t


def sample_depth(depth_map, pix_xy):
    """Bilinear sample depth at sub-pixel locations."""
    H, W = depth_map.shape
    x = np.clip(pix_xy[:, 0], 0, W - 1.001)
    y = np.clip(pix_xy[:, 1], 0, H - 1.001)
    x0 = np.floor(x).astype(np.int64); x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
    wx = x - x0; wy = y - y0
    d00 = depth_map[y0, x0]; d01 = depth_map[y0, x1]
    d10 = depth_map[y1, x0]; d11 = depth_map[y1, x1]
    return ((1 - wx) * (1 - wy) * d00
            + wx * (1 - wy) * d01
            + (1 - wx) * wy * d10
            + wx * wy * d11)


def weighted_procrustes(P0, Pt, w):
    """Weighted Procrustes: minimize Σ_n w_n ||R P0_n + t - Pt_n||².

    P0: [N, 3]  source (frame 0)
    Pt: [N, 3]  target (frame t)
    w:  [N]     weights ≥ 0
    Returns (R [3,3], t [3]).
    """
    eps = 1e-8
    w_sum = w.sum() + eps
    mu0 = (w[:, None] * P0).sum(axis=0) / w_sum
    mut = (w[:, None] * Pt).sum(axis=0) / w_sum
    Q0 = P0 - mu0
    Qt = Pt - mut
    H = (w[:, None] * Q0).T @ Qt                       # [3, 3]
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = mut - R @ mu0
    return R, t


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def kmeans_soft_init(feats, K, temp=0.5, n_iters=20, seed=0):
    """KMeans++ → soft assignment via exp(-d² / temp).

    feats: [N, C]  (np.float32)
    K:     int
    Returns w [N, K] (sums to 1 along K).
    """
    rng = np.random.default_rng(seed)
    N, C = feats.shape

    # KMeans++ init
    centers = [feats[rng.integers(0, N)]]
    for _ in range(K - 1):
        d2 = np.min(((feats[:, None, :] - np.stack(centers)[None]) ** 2).sum(-1), axis=1)
        d2 = np.maximum(d2, 1e-8)
        idx = rng.choice(N, p=d2 / d2.sum())
        centers.append(feats[idx])
    centers = np.stack(centers)                         # [K, C]

    # Lloyd iterations
    for _ in range(n_iters):
        d2 = ((feats[:, None] - centers[None]) ** 2).sum(-1)   # [N, K]
        assign = d2.argmin(axis=1)
        new = np.zeros_like(centers)
        for k in range(K):
            sel = feats[assign == k]
            new[k] = sel.mean(0) if len(sel) > 0 else centers[k]
        if np.allclose(centers, new, atol=1e-5):
            break
        centers = new

    d2 = ((feats[:, None] - centers[None]) ** 2).sum(-1)
    logits = -d2 / max(temp, 1e-4)
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return w


def motion_residual_init(
    tracks_3d,          # [S, N, 3]
    tracks_vis,         # [S, N]
    K,
    dino_feat=None,     # [N, C_d] or None — optional appearance side-channel
    dino_weight=0.3,
    temp=0.3,
    n_iters=30,
    seed=0,
):
    """Cluster tracks by *motion residual after a global rigid fit*.

    Rationale
    ---------
    DINOv2 features cluster by appearance, which fails on articulated objects
    whose parts share texture. Instead, fit a single global rigid
    transform per frame using all visible tracks (weighted Procrustes), then
    each track's residual sequence ``r[t,n] = P_t[n] - (R_t^g P_0[n] + τ_t^g)``
    encodes *part-specific motion*. KMeans in this residual space naturally
    separates parts that move differently from the dominant rigid motion.

    Slot 0 is explicitly seeded as the **zero-residual / static** cluster so
    the static base receives a consistent slot id across scenes.

    Returns
    -------
    w : [N, K] soft assignment (rows sum to 1).
    """
    S, N, _ = tracks_3d.shape

    # 1) Global rigid fit per frame (all visible tracks, frame 0 as source).
    residuals = np.zeros((S, N, 3), dtype=np.float32)
    for t in range(S):
        w_t = tracks_vis[t].astype(np.float32)
        if w_t.sum() < 3:
            continue
        R, tau = weighted_procrustes(tracks_3d[0], tracks_3d[t], w_t)
        pred = tracks_3d[0] @ R.T + tau
        residuals[t] = tracks_3d[t] - pred

    # 2) Scale-normalize residuals by scene motion scale (median magnitude),
    #    so `temp` is unit-independent.
    mag = np.linalg.norm(residuals, axis=-1)                      # [S, N]
    scale = np.median(mag[mag > 0]) + 1e-6
    residuals = residuals / scale

    # 3) Motion feature: stacked residuals + summary magnitude.
    vis_sum = tracks_vis.sum(axis=0) + 1e-6                       # [N]
    mag_mean = (mag / scale * tracks_vis).sum(0) / vis_sum        # [N]
    mag_max = (mag / scale).max(0)                                # [N]
    feat = residuals.transpose(1, 0, 2).reshape(N, -1)            # [N, 3S]
    feat = np.concatenate([feat, mag_mean[:, None], mag_max[:, None]], axis=1)

    # 4) Optional: append a *small* DINO appearance component to break ties
    #    among tracks with similar residuals but on different part.
    if dino_feat is not None and dino_weight > 0:
        df = dino_feat.astype(np.float32)
        df = df / (np.linalg.norm(df, axis=1, keepdims=True) + 1e-8)
        df = df * (dino_weight * feat.std())      # match magnitude loosely
        feat = np.concatenate([feat, df], axis=1)

    # 5) Seed slot 0 = zero vector; seed 1..K-1 via KMeans++ on the
    #    high-residual subset (low-residual tracks are likely static → belong
    #    to slot 0, don't let them become seeds).
    rng = np.random.default_rng(seed)
    C = feat.shape[1]
    centers = [np.zeros(C, dtype=np.float32)]

    # Candidate pool = top 50% by residual magnitude.
    thr = np.median(mag_max)
    cand_idx = np.where(mag_max > max(thr, 1e-6))[0]
    if len(cand_idx) < K:
        cand_idx = np.arange(N)

    first_local = rng.integers(0, len(cand_idx))
    centers.append(feat[cand_idx[first_local]].copy())
    for _ in range(K - 2):
        cand = feat[cand_idx]
        d2 = np.min(((cand[:, None, :] - np.stack(centers)[None]) ** 2).sum(-1), axis=1)
        d2 = np.maximum(d2, 1e-8)
        pick = rng.choice(len(cand_idx), p=d2 / d2.sum())
        centers.append(cand[pick].copy())
    centers = np.stack(centers).astype(np.float32)                # [K, C]

    # 6) Lloyd iterations (slot 0 anchored toward zero).
    for _ in range(n_iters):
        d2 = ((feat[:, None] - centers[None]) ** 2).sum(-1)       # [N, K]
        assign = d2.argmin(axis=1)
        new = centers.copy()
        for k in range(K):
            sel = feat[assign == k]
            if len(sel) > 0:
                new[k] = sel.mean(0)
        # Pull slot 0 toward zero so it remains "the static cluster".
        new[0] = 0.5 * new[0]
        if np.allclose(centers, new, atol=1e-5):
            break
        centers = new

    # 7) Soft assignment. Temperature scaled by feature std so it's
    #    unit-independent across scenes.
    d2 = ((feat[:, None] - centers[None]) ** 2).sum(-1)
    temp_eff = max(temp * (feat.std() ** 2), 1e-4)
    logits = -d2 / temp_eff
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return w


def optimize_part_weights(
    tracks_3d,            # [S, N, 3]
    tracks_vis,           # [S, N]
    w_init,               # [N, P]
    n_outer=10,
    n_inner=20,
    lr=0.05,
    lambda_smooth=0.0,    # disabled by default; needs feature graph
    lambda_ent=0.01,
    lambda_init=0.05,
    device="cuda",
):
    """Alternating weighted-Procrustes + Adam on soft part weights.

    Returns w [N, P] (numpy, sums to 1 along P).
    """
    S, N, _ = tracks_3d.shape
    P = w_init.shape[1]

    pts3d = torch.from_numpy(tracks_3d).float().to(device)        # [S, N, 3]
    vis   = torch.from_numpy(tracks_vis).float().to(device)       # [S, N]
    w     = torch.from_numpy(w_init).float().to(device)           # [N, P]
    w_init_t = w.clone()

    Rs = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(S, P, 1, 1)
    ts = torch.zeros(S, P, 3, device=device)

    P0 = pts3d[0]                                                  # [N, 3]
    eps = 1e-8

    for outer in range(n_outer):
        # --- M-step: closed-form weighted Procrustes per (t, k) -----------
        with torch.no_grad():
            wp_np = w.cpu().numpy()                                # [N, P]
            P0_np = P0.cpu().numpy()
            for t in range(S):
                Pt = pts3d[t].cpu().numpy()
                v  = tracks_vis[t]                                 # [N] np.float
                for k in range(P):
                    wk = wp_np[:, k] * v
                    if wk.sum() < 1e-3:
                        Rs[t, k] = torch.eye(3, device=device)
                        ts[t, k] = 0
                        continue
                    R_np, tau_np = weighted_procrustes(P0_np, Pt, wk)
                    Rs[t, k] = torch.from_numpy(R_np).float().to(device)
                    ts[t, k] = torch.from_numpy(tau_np).float().to(device)

        # --- E-step: Adam on logits over P -------------------------------
        # Re-parametrize w as softmax(logits) for stability
        logits = torch.log(w.clamp_min(1e-6))
        logits = logits.detach().requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=lr)

        for _ in range(n_inner):
            opt.zero_grad()
            wp = torch.softmax(logits, dim=-1)                    # [N, P]

            # Predicted positions for each (t, k): R_k P0 + t_k → [t, k, n, 3]
            P0_kk = P0.unsqueeze(0).unsqueeze(0)                  # [1,1,N,3]
            R_full = Rs.unsqueeze(2)                              # [S,P,1,3,3]
            t_full = ts.unsqueeze(2)                              # [S,P,1,3]
            pred = (R_full @ P0_kk.unsqueeze(-1)).squeeze(-1) + t_full  # [S,P,N,3]

            # Target
            target = pts3d.unsqueeze(1)                            # [S,1,N,3]
            res = ((pred - target) ** 2).sum(-1)                  # [S,P,N]

            # Weighted by visibility and soft assignment
            v = vis.unsqueeze(1)                                   # [S,1,N]
            l_main = (wp.unsqueeze(0).transpose(1, 2) * v * res).mean()

            # Entropy regularizer (push to hard assignments)
            l_ent = -(wp * torch.log(wp.clamp_min(eps))).sum(-1).mean()

            # Init regularizer (BCE-style, prevent collapse)
            l_init = F.kl_div(torch.log(wp.clamp_min(eps)),
                              w_init_t, reduction="batchmean")

            loss = l_main + lambda_ent * l_ent + lambda_init * l_init
            loss.backward()
            opt.step()

        w = torch.softmax(logits.detach(), dim=-1)

    return w.cpu().numpy()


# ---------------------------------------------------------------------------
# Pipeline per cam
# ---------------------------------------------------------------------------

def project_to_patches(weights_per_track, pix_xy, target_size, patch_size):
    """Scatter per-track weights → per-patch soft labels.

    weights_per_track: [N, P]
    pix_xy:            [N, 2]   (frame 0)
    Returns motion_mask [P, H_p, W_p].
    """
    H_p = target_size // patch_size
    W_p = target_size // patch_size
    P = weights_per_track.shape[1]

    px = np.clip((pix_xy[:, 0] / target_size * W_p).astype(np.int64), 0, W_p - 1)
    py = np.clip((pix_xy[:, 1] / target_size * H_p).astype(np.int64), 0, H_p - 1)

    accum = np.zeros((P, H_p, W_p), dtype=np.float32)
    counts = np.zeros((H_p, W_p), dtype=np.float32)
    for n in range(weights_per_track.shape[0]):
        accum[:, py[n], px[n]] += weights_per_track[n]
        counts[py[n], px[n]] += 1.0
    counts = np.maximum(counts, 1e-6)
    motion = accum / counts[None]                                 # [P, H_p, W_p]
    # Renormalize to a soft distribution per patch
    s = motion.sum(axis=0, keepdims=True)
    s = np.where(s > 1e-6, s, 1.0)
    return motion / s


@torch.no_grad()
def sample_dino_features(dino_model, image_uint8, pix_xy, target_size, patch_size, device):
    """Run DINOv2, bilinearly sample features at ``pix_xy`` (pixel coords)."""
    img = torch.from_numpy(image_uint8).float().to(device) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)                        # [1,3,H,W]
    # ImageNet normalization (DINOv2 default)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = (img - mean) / std
    feat = dino_model.get_intermediate_layers(img, n=1)[0]         # [1, N_p, C]
    H_p = target_size // patch_size
    W_p = target_size // patch_size
    C = feat.shape[-1]
    grid = feat.reshape(1, H_p, W_p, C).permute(0, 3, 1, 2)        # [1, C, H_p, W_p]

    # Normalize pixel coords → [-1, 1] for grid_sample
    x = pix_xy[:, 0] / target_size * 2 - 1
    y = pix_xy[:, 1] / target_size * 2 - 1
    grid_xy = torch.from_numpy(np.stack([x, y], -1)).float().to(device)
    grid_xy = grid_xy.view(1, 1, -1, 2)                            # [1, 1, N, 2]
    samp = F.grid_sample(grid, grid_xy, mode="bilinear",
                         align_corners=False)                       # [1, C, 1, N]
    return samp.squeeze().T.cpu().numpy()                          # [N, C]


def process_cam(cam_dir, scene_root, args, models):
    """Process a single (scene, cam). Saves motion_cache.npz next to ``images/``."""
    out_path = cam_dir / "motion_cache.npz"
    if out_path.exists() and not args.overwrite:
        print(f"  skip (exists): {out_path}")
        return

    fids, img_paths = get_frame_paths(cam_dir)
    if len(fids) < 2:
        print(f"  skip (only {len(fids)} frame): {cam_dir}")
        return

    if args.num_frames and len(fids) > args.num_frames:
        # CoWTracker (and every 2D point tracker) assumes temporally-adjacent
        # frames — subsampling with np.linspace creates multi-frame gaps that
        # silently break tracking. Use a *contiguous* window instead; centered
        # on the video midpoint to bias toward frames where the articulated
        # motion is most developed.
        N = args.num_frames
        start = max(0, (len(fids) - N) // 2)
        sel = list(range(start, start + N))
        fids      = [fids[i] for i in sel]
        img_paths = [img_paths[i] for i in sel]

    # ── Load video / depth / K / extrinsics ────────────────────────────
    video, (orig_w, orig_h) = load_video(img_paths, args.target_size)   # [S,3,H,W]
    S, _, H, W = video.shape

    depth_dir = cam_dir / "depth"
    depths = []
    for fid in fids:
        p = depth_dir / f"{fid}.npy"
        if not p.exists():
            print(f"  skip (no depth for {fid}): {cam_dir}")
            return
        depths.append(load_depth(str(p), args.target_size))
    depths = np.stack(depths)                                          # [S, H, W]

    K = load_intrinsics(str(cam_dir / "intrinsics.txt"),
                        orig_w, orig_h, args.target_size)              # [3,3]
    extr_dir = cam_dir / "extrinsics"
    extr = np.stack([
        np.loadtxt(str(extr_dir / f"{fid}.txt")).astype(np.float32)
        for fid in fids
    ])                                                                  # [S,4,4]

    # ── Step 1: CoWTracker dense tracking ──────────────────────────────
    # CoWTracker's fnet downsamples by 8 / 16, so its input H,W must be a
    # multiple of 32. DINO uses target_size (518) but tracker runs at
    # ``tracker_size`` (default 512), then we rescale tracks back.
    cowtracker, dino = models["cowtracker"], models["dino"]
    device = args.device
    ts = args.tracker_size
    assert ts % 224 == 0, f"tracker_size ({ts}) must be a multiple of 224 (LCM of patch=14 and stride=32)"
    scale = float(args.target_size) / float(ts)

    video_t = torch.from_numpy(video).float().to(device)               # [S,3,H,W] @ target_size
    video_cow = F.interpolate(video_t, size=(ts, ts), mode="bilinear",
                              align_corners=False)                     # [S,3,ts,ts]
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        pred = cowtracker.forward(video=video_cow, queries=None)
    track_dense = pred["track"][0].float().cpu().numpy()               # [S,ts,ts,2]
    vis_dense   = pred["vis"][0].float().cpu().numpy()                 # [S,ts,ts]
    conf_dense  = pred["conf"][0].float().cpu().numpy()                # [S,ts,ts]
    del pred, video_cow; torch.cuda.empty_cache()

    # Rescale pixel coords from tracker frame → target_size frame
    track_dense = track_dense * scale

    # ── Step 2: stride subsample + filter ──────────────────────────────
    stride = args.stride
    t2d = track_dense[:, ::stride, ::stride].reshape(S, -1, 2)         # [S, N, 2]
    vis = vis_dense[:, ::stride, ::stride].reshape(S, -1)              # [S, N]
    conf = conf_dense[:, ::stride, ::stride].reshape(S, -1)
    visconf = vis * conf
    valid = (visconf > 0.1).all(axis=0)                                # [N]
    # Also filter tracks whose frame-0 depth is invalid
    pix0 = t2d[0]                                                       # [N, 2]
    d0 = sample_depth(depths[0], pix0)                                  # [N]
    valid &= (d0 > 0.05) & (d0 < 50.0)
    t2d = t2d[:, valid]
    vis = vis[:, valid]
    N = t2d.shape[1]
    if N < 64:
        print(f"  skip (only {N} valid tracks): {cam_dir}")
        return
    if N > args.max_tracks:
        # Keep the most-confident max_tracks
        score = visconf[:, valid].mean(axis=0)
        keep = np.argsort(-score)[: args.max_tracks]
        t2d = t2d[:, keep]
        vis = vis[:, keep]
        N = args.max_tracks

    # ── Step 3: lift to 3D using per-frame depth ────────────────────────
    tracks_3d = np.zeros((S, N, 3), dtype=np.float32)
    for t in range(S):
        d = sample_depth(depths[t], t2d[t])                            # [N]
        # Tracks where current-frame depth invalid → mark vis = 0
        bad = (d < 0.05) | (d > 50.0)
        vis[t][bad] = 0.0
        d = np.where(bad, 1.0, d)                                       # placeholder
        tracks_3d[t] = unproject_to_world(t2d[t], d, K, extr[t])

    # ── Step 4: motion-residual KMeans init (w/ optional DINO tie-breaker).
    #    Old path (DINO-only) collapsed all articulated parts to slot 0
    #    because parts share texture. See `motion_residual_init` docstring.
    dino_feat = None
    if args.dino_weight > 0:
        dino_feat = sample_dino_features(
            dino, video[0].transpose(1, 2, 0), t2d[0],
            args.target_size, args.patch_size, device,
        )                                                                # [N, C_d]
    w_init = motion_residual_init(
        tracks_3d, vis, args.num_parts,
        dino_feat=dino_feat, dino_weight=args.dino_weight,
        temp=0.3, n_iters=30, seed=0,
    )                                                                    # [N, P]

    # ── Step 5: optimize ────────────────────────────────────────────────
    w = optimize_part_weights(
        tracks_3d, vis, w_init,
        n_outer=args.n_outer, n_inner=args.n_inner,
        lr=args.lr, lambda_ent=0.01, lambda_init=0.05,
        device=device,
    )                                                                    # [N, P]

    # ── Step 6: scatter → motion_mask ───────────────────────────────────
    motion_mask = project_to_patches(w, t2d[0], args.target_size, args.patch_size)

    # ── Save ────────────────────────────────────────────────────────────
    tracks_2d_norm = t2d.copy()
    tracks_2d_norm[..., 0] /= args.target_size
    tracks_2d_norm[..., 1] /= args.target_size

    np.savez_compressed(
        str(out_path),
        frame_ids=np.array(fids),
        tracks_2d_norm=tracks_2d_norm.astype(np.float32),
        tracks_3d=tracks_3d.astype(np.float32),
        tracks_vis=vis.astype(np.float32),
        track_part_label=w.astype(np.float32),
        motion_mask=motion_mask.astype(np.float32),
    )
    print(f"  saved {out_path}  N={N}  parts={args.num_parts}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cowtracker(ckpt_path, device):
    from cowtracker import CoWTracker
    model = CoWTracker.from_checkpoint(
        ckpt_path, device=device,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model.eval()
    return model


def load_dinov2(device, weights_from=None, img_size=518, patch_size=14):
    """Build a local DINOv2 ViT-L/14 + reg4 (no download) and initialize its
    weights from a VGGT-style checkpoint (``aggregator.patch_embed.*``)."""
    from dggt.layers.vision_transformer import vit_large

    model = vit_large(
        patch_size=patch_size,
        num_register_tokens=4,
        img_size=img_size,
        block_chunks=0,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    if weights_from is not None:
        ck = torch.load(str(weights_from), map_location="cpu")
        sd = ck.get("model", ck)
        prefix = "aggregator.patch_embed."
        dino_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(dino_sd, strict=False)
        n_loaded = len(dino_sd) - len(unexpected)
        print(f"  DINOv2 loaded {n_loaded} tensors from {weights_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--tracker_size", type=int, default=448,
                    help="Resolution fed to CoWTracker. Must be a multiple of "
                         "LCM(patch=14, stride=32) = 224. Tracks are rescaled "
                         "back to target_size after.")
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--num_frames", type=int, default=None,
                    help="Subsample to this many frames (None = use all).")
    ap.add_argument("--num_parts", type=int, default=8,
                    help="Number of slots / clusters (= max_parts in dataset).")
    ap.add_argument("--max_tracks", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--n_outer", type=int, default=10)
    ap.add_argument("--n_inner", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cowtracker_ckpt", type=str, default=None)
    ap.add_argument("--dino_weights", type=str,
                    default="/data2/cyt/checkpoints/art_v20_phase1a/ckpt_warmup_003500.pth",
                    help="VGGT-style checkpoint — DINOv2 is extracted from "
                         "aggregator.patch_embed.* keys. Avoids torch.hub download.")
    ap.add_argument("--dino_weight", type=float, default=0.3,
                    help="Blend weight for DINO appearance features in the "
                         "motion-residual init (0 = pure motion, higher = more "
                         "appearance bias). Set 0 to skip DINO entirely.")
    ap.add_argument("--exclude_cams", type=str, nargs="*", default=["cam_00"])
    ap.add_argument("--scene_glob", type=str, default="*",
                    help="Filter scenes (glob on scene_id).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N (scene, cam) pairs (debug).")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    assert data_root.is_dir(), data_root

    print("Loading models...")
    cowtracker = load_cowtracker(args.cowtracker_ckpt, args.device)
    dino       = load_dinov2(args.device, weights_from=args.dino_weights,
                             img_size=args.target_size, patch_size=args.patch_size)
    models     = {"cowtracker": cowtracker, "dino": dino}
    print("Models loaded.")

    # Discover (scene, cam) pairs (multi-cam format only — single-cam not supported here).
    scenes = sorted([
        s for s in data_root.iterdir()
        if s.is_dir() and (s / "joint_params.json").exists()
        and s.match(args.scene_glob)
    ])
    pairs = []
    for s in scenes:
        for cam in list_cam_dirs(s, set(args.exclude_cams)):
            pairs.append((s, cam))
    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"Total (scene, cam) pairs: {len(pairs)}")

    t0 = time.time()
    for i, (scene, cam) in enumerate(pairs):
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-3)
        eta = (len(pairs) - i - 1) / max(rate, 1e-3) / 60.0
        print(f"[{i+1}/{len(pairs)}] {scene.name}/{cam.name}  "
              f"({rate:.2f} pairs/s, ETA {eta:.1f} min)")
        try:
            process_cam(cam, scene, args, models)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
