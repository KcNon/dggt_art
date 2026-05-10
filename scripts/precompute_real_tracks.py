"""
precompute_real_tracks.py — CoTracker3-based offline track precomputation
for /data2/cyt/video2articulation/real_data/.

Output (per scene):
    <prep>/motion_cache_real.npz
        frame_ids        : [S_full]   list of frame indices used (covers full video)
        tracks_2d_norm   : [S_full, N_raw, 2]   normalized [0, 1] in (x, y)
        tracks_3d        : [S_full, N_raw, 3]   metric world-space (NaN where invalid)
        tracks_vis       : [S_full, N_raw]      0/1
        track_part_label : [N_raw, P]           soft labels at frame 0 (sum=1 per row)
        motion_mask      : [P, h_p, w_p]        coarse part-region prior

Where P = max_parts (default 8). Slot 0 = background; slots 1..k = the same
top-(P-1) SAM2 instances chosen by the dataset's _load_masks scoring.
This way the cache aligns 1:1 with what iTACORealDataset returns at training time.

Usage:
    python scripts/precompute_real_tracks.py \
        --data_root /data2/cyt/video2articulation/real_data \
        --frame_stride 5 \
        --grid_size 32 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
def load_video(rgb_dir: Path, fids: list[int]) -> torch.Tensor:
    """Returns video [T, 3, H, W] uint8."""
    imgs = []
    for fid in fids:
        p = rgb_dir / f"{fid:06d}.jpg"
        imgs.append(np.array(Image.open(p).convert("RGB"), dtype=np.uint8))
    return torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2)   # [T, 3, H, W]


def load_dyn_union(prep: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Union of MonST3R dynamic-mask keyframes, resized to target."""
    files = sorted((prep / "monst3r").glob("dynamicmask_*.png"))
    if not files:
        return None
    union = None
    for fp in files:
        m = np.array(Image.open(fp).convert("L")) > 127
        union = m if union is None else (m | union if m.shape == union.shape else union)
    if union is None:
        return None
    H_t, W_t = target_hw
    pil = Image.fromarray(union.astype(np.uint8) * 255).resize((W_t, H_t), Image.NEAREST)
    return np.array(pil) > 127


def load_segmentation_for_frame(prep: Path, fid: int) -> np.ndarray | None:
    """[P_inst, H, W] bool from SAM2 npz at frame fid, or None."""
    seg_dir = prep / "video_segment_reverse" / "small" / "final-output"
    p = seg_dir / f"mask_{fid:03d}.npz"
    if not p.exists():
        p = seg_dir / f"mask_{fid:06d}.npz"
    if not p.exists():
        return None
    arr = np.load(p)
    arr = arr[arr.files[0]]                 # [P_inst, 1, H, W]
    return arr.squeeze(1).astype(bool)


def rank_instances(seg_per_frame: list[np.ndarray | None],
                   dyn: np.ndarray | None,
                   max_parts: int) -> list[int]:
    """
    Mirror of iTACORealDataset._load_masks ranking:
        score = (instance ∩ dyn ? 1e9 : 0) + total area across frames
    Returns list of inst_ids sorted by descending score (length = top-K = max_parts-1).
    """
    valids = [s for s in seg_per_frame if s is not None]
    if not valids:
        return []
    P_inst = max(s.shape[0] for s in valids)
    H_o, W_o = valids[0].shape[1:]

    normed = []
    for s in seg_per_frame:
        if s is None:
            normed.append(np.zeros((P_inst, H_o, W_o), dtype=bool))
        elif s.shape[0] < P_inst:
            pad = np.zeros((P_inst - s.shape[0], H_o, W_o), dtype=bool)
            normed.append(np.concatenate([s, pad], axis=0))
        else:
            normed.append(s[:P_inst])
    stacked = np.stack(normed, axis=0)                    # [T, P_inst, H, W]
    area = stacked.reshape(stacked.shape[0], P_inst, -1).sum(axis=(0, 2)).astype(np.float64)
    score = area.copy()
    if dyn is not None and dyn.shape == (H_o, W_o):
        for i in range(P_inst):
            hit = (stacked[:, i].any(axis=0) & dyn).sum()
            if hit > 0:
                score[i] += 1e9
    return np.argsort(-score)[: max_parts - 1].tolist()


def back_project(uv: np.ndarray,           # [N, 2] pixel
                 depth: np.ndarray,         # [H, W] metric
                 K: np.ndarray,             # [3, 3]
                 c2w: np.ndarray,           # [4, 4]
                 ) -> np.ndarray:
    """[N, 3] world coords; (NaN, NaN, NaN) where depth invalid (≤0)."""
    H, W = depth.shape
    u = uv[:, 0].astype(np.int64).clip(0, W - 1)
    v = uv[:, 1].astype(np.int64).clip(0, H - 1)
    z = depth[v, u]                                      # [N]
    valid = z > 1e-3
    out = np.full((uv.shape[0], 3), np.nan, dtype=np.float32)
    if not valid.any():
        return out
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    x_cam = (uv[valid, 0] - cx) * z[valid] / fx
    y_cam = (uv[valid, 1] - cy) * z[valid] / fy
    z_cam = -z[valid]                                     # OpenGL: z<0 in front
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    pts_w = (c2w @ pts_cam.T).T                           # [n, 4]
    out[valid] = pts_w[:, :3].astype(np.float32)
    return out


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w);     R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w);     R[..., 2, 1] = 2 * (y * z + x * w);     R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def pose_xyzquat_to_c2w(p7: np.ndarray) -> np.ndarray:
    T = p7.shape[0]
    M = np.zeros((T, 4, 4), dtype=np.float32)
    M[:, :3, :3] = quat_wxyz_to_R(p7[:, 3:7].astype(np.float32))
    M[:, :3,  3] = p7[:, :3].astype(np.float32)
    M[:,  3,  3] = 1.0
    return M


# ─────────────────────────────────────────────────────────────────────────────
def process_scene(
    raw_dir: Path,
    prep_dir: Path,
    out_path: Path,
    grid_size: int,
    frame_stride: int,
    max_parts: int,
    device: torch.device,
    cotracker_model,
):
    print(f"\n[scene {raw_dir.name}]")
    meta = json.load(open(raw_dir / "metadata.json"))
    T_meta = len(meta["poses"])
    rgb_files = sorted((raw_dir / "rgb").glob("*.jpg"))
    T_rgb  = len(rgb_files)
    T_full = min(T_meta, T_rgb)

    # Sample frame ids at stride
    fids = list(range(0, T_full, frame_stride))
    if fids[-1] != T_full - 1:
        fids.append(T_full - 1)
    S = len(fids)
    print(f"  T_full={T_full}  using S={S} frames at stride={frame_stride}")

    # Load video
    video = load_video(raw_dir / "rgb", fids).to(device).float()      # [S, 3, H, W] in [0, 255]
    video = video.unsqueeze(0)                                          # CoTracker expects [B, T, 3, H, W]
    H, W = video.shape[-2:]

    # ── Run CoTracker3 ─────────────────────────────────────────────────
    queries = None
    print(f"  running CoTracker3 (grid={grid_size}, H={H}, W={W}) ...")
    with torch.no_grad():
        tracks, vis = cotracker_model(
            video,
            grid_size=grid_size,
            queries=queries,
        )
    # tracks: [1, S, N_raw, 2]  in pixel coords (x, y)  on the network's input scale
    # vis:    [1, S, N_raw]     bool
    tracks = tracks[0].cpu().numpy().astype(np.float32)                 # [S, N, 2]
    vis    = vis[0].cpu().numpy().astype(np.float32)                    # [S, N]
    N_raw  = tracks.shape[1]
    print(f"  → {N_raw} tracks")

    # Normalised xy ∈ [0, 1] (per current resolution)
    tracks_2d_norm = tracks.copy()
    tracks_2d_norm[..., 0] /= max(W - 1, 1)
    tracks_2d_norm[..., 1] /= max(H - 1, 1)

    # ── Pose / intrinsics ──────────────────────────────────────────────
    poses = np.array(meta["poses"], dtype=np.float32)
    c2w_all = pose_xyzquat_to_c2w(poses)                                # [T_meta, 4, 4]

    if "perFrameIntrinsicCoeffs" in meta and meta["perFrameIntrinsicCoeffs"]:
        coeffs = np.array(meta["perFrameIntrinsicCoeffs"], dtype=np.float32)
        fx, fy, cx, cy = coeffs.mean(axis=0)
    else:
        K_meta = np.array(meta["K"], dtype=np.float32).reshape(3, 3).T
        fx, fy = K_meta[0, 0], K_meta[1, 1]
        cx, cy = K_meta[0, 2], K_meta[1, 2]

    H_orig, W_orig = meta["h"], meta["w"]
    sx, sy = W / W_orig, H / H_orig
    K_at_video_res = np.array([[fx * sx, 0, cx * sx],
                                [0, fy * sy, cy * sy],
                                [0, 0,        1.0   ]], dtype=np.float32)

    # ── 3D back-projection per frame ───────────────────────────────────
    depth_dir = prep_dir / "prompt_depth_video"
    tracks_3d = np.full((S, N_raw, 3), np.nan, dtype=np.float32)
    if depth_dir.exists():
        for s_idx, fid in enumerate(fids):
            p = depth_dir / f"{fid:06d}.npy"
            if not p.exists():
                continue
            d = np.load(p).astype(np.float32)                            # [H_orig, W_orig]
            if d.shape != (H, W):
                d_t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
                d = F.interpolate(d_t, size=(H, W), mode="nearest").squeeze().numpy()
            tracks_3d[s_idx] = back_project(
                tracks[s_idx], d, K_at_video_res, c2w_all[fid],
            )

    # ── Per-track part label (sampled at frame 0) ──────────────────────
    seg_per_frame = [load_segmentation_for_frame(prep_dir, fid) for fid in fids]
    dyn = load_dyn_union(prep_dir, (H, W))
    top_inst_ids = rank_instances(seg_per_frame, dyn, max_parts)

    P = max_parts
    track_part_label = np.zeros((N_raw, P), dtype=np.float32)
    if top_inst_ids and seg_per_frame[0] is not None:
        seg0 = seg_per_frame[0]                                          # [P_inst, H_o, W_o]
        H_o, W_o = seg0.shape[1:]
        # query pixel position at frame 0; resize to seg's resolution
        u0 = (tracks[0, :, 0] / max(W - 1, 1) * (W_o - 1)).astype(np.int64).clip(0, W_o - 1)
        v0 = (tracks[0, :, 1] / max(H - 1, 1) * (H_o - 1)).astype(np.int64).clip(0, H_o - 1)
        for slot_j, inst_id in enumerate(top_inst_ids):
            sel = seg0[inst_id, v0, u0]                                  # [N_raw] bool
            track_part_label[sel, slot_j + 1] = 1.0
    track_part_label[track_part_label.sum(-1) == 0, 0] = 1.0             # bg slot
    # Renormalise to sum=1
    track_part_label /= track_part_label.sum(-1, keepdims=True).clip(min=1e-6)

    # ── motion_mask [P, h_p, w_p] ───────────────────────────────────────
    h_p, w_p = H // 14, W // 14                                          # patch grid
    motion_mask = np.zeros((P, h_p, w_p), dtype=np.float32)
    if top_inst_ids and seg_per_frame[0] is not None:
        seg0 = seg_per_frame[0]
        for slot_j, inst_id in enumerate(top_inst_ids):
            m = seg0[inst_id].astype(np.float32)
            mt = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            mt = F.interpolate(mt, size=(h_p, w_p), mode="bilinear", align_corners=False)
            motion_mask[slot_j + 1] = mt.squeeze().clamp(0, 1).numpy()
    if dyn is not None:
        dyn_t = torch.from_numpy(dyn.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        dyn_t = F.interpolate(dyn_t, size=(h_p, w_p), mode="bilinear", align_corners=False)
        motion_mask[0] = (1.0 - dyn_t.squeeze().clamp(0, 1).numpy())     # bg = 1 - dyn

    # ── Save ────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_ids        = np.array([f"{f:06d}" for f in fids]),
        tracks_2d_norm   = tracks_2d_norm,
        tracks_3d        = tracks_3d,
        tracks_vis       = vis,
        track_part_label = track_part_label,
        motion_mask      = motion_mask,
    )
    print(f"  saved → {out_path}  (S={S}, N={N_raw}, P={P})")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="path to /data2/cyt/video2articulation/real_data")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="explicit scene list; default = book cabinet drawer storage")
    p.add_argument("--grid_size", type=int, default=32,
                   help="CoTracker3 dense query grid edge (N = grid_size**2)")
    p.add_argument("--frame_stride", type=int, default=5,
                   help="stride along the original video; smaller = more frames in cache")
    p.add_argument("--max_parts", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache_name", default="motion_cache_real.npz")
    p.add_argument("--cotracker_variant", default="cotracker3_offline",
                   help="torch.hub model name (cotracker3_offline / cotracker3_online)")
    return p.parse_args()


def main():
    cfg = parse_args()
    root = Path(cfg.data_root)
    scenes = cfg.scenes or ["book", "cabinet", "drawer", "storage"]
    device = torch.device(cfg.device)

    print("[cotracker] downloading / loading via torch.hub …")
    cotracker = torch.hub.load(
        "facebookresearch/co-tracker", cfg.cotracker_variant
    ).to(device).eval()
    print("[cotracker] ready")

    for cat in scenes:
        raw_dir  = root / "raw_data" / cat
        prep_dir = root / "exp_results" / "preprocessing" / cat
        if not (raw_dir / "metadata.json").exists():
            print(f"[skip] {cat}: no metadata.json"); continue
        if not (raw_dir / "rgb").exists():
            print(f"[skip] {cat}: no rgb/"); continue
        out_path = prep_dir / cfg.cache_name
        try:
            process_scene(
                raw_dir, prep_dir, out_path,
                grid_size=cfg.grid_size,
                frame_stride=cfg.frame_stride,
                max_parts=cfg.max_parts,
                device=device,
                cotracker_model=cotracker,
            )
        except Exception as e:
            import traceback
            print(f"[scene {cat}] FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
