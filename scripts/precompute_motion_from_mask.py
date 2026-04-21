#!/usr/bin/env python3
"""Generate per-scene motion pseudo-labels directly from GT part masks + joint params.

Use this when CoWTracker-based clustering is unreliable (e.g. complex multi-joint
scenes). Output schema is identical to ``precompute_motion_data.py`` so the
training pipeline can consume either source transparently.

For each (scene, cam):
  1. read joint_params (axis, pivot, type) and joint_angles per frame
  2. on frame 0, sample N points per dynamic part inside its GT mask, plus
     N points inside the static root region
  3. unproject each point to world via depth + K + cam_to_world
  4. for every subsequent frame, transform world points by the per-joint
     rigid motion implied by (angle_t - angle_0); static points stay put
  5. project back to image at frame t using K + cam_to_world_t
  6. visibility = in-image AND |z_proj - depth_at_proj| < tau (or no depth lookup)
  7. add configurable noise (pixel xy gaussian, vis dropout, label smoothing)

Output: ``{cam_dir}/motion_cache_gt.npz`` with same keys as cowtracker variant.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_intrinsics(path, orig_w, orig_h, target_size):
    K = np.loadtxt(path).astype(np.float32)
    K[0, 0] *= target_size / orig_w
    K[1, 1] *= target_size / orig_h
    K[0, 2] *= target_size / orig_w
    K[1, 2] *= target_size / orig_h
    return K


def load_extrinsic(path):
    return np.loadtxt(path).astype(np.float32)   # [4, 4]  cam_to_world


def load_depth(path, target_size):
    d = np.load(path).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    H_o, W_o = d.shape
    if (H_o, W_o) != (target_size, target_size):
        im = Image.fromarray(d).resize((target_size, target_size), Image.NEAREST)
        d = np.asarray(im, dtype=np.float32)
    return d


def load_mask(path, target_size):
    if not Path(path).exists():
        return np.zeros((target_size, target_size), dtype=np.uint8)
    m = np.asarray(Image.open(path).convert("L"))
    if m.shape != (target_size, target_size):
        m = np.asarray(Image.fromarray(m).resize(
            (target_size, target_size), Image.NEAREST))
    return (m > 127).astype(np.uint8)


def get_frame_ids(cam_dir):
    img_dir = cam_dir / "images"
    paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return [p.stem for p in paths]


def img_orig_size(cam_dir, fid):
    for ext in ("jpg", "png"):
        p = cam_dir / "images" / f"{fid}.{ext}"
        if p.exists():
            with Image.open(p) as im:
                return im.size  # (W, H)
    raise FileNotFoundError(f"No image for {fid}")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def unproject(pix_xy, depth_vals, K, c2w):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pix_xy[:, 0] - cx) / fx * depth_vals
    y = (pix_xy[:, 1] - cy) / fy * depth_vals
    z = depth_vals
    cam = np.stack([x, y, z], axis=-1)
    R, t = c2w[:3, :3], c2w[:3, 3]
    return cam @ R.T + t


def project(world_pts, K, c2w):
    """world → 2D pixel + cam-z. world_pts: [N, 3]"""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (world_pts - t) @ R           # cam = R^T @ (world - t)
    z = cam[:, 2]
    eps = 1e-6
    u = K[0, 0] * cam[:, 0] / np.clip(z, eps, None) + K[0, 2]
    v = K[1, 1] * cam[:, 1] / np.clip(z, eps, None) + K[1, 2]
    return np.stack([u, v], axis=-1), z


def rodrigues(axis, angle):
    """[3] axis (unit), scalar angle → [3, 3] R."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def apply_joint(pts_world, jtype, axis, pivot, delta):
    """Transform points in part frame given delta angle/displacement."""
    if jtype == "revolute":
        R = rodrigues(np.asarray(axis, np.float32), float(delta))
        return (pts_world - pivot) @ R.T + pivot
    if jtype == "prismatic":
        return pts_world + float(delta) * np.asarray(axis, np.float32)[None, :]
    return pts_world


# ---------------------------------------------------------------------------
# Per (scene, cam) processing
# ---------------------------------------------------------------------------

def process_cam(scene_dir, cam_dir, args):
    fids = get_frame_ids(cam_dir)
    if len(fids) == 0:
        return None
    S = len(fids)
    T = args.target_size

    # joint params + angles
    with open(scene_dir / "joint_params.json") as f:
        jp_raw = json.load(f)
    with open(scene_dir / "joint_angles.json") as f:
        ja_raw = json.load(f)
    joint_keys = sorted(jp_raw.keys())   # ["joint_0", ...]
    n_joints = len(joint_keys)

    # Slot 0 = static (root + bg). Dynamic slots: p = k+1.
    P = n_joints + 1

    # Per-frame angle arrays
    ang = np.zeros((n_joints, S), dtype=np.float32)
    for k, jk in enumerate(joint_keys):
        for s, fid in enumerate(fids):
            v = ja_raw.get(fid, 0.0)
            if isinstance(v, dict):
                ang[k, s] = float(v.get(jk, 0.0))
            else:
                ang[k, s] = float(v) if k == 0 else 0.0
    delta = ang - ang[:, :1]   # [n_joints, S], frame-0 anchored

    # Intrinsics
    W_o, H_o = img_orig_size(cam_dir, fids[0])
    K = load_intrinsics(cam_dir / "intrinsics.txt", W_o, H_o, T)

    # Frame-0 depth + extrinsic
    d0 = load_depth(cam_dir / "depth" / f"{fids[0]}.npy", T)
    e0 = load_extrinsic(cam_dir / "extrinsics" / f"{fids[0]}.txt")

    # Per-frame extrinsics
    cams = np.stack([load_extrinsic(cam_dir / "extrinsics" / f"{fid}.txt") for fid in fids], 0)

    # Sample query points per part
    rng = np.random.default_rng(args.seed)
    points_per_part = args.max_tracks // P
    masks_dir = cam_dir / "part_masks"

    pts_world_all = []
    part_label_all = []   # one-hot row → [N, P]
    for p in range(P):
        if p == 0:
            # static = root (id = n_joints + 2)
            root_id = n_joints + 2
            mk = load_mask(masks_dir / f"{fids[0]}_{root_id}.png", T)
        else:
            k = p - 1
            part_id = k + 2
            mk = load_mask(masks_dir / f"{fids[0]}_{part_id}.png", T)

        valid = (mk > 0) & (d0 > 1e-3)
        ys, xs = np.where(valid)
        if len(xs) == 0:
            continue
        n_take = min(points_per_part, len(xs))
        sel = rng.choice(len(xs), size=n_take, replace=False)
        px = xs[sel].astype(np.float32) + 0.5
        py = ys[sel].astype(np.float32) + 0.5
        depths = d0[ys[sel], xs[sel]]
        pix = np.stack([px, py], -1)
        wpts = unproject(pix, depths, K, e0)
        pts_world_all.append(wpts)
        lbl = np.zeros((n_take, P), dtype=np.float32)
        lbl[:, p] = 1.0
        part_label_all.append(lbl)

    if not pts_world_all:
        return None

    pts0 = np.concatenate(pts_world_all, 0)   # [N, 3] world frame at angle 0
    track_part_label = np.concatenate(part_label_all, 0)  # [N, P]
    N = pts0.shape[0]
    part_idx = track_part_label.argmax(-1)   # [N]

    tracks_2d_norm = np.zeros((S, N, 2), dtype=np.float32)
    tracks_3d      = np.zeros((S, N, 3), dtype=np.float32)
    tracks_vis     = np.zeros((S, N), dtype=np.float32)

    for s in range(S):
        # Build per-part rigid transform → apply to all points of that part
        wpts = pts0.copy()
        for p in range(1, P):
            k = p - 1
            jdata = jp_raw[joint_keys[k]]
            jtype = jdata["type"]
            axis = np.asarray(jdata["axis"], np.float32)
            pivot = np.asarray(jdata.get("pivot", [0, 0, 0]), np.float32)
            mask_p = (part_idx == p)
            if not np.any(mask_p):
                continue
            wpts[mask_p] = apply_joint(pts0[mask_p], jtype, axis, pivot, delta[k, s])

        tracks_3d[s] = wpts
        pix, z = project(wpts, K, cams[s])
        tracks_2d_norm[s, :, 0] = pix[:, 0] / max(T - 1, 1)
        tracks_2d_norm[s, :, 1] = pix[:, 1] / max(T - 1, 1)

        # vis: in-image + depth consistency (if depth file present)
        in_img = (pix[:, 0] >= 0) & (pix[:, 0] < T) & (pix[:, 1] >= 0) & (pix[:, 1] < T) & (z > 0)
        vis = in_img.copy()
        try:
            d_s = load_depth(cam_dir / "depth" / f"{fids[s]}.npy", T)
            ix = np.clip(pix[:, 0].astype(np.int32), 0, T - 1)
            iy = np.clip(pix[:, 1].astype(np.int32), 0, T - 1)
            d_at = d_s[iy, ix]
            depth_ok = (d_at > 1e-3) & (np.abs(d_at - z) < args.depth_tau)
            vis = vis & depth_ok
        except FileNotFoundError:
            pass
        tracks_vis[s] = vis.astype(np.float32)

    # Add noise --------------------------------------------------------------
    if args.noise_xy_px > 0:
        sigma = args.noise_xy_px / max(T - 1, 1)
        tracks_2d_norm += rng.normal(0, sigma, size=tracks_2d_norm.shape).astype(np.float32)
        tracks_2d_norm = np.clip(tracks_2d_norm, 0.0, 1.0)
    if args.vis_drop > 0:
        keep = rng.random(tracks_vis.shape) > args.vis_drop
        tracks_vis = tracks_vis * keep.astype(np.float32)
    if args.label_smooth > 0:
        eps = args.label_smooth
        track_part_label = track_part_label * (1 - eps) + eps / P

    # motion_mask: GT first-frame masks downsampled to patch grid
    H_p = T // args.patch_size
    W_p = T // args.patch_size
    motion_mask = np.zeros((P, H_p, W_p), dtype=np.float32)
    for p in range(P):
        if p == 0:
            mk = load_mask(masks_dir / f"{fids[0]}_{n_joints + 2}.png", T)
        else:
            mk = load_mask(masks_dir / f"{fids[0]}_{p + 1}.png", T)
        # average-pool patch_size×patch_size
        mk_f = mk.astype(np.float32).reshape(H_p, args.patch_size, W_p, args.patch_size).mean((1, 3))
        motion_mask[p] = mk_f
    # Slot 0 augment with background
    bg = 1.0 - np.clip(motion_mask[1:].sum(0), 0, 1) if n_joints else np.ones_like(motion_mask[0])
    motion_mask[0] = np.clip(motion_mask[0] + bg, 0, 1)
    # Normalize across slots so each patch sums to 1 (soft label)
    s = motion_mask.sum(0, keepdims=True)
    motion_mask = motion_mask / np.clip(s, 1e-6, None)

    return {
        "frame_ids":        np.array(fids),
        "tracks_2d_norm":   tracks_2d_norm.astype(np.float32),
        "tracks_3d":        tracks_3d.astype(np.float32),
        "tracks_vis":       tracks_vis.astype(np.float32),
        "track_part_label": track_part_label.astype(np.float32),
        "motion_mask":      motion_mask.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(cam_dir, out_path, payload, max_frames=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fids = payload["frame_ids"]
    tracks = payload["tracks_2d_norm"]
    vis = payload["tracks_vis"]
    lbl = payload["track_part_label"]
    P = lbl.shape[-1]
    part_idx = lbl.argmax(-1)
    cmap = plt.get_cmap("tab10")

    n = min(max_frames, len(fids))
    idxs = np.linspace(0, len(fids) - 1, n).astype(int)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for j, s in enumerate(idxs):
        fid = fids[s]
        img_path = None
        for ext in ("jpg", "png"):
            p = cam_dir / "images" / f"{fid}.{ext}"
            if p.exists():
                img_path = p; break
        ax = axes[j]
        T = tracks.shape[0]  # not used; we want render size
        T_show = 518
        if img_path is not None:
            im = Image.open(img_path).convert("RGB").resize((T_show, T_show), Image.BILINEAR)
            im = np.asarray(im)
            ax.imshow(im)
            H, W = im.shape[:2]
        else:
            H = W = T_show
        x = tracks[s, :, 0] * (W - 1)
        y = tracks[s, :, 1] * (H - 1)
        v = vis[s] > 0.5
        for p in range(P):
            mask = (part_idx == p) & v
            if mask.sum() == 0: continue
            ax.scatter(x[mask], y[mask], s=4, c=[cmap(p % 10)], alpha=0.7,
                       label=f"slot {p}")
        ax.set_title(f"f{fid} vis={int(v.sum())}/{len(v)}")
        ax.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(P, 6))
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--scenes", type=str, nargs="*", default=None,
                    help="If set, only process these scene names.")
    ap.add_argument("--cams", type=str, nargs="*", default=["cam_01", "cam_02", "cam_03"])
    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_tracks", type=int, default=4096)
    ap.add_argument("--depth_tau", type=float, default=0.15,
                    help="Depth consistency threshold in meters.")
    ap.add_argument("--noise_xy_px", type=float, default=1.5)
    ap.add_argument("--vis_drop", type=float, default=0.10)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_name", type=str, default="motion_cache_gt.npz")
    ap.add_argument("--vis_dir", type=str, default=None,
                    help="If set, save per-(scene,cam) visualization PNG here.")
    args = ap.parse_args()

    root = Path(args.data_root)
    scene_names = args.scenes if args.scenes else sorted(p.name for p in root.iterdir() if p.is_dir())
    if args.vis_dir:
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)

    n_done = 0
    for sn in scene_names:
        sd = root / sn
        if not (sd / "joint_params.json").exists():
            continue
        for cam_name in args.cams:
            cd = sd / cam_name
            if not cd.exists() or not (cd / "images").exists():
                continue
            try:
                payload = process_cam(sd, cd, args)
            except Exception as e:
                print(f"[SKIP] {sn}/{cam_name}: {e}", file=sys.stderr)
                continue
            if payload is None:
                continue
            out_path = cd / args.out_name
            np.savez_compressed(out_path, **payload)
            n_done += 1
            print(f"[OK] {sn}/{cam_name}  N={payload['tracks_2d_norm'].shape[1]} "
                  f"vis_mean={float(payload['tracks_vis'].mean()):.3f}")
            if args.vis_dir:
                vis_path = Path(args.vis_dir) / f"{sn}_{cam_name}.png"
                visualize(cd, vis_path, payload)
    print(f"Done. {n_done} files written.")


if __name__ == "__main__":
    main()
