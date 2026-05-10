"""
iTACO real-data dataset wrapper.

Source: /data2/cyt/video2articulation/real_data/
  raw_data/{book, cabinet, drawer, storage}/
  exp_results/preprocessing/{book, cabinet, drawer, storage}/

Returns batches in the SAME shape & key set as `ArticulatedDataset`, so
the rest of the training pipeline (model, losses, eval) is unchanged.

Key differences vs sim:
  • No GT kinematic params  → has_kin_gt=False, kin loss weights skipped.
  • Only 4 scenes total      → primarily a Phase-2 / eval target.
  • Long videos (200+ frames) → uniform stride sampling to S frames.
  • SAM2 over-segmented masks (P_inst up to 42) → reduced to 8 channels:
        slot 0  = background
        slots 1..k = top-(max_parts-1) instances by area
        priority = (instance ∈ MonST3R dynamic mask) > area
  • Depth from prompt_depth_video/*.npy (metric, full-res).
  • Pose from metadata.poses (Polycam SLAM, treat as has_pose=True).
  • Tracks from a separate motion_cache_real.npz produced by
        scripts/precompute_real_tracks.py
    If absent → tracks_2d/vis are zeros and has_motion_data=False.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """[..., 4] (w, x, y, z) → [..., 3, 3] rotation matrix.  Normalises q first."""
    qn = np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
    q  = q / qn
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _pose_xyzquat_to_c2w(pose_7: np.ndarray) -> np.ndarray:
    """[T, 7] (xyz + quat_wxyz) → [T, 4, 4] cam-to-world."""
    T = pose_7.shape[0]
    M = np.zeros((T, 4, 4), dtype=np.float32)
    M[:, :3, :3] = _quat_wxyz_to_R(pose_7[:, 3:7].astype(np.float32))
    M[:, :3,  3] = pose_7[:, :3].astype(np.float32)
    M[:,  3,  3] = 1.0
    return M


# ─────────────────────────────────────────────────────────────────────────────
class iTACORealDataset(Dataset):
    """
    Args:
        data_root:        path to /data2/cyt/video2articulation/real_data
        target_size:      square crop / resize edge (default 518)
        num_frames:       S frames sampled uniformly along the video
        max_parts:        keep top-(max_parts-1) instance masks plus background
        max_tracks:       fixed N for tracks_2d/vis padding
        motion_cache_name: filename inside preprocessing/{cat}/, optional
        scenes:           explicit subset of categories;  None = all four
        split:            "train" / "val" / "all"  (val_ratio ignored — only 4 scenes)
        random_start:     if True (train), random first-frame offset; else 0
    """

    CATEGORIES = ("book", "cabinet", "drawer", "storage")

    def __init__(
        self,
        data_root: str,
        target_size: int = 518,
        num_frames: int = 8,
        max_parts: int = 8,
        max_tracks: int = 1024,
        patch_size: int = 14,
        motion_cache_name: str = "motion_cache_real.npz",
        scenes: Optional[list[str]] = None,
        split: str = "all",
        random_start: bool = True,
    ):
        super().__init__()
        self.root             = Path(data_root)
        self.target_size      = target_size
        self.num_frames       = num_frames
        self.max_parts        = max_parts
        self.max_tracks       = max_tracks
        self.patch_size       = patch_size
        self.motion_cache_name = motion_cache_name
        self.random_start     = random_start

        if scenes is None:
            scenes = list(self.CATEGORIES)

        # Trivial split: 3 train / 1 val. Caller can override via `scenes`.
        if split == "val":
            scenes = scenes[-1:]
        elif split == "train":
            scenes = scenes[:-1]
        # else "all"

        self.entries: list[tuple[Path, Path]] = []
        for cat in scenes:
            raw  = self.root / "raw_data" / cat
            prep = self.root / "exp_results" / "preprocessing" / cat
            if (raw / "metadata.json").exists() and (raw / "rgb").exists():
                self.entries.append((raw, prep))
            else:
                print(f"[iTACO] skip {cat}: missing raw/metadata or rgb dir")

        assert self.entries, f"No valid scenes under {data_root}"
        print(f"[iTACO] {len(self.entries)} scene(s): {[r.name for r, _ in self.entries]}")

    def __len__(self):
        return len(self.entries)

    # ------------------------------------------------------------------ #
    def _select_frames(self, n_total: int) -> list[int]:
        """Uniformly stride-sample num_frames indices over [0, n_total)."""
        S = self.num_frames
        if n_total <= S:
            ids = list(range(n_total)) + [n_total - 1] * (S - n_total)
            return ids
        max_i = n_total - 1
        if S > 1:
            stride = max_i / (S - 1)
        else:
            stride = 0
        if self.random_start and S > 1:
            # Random offset, but keep last index ≤ max_i.  Use half-stride
            # offset window centered so last index lies in [max_i - half, max_i].
            offset = np.random.uniform(0, stride * 0.5)
        else:
            offset = 0.0
        ids = [int(round(offset + stride * i)) for i in range(S)]
        ids = [min(max(i, 0), max_i) for i in ids]   # clip to valid range
        return ids

    def _load_rgb(self, raw: Path, fids: list[int], H: int, W: int) -> torch.Tensor:
        imgs = []
        for fid in fids:
            p = raw / "rgb" / f"{fid:06d}.jpg"
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            imgs.append(arr)
        x = np.stack(imgs, axis=0).astype(np.float32) / 255.0      # [S, H_orig, W_orig, 3]
        x = torch.from_numpy(x).permute(0, 3, 1, 2)                # [S, 3, H, W]
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x

    def _load_depth(self, prep: Path, fids: list[int], H: int, W: int) -> tuple[torch.Tensor, bool]:
        depth_dir = prep / "prompt_depth_video"
        if not depth_dir.exists():
            return torch.zeros(self.num_frames, H, W), False
        ds = []
        ok = True
        for fid in fids:
            p = depth_dir / f"{fid:06d}.npy"
            if not p.exists():
                ok = False
                ds.append(np.zeros((H, W), np.float32))
                continue
            d = np.load(p).astype(np.float32)
            t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(H, W), mode="nearest").squeeze(0).squeeze(0)
            ds.append(t.numpy())
        return torch.from_numpy(np.stack(ds, axis=0)), ok

    def _load_masks(
        self, prep: Path, fids: list[int],
        H: int, W: int, dyn_mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Output: [S, max_parts, H, W] float32 in {0, 1}.
        slot 0 = background; slots 1..k = top-(max_parts-1) SAM2 instances.

        Instance ranking (descending priority):
          1) intersects MonST3R dynamic mask (if provided) → highest
          2) total mask area across all S frames

        Per-frame masks are loaded from
            <prep>/video_segment_reverse/small/final-output/mask_NNN.npz
        with key 'a' shape [P_inst, 1, H_orig, W_orig] bool.
        """
        seg_dir = prep / "video_segment_reverse" / "small" / "final-output"
        if not seg_dir.exists():
            return torch.zeros(self.num_frames, self.max_parts, H, W)

        # Load all relevant frames
        per_frame = []                  # list of [P_inst, H_orig, W_orig] bool
        for fid in fids:
            p = seg_dir / f"mask_{fid:03d}.npz"
            if not p.exists():
                # Some categories use 6-digit naming; try fallback
                p = seg_dir / f"mask_{fid:06d}.npz"
            if not p.exists():
                per_frame.append(None)
                continue
            arr = np.load(p)
            arr = arr[arr.files[0]]                      # [P_inst, 1, H, W]
            per_frame.append(arr.squeeze(1))             # [P_inst, H, W]

        # Determine canonical P_inst (use frame with max instances; assume all match)
        valid_ones = [a for a in per_frame if a is not None]
        if not valid_ones:
            return torch.zeros(self.num_frames, self.max_parts, H, W)
        P_inst = max(a.shape[0] for a in valid_ones)
        H_o, W_o = valid_ones[0].shape[1:]

        # Pad each frame to P_inst rows of zeros if shorter
        normed = []
        for a in per_frame:
            if a is None:
                normed.append(np.zeros((P_inst, H_o, W_o), dtype=bool))
            elif a.shape[0] < P_inst:
                pad = np.zeros((P_inst - a.shape[0], H_o, W_o), dtype=bool)
                normed.append(np.concatenate([a, pad], axis=0))
            else:
                normed.append(a[:P_inst])
        stacked = np.stack(normed, axis=0)               # [S, P_inst, H_o, W_o]

        # Score each instance.
        # Strict dynamic = (≥30% of mask pixels fall inside MonST3R dyn-union)
        # Pre-filter: drop tiny instances < 0.5% of image area (noise).
        per_frame_area = stacked.reshape(self.num_frames, P_inst, -1).sum(axis=(0, 2)).astype(np.float64)
        area = per_frame_area
        img_area = H_o * W_o * self.num_frames
        score = area.copy()
        score[area < 0.005 * img_area] = -1.0          # demote tiny

        if dyn_mask is not None and dyn_mask.shape == (H_o, W_o):
            for i in range(P_inst):
                m_any = stacked[:, i].any(axis=0)
                hit = (m_any & dyn_mask).sum()
                m_a = m_any.sum()
                ratio = hit / max(m_a, 1)
                if ratio >= 0.3:
                    score[i] += 1e9
        # Top-(max_parts-1) by score, dropping anything that didn't qualify.
        top = np.argsort(-score)[: self.max_parts - 1]
        top = [t for t in top if score[t] > 0]

        # Build [S, max_parts, H_o, W_o]: slot 0 = bg (1 - union(top)) — leave as zeros for now
        out = np.zeros((self.num_frames, self.max_parts, H_o, W_o), dtype=np.float32)
        for j, inst_id in enumerate(top):
            out[:, j + 1] = stacked[:, inst_id].astype(np.float32)
        # Resize to (H, W)
        t = torch.from_numpy(out)                         # [S, P, H_o, W_o]
        t = F.interpolate(t, size=(H, W), mode="nearest")
        return t

    def _load_dyn_mask(self, prep: Path, target_hw: tuple[int, int]) -> Optional[np.ndarray]:
        """Aggregate all MonST3R dynamic mask keyframes → union, resized to target."""
        d = prep / "monst3r"
        files = sorted(d.glob("dynamic_mask_*.png"))    # iTACO real-data naming
        if not files:
            files = sorted(d.glob("dynamicmask_*.png"))  # iTACO sim_data naming
        if not files:
            return None
        union = None
        for fp in files:
            m = np.array(Image.open(fp).convert("L")) > 127
            if union is None:
                union = m
            elif m.shape == union.shape:
                union = union | m
        if union is None:
            return None
        # Resize to target_hw using PIL nearest
        H_t, W_t = target_hw
        u_pil = Image.fromarray(union.astype(np.uint8) * 255).resize((W_t, H_t), Image.NEAREST)
        return np.array(u_pil) > 127

    def _load_motion_cache(
        self, prep: Path, fids: list[int],
        H: int, W: int, H_p: int, W_p: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Returns (tracks_2d [S, N_t, 2], tracks_3d [S, N_t, 3],
                 tracks_vis [S, N_t], track_part_label [N_t, P],
                 motion_mask [P, H_p, W_p], has_motion_data).
        Zero placeholders when motion_cache_real.npz is missing.
        """
        S, N_t = self.num_frames, self.max_tracks
        P      = self.max_parts
        z2 = torch.zeros(S, N_t, 2)
        z3 = torch.zeros(S, N_t, 3)
        zv = torch.zeros(S, N_t)
        zl = torch.zeros(N_t, P)
        zm = torch.zeros(P, H_p, W_p)

        cache_path = prep / self.motion_cache_name
        if not cache_path.exists():
            return z2, z3, zv, zl, zm, False
        try:
            cache = np.load(str(cache_path), allow_pickle=True)
            S_full = cache["tracks_2d_norm"].shape[0]
            sel = [fid for fid in fids if 0 <= fid < S_full]
            if len(sel) != S:
                return z2, z3, zv, zl, zm, False
            raw_t2d = cache["tracks_2d_norm"][sel]              # [S, N_raw, 2] in [0,1]
            raw_t3d = cache["tracks_3d"][sel] if "tracks_3d" in cache.files else \
                      np.zeros((*raw_t2d.shape[:-1], 3), np.float32)
            raw_vis = cache["tracks_vis"][sel]
            raw_lbl = cache["track_part_label"]                  # [N_raw, P_cache]
            raw_msk = cache["motion_mask"]                       # [P_cache, h, w]

            N_raw = raw_t2d.shape[1]
            if N_raw >= N_t:
                perm = np.random.permutation(N_raw)[:N_t]
                raw_t2d, raw_t3d, raw_vis, raw_lbl = (
                    raw_t2d[:, perm], raw_t3d[:, perm],
                    raw_vis[:, perm], raw_lbl[perm],
                )
            else:
                pad_n = N_t - N_raw
                raw_t2d = np.concatenate([raw_t2d, np.zeros((S, pad_n, 2), raw_t2d.dtype)], 1)
                raw_t3d = np.concatenate([raw_t3d, np.zeros((S, pad_n, 3), raw_t3d.dtype)], 1)
                raw_vis = np.concatenate([raw_vis, np.zeros((S, pad_n),    raw_vis.dtype)], 1)
                raw_lbl = np.concatenate([raw_lbl, np.zeros((pad_n, raw_lbl.shape[1]), raw_lbl.dtype)], 0)

            t2d = torch.from_numpy(raw_t2d).float()
            t2d[..., 0] *= W; t2d[..., 1] *= H
            t3d = torch.from_numpy(raw_t3d).float()
            tvis = torch.from_numpy(raw_vis).float()

            P_cache = raw_lbl.shape[1]
            if P_cache < P:
                raw_lbl = np.concatenate([raw_lbl, np.zeros((N_t, P - P_cache), raw_lbl.dtype)], 1)
            elif P_cache > P:
                raw_lbl = raw_lbl[:, :P]
            tpl = torch.from_numpy(raw_lbl).float()

            P_c, h_c, w_c = raw_msk.shape
            mm = torch.from_numpy(raw_msk).float().unsqueeze(0)
            if (h_c, w_c) != (H_p, W_p):
                mm = F.interpolate(mm, size=(H_p, W_p), mode="bilinear", align_corners=False)
            mm = mm.squeeze(0)
            if P_c < P:
                mm_full = torch.zeros(P, H_p, W_p)
                mm_full[:P_c] = mm
                mm = mm_full
            else:
                mm = mm[:P]

            return t2d, t3d, tvis, tpl, mm, True
        except Exception as e:
            print(f"[iTACO] motion_cache load failed: {e}")
            return z2, z3, zv, zl, zm, False

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        raw, prep = self.entries[idx]
        meta = json.load(open(raw / "metadata.json"))
        n_frames_raw = len(meta["poses"])
        rgb_files = sorted((raw / "rgb").glob("*.jpg"))
        n_frames = min(n_frames_raw, len(rgb_files))

        S = self.num_frames
        H = W = self.target_size
        H_p, W_p = H // self.patch_size, W // self.patch_size

        fids = self._select_frames(n_frames)

        # ── RGB ────────────────────────────────────────────────────────
        images = self._load_rgb(raw, fids, H, W)                       # [S, 3, H, W]

        # ── Extrinsics (cam-to-world) ──────────────────────────────────
        poses = np.array(meta["poses"], dtype=np.float32)               # [T, 7]
        c2w_all = _pose_xyzquat_to_c2w(poses)                           # [T, 4, 4]
        extrinsics = torch.from_numpy(c2w_all[fids])                    # [S, 4, 4]

        # ── Intrinsics: use mean of perFrameIntrinsicCoeffs (fx,fy,cx,cy) ──
        H_orig, W_orig = meta["h"], meta["w"]
        if "perFrameIntrinsicCoeffs" in meta and meta["perFrameIntrinsicCoeffs"]:
            coeffs = np.array(meta["perFrameIntrinsicCoeffs"], dtype=np.float32)  # [T, 4]
            fx, fy, cx, cy = coeffs.mean(axis=0)
        else:
            K = np.array(meta["K"], dtype=np.float32).reshape(3, 3).T   # column-major in metadata
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        sx, sy = W / W_orig, H / H_orig
        K_full = torch.tensor([
            [fx * sx, 0,        cx * sx],
            [0,        fy * sy, cy * sy],
            [0,        0,        1.0   ],
        ], dtype=torch.float32)

        # ── Timestamps (normalised to [0, 1]) ──────────────────────────
        ts_all = np.array(meta.get("frameTimestamps", list(range(n_frames_raw))), dtype=np.float32)
        ts = ts_all[fids]
        ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6) if S > 1 else np.zeros(1, np.float32)
        timestamps = torch.from_numpy(ts).float()

        # ── Depth ──────────────────────────────────────────────────────
        depth, has_depth = self._load_depth(prep, fids, H, W)

        # ── MonST3R dynamic union (for instance scoring + motion_mask fallback) ──
        dyn = self._load_dyn_mask(prep, (H_orig, W_orig))

        # ── Per-frame part masks (SAM2 over-seg → reduced to max_parts) ──
        part_masks = self._load_masks(prep, fids, H, W, dyn_mask=dyn)   # [S, P, H, W]
        # Phase-2 pseudo-masks share the same content
        pseudo_masks = part_masks.clone()
        has_pseudo = bool(part_masks.sum() > 0)

        # ── Tracks (precomputed cache; zeros if absent) ────────────────
        tracks_2d, tracks_3d, tracks_vis, track_part_label, motion_mask, has_motion = (
            self._load_motion_cache(prep, fids, H, W, H_p, W_p)
        )
        # Use MonST3R dyn as a coarse motion_mask fallback (in slot 1) if missing
        if not has_motion and dyn is not None:
            dyn_pp = torch.from_numpy(dyn).float().unsqueeze(0).unsqueeze(0)
            dyn_pp = F.interpolate(dyn_pp, size=(H_p, W_p), mode="bilinear", align_corners=False)
            motion_mask[1] = dyn_pp.squeeze().clamp(0, 1)
            has_motion = False  # tracks themselves still missing → keep flag false

        return {
            "images":           images,
            "extrinsics":       extrinsics,
            "intrinsics":       K_full,
            "timestamps":       timestamps,
            "part_masks":       part_masks,
            "pseudo_masks":     pseudo_masks,
            "has_pseudo_masks": has_pseudo,
            "depth":            depth,
            "has_depth":        has_depth,
            "tracks_2d":        tracks_2d,
            "tracks_3d":        tracks_3d,
            "tracks_vis":       tracks_vis,
            "motion_mask":      motion_mask,
            "track_part_label": track_part_label,
            "has_motion_data":  has_motion,
            # Real data has no GT kinematics — leave zeros + flag for compute_loss to skip
            "gt_motion_type":   torch.zeros(self.max_parts, dtype=torch.long),
            "gt_axis":          torch.zeros(self.max_parts, 3),
            "gt_pivot":         torch.zeros(self.max_parts, 3),
            "gt_scalars":       torch.zeros(self.max_parts, S),
            "has_kin_gt":       False,
            "has_pose":         True,
            "scene_id":         raw.name,
            "n_active_parts":   self.max_parts,   # unknown → upper bound
        }
