"""
iTACO sim_data dataset wrapper.

Source layout (verified by scripts/probe_itaco_sim.py):
  /data2/cyt/video2articulation/sim_data
  ├── partnet_mobility/{Cat}/{instance_id}/joint_{j}_bg/
  │   ├── meta.json                 {"joint_id": int, "init": float, "target": float, ...}
  │   ├── joint_id_list.txt         lines: "joint_0", "joint_1", ...
  │   ├── gt_joint_value.npy        [T] float64 — active joint scalar over time
  │   ├── qpos.npy                  [T, n_joints] float32
  │   ├── actor_pose.pkl            {"actor_X": list of [7] xyz+wxyz, len T}
  │   └── view_{0,1,init}/
  │       ├── rgb/{NNNNNN}.jpg      480×640
  │       ├── depth/{NNNNNN}.npz    uint16 mm, key='a' shape (480, 640)
  │       ├── segment/{NNNNNN}.npz  uint8 link-id, key='a' shape (480, 640)
  │       ├── camera_pose.npy       [T, 7] xyz + wxyz, c2w direct
  │       └── intrinsics.npy        [3, 3]
  └── exp_results/preprocessing/{Cat}/{instance_id}/joint_{j}_bg/view_{0,1}/
      ├── monst3r/dynamic_mask_{0..17}.png   sparse keyframes
      └── video_segment_reverse/             SAM2 propagation outputs

Global GT kinematics:
  /data2/cyt/video2articulation/new_partnet_mobility_dataset_correct_intr_meta.json
  {Cat: {instance_id: {boundingbox, interaction_list:
                       [{id, type:hinge|slider, joint:{axis:{origin,direction}, limit}}]}}}
  axis/pivot are in **object local frame**.

Key design (Plan A active-part detection):
  • Per-scene, rank actor_pose entries by (translation_std + rotation_std).
  • Active actor = top-ranked.  segment_id = int(actor_name suffix).
  • Other actors' segment ids fill auxiliary slots 2..k (no kin GT).
  • Object-to-world frame: use the lowest-id actor's frame-0 pose.
    (URDF root link, has near-zero pose variance — verified in probe.)

Output dict matches ArticulatedDataset / iTACORealDataset schema; new field:
  • dataset_tag = "itaco_sim"
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
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


def _pose7_to_M(pose7: np.ndarray) -> np.ndarray:
    """[..., 7] (xyz + wxyz) → [..., 4, 4]"""
    M = np.zeros(pose7.shape[:-1] + (4, 4), dtype=np.float32)
    M[..., :3, :3] = _quat_wxyz_to_R(pose7[..., 3:7].astype(np.float32))
    M[..., :3,  3] = pose7[..., :3].astype(np.float32)
    M[...,  3,  3] = 1.0
    return M


def _normalize_scalars(values: np.ndarray) -> np.ndarray:
    """Match articulated_dataset._normalize_scalars: rest-shift, scale by max-abs → [-1, 1]."""
    rest = values[0]
    shifted = values - rest
    max_abs = float(np.abs(shifted).max())
    if max_abs < 1e-8:
        return np.zeros_like(values)
    return shifted / max_abs


# ─────────────────────────────────────────────────────────────────────────────
# Global GT-kin meta (cached at module level — single ~1MB JSON read).
_GLOBAL_META_PATH = Path("/data2/cyt/video2articulation/new_partnet_mobility_dataset_correct_intr_meta.json")


@lru_cache(maxsize=1)
def _load_global_meta() -> dict:
    if not _GLOBAL_META_PATH.exists():
        raise FileNotFoundError(f"iTACO global meta not found: {_GLOBAL_META_PATH}")
    with open(_GLOBAL_META_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
class iTACOSimDataset(Dataset):
    """Returns batches in same shape/key set as ArticulatedDataset."""

    TYPE_MAP = {"static": 0, "slider": 1, "hinge": 2}

    def __init__(
        self,
        data_root: str = "/data2/cyt/video2articulation/sim_data",
        target_size: int = 518,
        num_frames: int = 8,
        max_parts: int = 8,
        max_tracks: int = 1024,
        patch_size: int = 14,
        categories: Optional[list[str]] = None,
        views: tuple[str, ...] = ("view_0", "view_1"),
        split: str = "all",
        val_ratio: float = 0.05,
        split_seed: int = 42,
        random_start: bool = True,
    ):
        super().__init__()
        self.root          = Path(data_root)
        self.target_size   = target_size
        self.num_frames    = num_frames
        self.max_parts     = max_parts
        self.max_tracks    = max_tracks
        self.patch_size    = patch_size
        self.random_start  = random_start

        pm_root = self.root / "partnet_mobility"
        prep_root = self.root / "exp_results" / "preprocessing"
        if not pm_root.exists():
            raise FileNotFoundError(f"missing {pm_root}")

        meta = _load_global_meta()
        cats = sorted(p.name for p in pm_root.iterdir() if p.is_dir())
        if categories is not None:
            cats = [c for c in cats if c in categories]

        # Build (Cat, instance_id, joint_dir, view) entries — only those with
        # complete files AND a matching GT entry in global meta.
        self.entries: list[dict] = []
        n_skipped = 0
        for cat in cats:
            cat_dir = pm_root / cat
            cat_meta = meta.get(cat, {})
            for inst_dir in sorted(cat_dir.iterdir()):
                if not inst_dir.is_dir(): continue
                inst_id = inst_dir.name
                inst_meta = cat_meta.get(inst_id)
                if inst_meta is None:
                    n_skipped += 1; continue
                for jdir in sorted(inst_dir.iterdir()):
                    if not jdir.is_dir() or not jdir.name.startswith("joint_"): continue
                    scene_meta_path = jdir / "meta.json"
                    if not scene_meta_path.exists():
                        n_skipped += 1; continue
                    scene_meta = json.load(open(scene_meta_path))
                    j_id = scene_meta["joint_id"]
                    # find matching interaction
                    inter = next(
                        (it for it in inst_meta.get("interaction_list", [])
                         if it.get("id") == j_id), None
                    )
                    if inter is None:
                        n_skipped += 1; continue
                    for v in views:
                        v_dir = jdir / v
                        if not v_dir.exists(): continue
                        if not (v_dir / "rgb").exists(): continue
                        if not (v_dir / "camera_pose.npy").exists(): continue
                        prep_v = prep_root / cat / inst_id / jdir.name / v
                        self.entries.append({
                            "cat":       cat,
                            "inst":      inst_id,
                            "jdir":      jdir.name,
                            "view":      v,
                            "raw_v":     v_dir,
                            "prep_v":    prep_v,
                            "joint_id":  j_id,
                            "type_str":  inter["type"],     # "hinge" | "slider"
                            "axis_dir":  inter["joint"]["axis"]["direction"],
                            "axis_org":  inter["joint"]["axis"]["origin"],
                        })

        # Deterministic train/val split on the entry list
        rng = np.random.default_rng(split_seed)
        order = np.arange(len(self.entries)); rng.shuffle(order)
        n_val = max(1, int(len(self.entries) * val_ratio))
        val_idx = set(order[:n_val].tolist())
        if split == "train":
            self.entries = [e for i, e in enumerate(self.entries) if i not in val_idx]
        elif split == "val":
            self.entries = [e for i, e in enumerate(self.entries) if i in val_idx]
        # else "all"

        assert self.entries, f"No iTACO sim scenes found under {data_root}"
        print(f"[iTACOSim] {len(self.entries)} scenes (split={split}, skipped={n_skipped})  "
              f"categories={sorted({e['cat'] for e in self.entries})}")

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.entries)

    # ------------------------------------------------------------------ #
    def _select_frames(self, n_total: int) -> list[int]:
        S = self.num_frames
        if n_total <= S:
            return list(range(n_total)) + [n_total - 1] * (S - n_total)
        max_i = n_total - 1
        stride = max_i / (S - 1) if S > 1 else 0
        offset = float(np.random.uniform(0, stride * 0.5)) if (self.random_start and S > 1) else 0.0
        ids = [int(round(offset + stride * i)) for i in range(S)]
        return [min(max(i, 0), max_i) for i in ids]

    # ------------------------------------------------------------------ #
    def _detect_active_part(
        self, ap: dict, seg_uniques: set[int]
    ) -> tuple[Optional[int], Optional[str], list[int]]:
        """
        Returns (active_segment_id, active_actor_name, other_actor_segment_ids).

        Plan A: rank actors by (trans_std + rot_std); the top one's name suffix
        is the active segment. Other movable actors fill auxiliary slots.
        Actors with zero variance (URDF root / static appendages) are excluded.
        """
        scores = {}
        for k, lst in ap.items():
            arr = np.stack(lst)  # [T, 7]
            scores[k] = float(arr[:, :3].std(0).sum() + arr[:, 3:7].std(0).sum())
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])

        def _suffix_id(name: str) -> Optional[int]:
            try:
                v = int(name.split("_")[-1])
                return v if v in seg_uniques else None
            except Exception:
                return None

        active_seg = None
        active_name = None
        others: list[int] = []
        for name, sc in ranked:
            sid = _suffix_id(name)
            if sid is None: continue
            if sc < 1e-6: continue
            if active_seg is None:
                active_seg = sid
                active_name = name
            else:
                others.append(sid)
        return active_seg, active_name, others

    # ------------------------------------------------------------------ #
    def _fit_kin_from_actor_pose(
        self, active_actor_poses: list[np.ndarray], type_str: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit world-frame (axis, pivot) from active actor's full SE(3) trajectory.

        For revolute (hinge):
          R_rel = R_t @ R_0^T   (relative rotation)
          axis  = log(R_rel) / θ      (Rodrigues inverse)
          pivot = (I - R_rel)^{+} @ (T_t - R_rel @ T_0)

        For prismatic (slider):
          axis  = (T_last - T_0) / ||·||
          pivot = T_0  (point on motion line)

        Inputs:
          active_actor_poses: list of [7] pose vectors (xyz + wxyz), len T
          type_str: "hinge" | "slider"

        Returns: (axis [3] unit, pivot [3]) in **world frame**.
        """
        arr = np.stack(active_actor_poses).astype(np.float64)        # [T, 7]
        T = arr.shape[0]
        T0 = arr[0, :3]
        R0 = _quat_wxyz_to_R(arr[0, 3:7])

        if type_str == "slider":
            # find frame with biggest translation displacement
            dT = arr[:, :3] - T0
            t_far = int(np.argmax(np.linalg.norm(dT, axis=1)))
            disp = arr[t_far, :3] - T0
            n = float(np.linalg.norm(disp))
            if n < 1e-6:
                return np.array([1, 0, 0], np.float32), T0.astype(np.float32)
            axis = disp / n
            return axis.astype(np.float32), T0.astype(np.float32)

        # ── revolute: pick frame with biggest rotation angle ──────────
        best_t, best_theta = 1, 0.0
        for t in range(1, T):
            R_t = _quat_wxyz_to_R(arr[t, 3:7])
            R_rel = R_t @ R0.T
            cos_th = (np.trace(R_rel) - 1) / 2
            cos_th = float(np.clip(cos_th, -1, 1))
            theta = float(np.arccos(cos_th))
            if theta > best_theta:
                best_theta = theta; best_t = t
        if best_theta < 1e-3:
            return np.array([0, 0, 1], np.float32), T0.astype(np.float32)

        R_t = _quat_wxyz_to_R(arr[best_t, 3:7])
        T_t = arr[best_t, :3]
        R_rel = R_t @ R0.T
        # axis from skew-symmetric part: (R - R^T) / (2 sin θ)
        sin_th = np.sin(best_theta)
        skew = (R_rel - R_rel.T) / max(2 * sin_th, 1e-8)
        axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-8 else np.array([0, 0, 1], np.float64)

        # pivot: solve (I - R_rel) @ c = (T_t - R_rel @ T_0)  via pseudoinverse
        rhs = T_t - R_rel @ T0
        pivot = np.linalg.pinv(np.eye(3) - R_rel) @ rhs
        return axis.astype(np.float32), pivot.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _load_rgb(self, raw_v: Path, fids: list[int], H: int, W: int) -> torch.Tensor:
        imgs = []
        for fid in fids:
            p = raw_v / "rgb" / f"{fid:06d}.jpg"
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            imgs.append(arr)
        x = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2)
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    def _load_depth(self, raw_v: Path, fids: list[int], H: int, W: int) -> tuple[torch.Tensor, bool]:
        ds = []
        ok = True
        for fid in fids:
            p = raw_v / "depth" / f"{fid:06d}.npz"
            if not p.exists():
                ok = False; ds.append(np.zeros((H, W), np.float32)); continue
            arr = np.load(p)["a"].astype(np.float32) / 1000.0   # uint16 mm → meters
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(H, W), mode="nearest").squeeze(0).squeeze(0)
            ds.append(t.numpy())
        return torch.from_numpy(np.stack(ds, axis=0)), ok

    def _load_segments(self, raw_v: Path, fids: list[int], H: int, W: int) -> tuple[torch.Tensor, set[int]]:
        """Returns ([S, H, W] uint8 segment ids, union of unique values)."""
        segs = []
        uniques: set[int] = set()
        for fid in fids:
            p = raw_v / "segment" / f"{fid:06d}.npz"
            if not p.exists():
                segs.append(np.zeros((H, W), np.uint8)); continue
            arr = np.load(p)["a"]                         # uint8 (H_o, W_o)
            uniques.update(np.unique(arr).tolist())
            t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(H, W), mode="nearest").squeeze(0).squeeze(0)
            segs.append(t.numpy().astype(np.uint8))
        return torch.from_numpy(np.stack(segs)), uniques

    def _build_part_masks(
        self, segs: torch.Tensor, active_seg: Optional[int], other_segs: list[int]
    ) -> tuple[torch.Tensor, int]:
        """
        segs: [S, H, W] uint8
        Output: [S, max_parts, H, W] float32, n_parts (incl. bg)
          slot 0 = bg = (segment ∉ {active, others})
          slot 1 = (segment == active_seg)
          slots 2..k = (segment == others[i])
        """
        S, H, W = segs.shape
        out = torch.zeros(S, self.max_parts, H, W, dtype=torch.float32)
        movable = []
        if active_seg is not None:
            movable.append(active_seg)
        movable += other_segs[: self.max_parts - 2]

        # foreground = pixels that are any movable actor
        fg = torch.zeros_like(segs, dtype=torch.bool)
        for slot_idx, sid in enumerate(movable):
            m = (segs == sid)
            out[:, slot_idx + 1] = m.float()
            fg |= m
        # bg = NOT any movable
        out[:, 0] = (~fg).float()
        n_parts = 1 + len(movable)   # incl. bg
        return out, n_parts

    def _gt_kin(
        self, entry: dict, fids: list[int],
        active_actor_poses: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (gt_motion_type [P], gt_axis [P, 3], gt_pivot [P, 3], gt_scalars [P, S]).
        Active-part fields go in slot 1; rest stay zero.
        Axis/pivot derived directly from active actor's world-pose trajectory.
        """
        P = self.max_parts
        S = len(fids)
        mtype = torch.zeros(P, dtype=torch.long)
        gaxis = torch.zeros(P, 3, dtype=torch.float32)
        gpiv  = torch.zeros(P, 3, dtype=torch.float32)
        gscl  = torch.zeros(P, S, dtype=torch.float32)

        mtype[1] = self.TYPE_MAP.get(entry["type_str"], 0)

        ax_w, pv_w = self._fit_kin_from_actor_pose(active_actor_poses, entry["type_str"])
        gaxis[1] = torch.from_numpy(ax_w)
        gpiv[1]  = torch.from_numpy(pv_w)

        # gt_scalars: rest-zero, [-1, 1]
        gj_path = (entry["raw_v"].parent / "gt_joint_value.npy")
        if gj_path.exists():
            gj = np.load(gj_path).astype(np.float32)
            sel = gj[np.clip(np.asarray(fids), 0, len(gj)-1)]
            gscl[1, :] = torch.from_numpy(_normalize_scalars(sel))
        return mtype, gaxis, gpiv, gscl

    def _load_dyn_mask(self, prep_v: Path, target_hw: tuple[int, int]) -> Optional[np.ndarray]:
        """Union of all MonST3R dynamic_mask_*.png keyframes, resized to target_hw."""
        d = prep_v / "monst3r"
        if not d.exists(): return None
        files = sorted(d.glob("dynamic_mask_*.png"))
        if not files: return None
        union = None
        for fp in files:
            m = np.array(Image.open(fp).convert("L")) > 127
            if union is None:
                union = m
            elif m.shape == union.shape:
                union |= m
        if union is None: return None
        H_t, W_t = target_hw
        u_pil = Image.fromarray(union.astype(np.uint8) * 255).resize((W_t, H_t), Image.NEAREST)
        return np.array(u_pil) > 127

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        raw_v: Path = entry["raw_v"]
        prep_v: Path = entry["prep_v"]
        raw_j = raw_v.parent

        S = self.num_frames
        H = W = self.target_size
        H_p, W_p = H // self.patch_size, W // self.patch_size

        # ── Frame count (use #rgb) ─────────────────────────────────────
        rgb_files = sorted((raw_v / "rgb").glob("*.jpg"))
        n_total = len(rgb_files)
        fids = self._select_frames(n_total)

        # ── RGB ────────────────────────────────────────────────────────
        images = self._load_rgb(raw_v, fids, H, W)               # [S, 3, H, W]

        # ── Camera pose (T, 7) wxyz, c2w direct ────────────────────────
        # iTACO renders with OpenGL camera convention (cam looks at -Z, +Y up).
        # The rest of the pipeline (and ArticulatedDataset) uses OpenCV convention
        # (cam looks at +Z, +Y down). Flip Y/Z axes of the camera frame.
        cp7 = np.load(raw_v / "camera_pose.npy").astype(np.float32)
        c2w_all = _pose7_to_M(cp7)                                # [T, 4, 4] OpenGL
        _gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        c2w_all = c2w_all @ _gl_to_cv                             # → OpenCV
        # Clip frame ids to valid range of camera_pose (some scenes may have mismatched lengths)
        n_cam = c2w_all.shape[0]
        fids_c = [min(f, n_cam - 1) for f in fids]
        extrinsics = torch.from_numpy(c2w_all[fids_c].astype(np.float32))   # [S, 4, 4]

        # ── Intrinsics ─────────────────────────────────────────────────
        K_orig = np.load(raw_v / "intrinsics.npy").astype(np.float32)
        # iTACO sim renders at 480×640
        H_orig, W_orig = 480, 640
        sx, sy = W / W_orig, H / H_orig
        K_full = torch.tensor([
            [K_orig[0, 0] * sx, 0,                   K_orig[0, 2] * sx],
            [0,                  K_orig[1, 1] * sy, K_orig[1, 2] * sy],
            [0,                  0,                   1.0],
        ], dtype=torch.float32)

        # ── Timestamps (uniform [0, 1]) ────────────────────────────────
        timestamps = torch.linspace(0, 1, S) if S > 1 else torch.zeros(1)

        # ── Depth ──────────────────────────────────────────────────────
        depth, has_depth = self._load_depth(raw_v, fids, H, W)

        # ── Segment + active-part detection ────────────────────────────
        with open(raw_j / "actor_pose.pkl", "rb") as f:
            ap = pickle.load(f)
        segs, seg_uniques = self._load_segments(raw_v, fids, H, W)         # [S, H, W]
        active_seg, active_name, other_segs = self._detect_active_part(ap, seg_uniques)
        part_masks, n_parts = self._build_part_masks(segs, active_seg, other_segs)

        # ── GT kinematics from active actor's world-pose trajectory ────
        active_poses = ap[active_name] if active_name is not None else ap[next(iter(ap))]
        gt_motion_type, gt_axis, gt_pivot, gt_scalars = self._gt_kin(
            entry, fids, active_poses
        )

        # ── MonST3R dynamic union → coarse motion_mask placeholder ────
        dyn = self._load_dyn_mask(prep_v, (H, W))
        motion_mask = torch.zeros(self.max_parts, H_p, W_p)
        if dyn is not None:
            t = torch.from_numpy(dyn).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(H_p, W_p), mode="bilinear", align_corners=False)
            motion_mask[1] = t.squeeze().clamp(0, 1)

        # ── Tracks: empty placeholder (pre-compute later if needed) ────
        tracks_2d  = torch.zeros(S, self.max_tracks, 2)
        tracks_3d  = torch.zeros(S, self.max_tracks, 3)
        tracks_vis = torch.zeros(S, self.max_tracks)
        track_part_label = torch.zeros(self.max_tracks, self.max_parts)

        return {
            "images":           images,
            "extrinsics":       extrinsics,
            "intrinsics":       K_full,
            "timestamps":       timestamps,
            "part_masks":       part_masks,
            "pseudo_masks":     part_masks.clone(),
            "has_pseudo_masks": True,
            "depth":            depth,
            "has_depth":        has_depth,
            "tracks_2d":        tracks_2d,
            "tracks_3d":        tracks_3d,
            "tracks_vis":       tracks_vis,
            "motion_mask":      motion_mask,
            "track_part_label": track_part_label,
            "has_motion_data":  False,    # tracks not precomputed yet
            "gt_motion_type":   gt_motion_type,
            "gt_axis":          gt_axis,
            "gt_pivot":         gt_pivot,
            "gt_scalars":       gt_scalars,
            "has_kin_gt":       True,
            "has_pose":         True,
            "scene_id":         f"{entry['cat']}/{entry['inst']}/{entry['jdir']}/{entry['view']}",
            "n_active_parts":   n_parts,
            "dataset_tag":      "itaco_sim",
        }