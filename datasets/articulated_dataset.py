"""
ArticulatedDataset — PartNet-Mobility 格式的关节物体数据集加载器

支持两种数据布局：

  (A) 单相机格式（简单格式）:
    data_root/
      {object_id}/
        images/            000.jpg 001.jpg ...
        intrinsics.txt     3×3 内参
        extrinsics/        000.txt 001.txt ...
        joint_params.json  {"joint_0": {...}, ...}
        joint_angles.json  {"000": {"joint_0": val, ...}, ...}
        part_masks/        {frame_id}_{part_id}.png

  (B) 多相机格式:
    data_root/
      {object_id}/
        cam_00/
          images/  extrinsics/  intrinsics.txt  part_masks/
        cam_01/ ...
        cam_02/ ...
        cam_03/ ...
        joint_params.json   (scene 级别)
        joint_angles.json   (scene 级别)

  格式 (B) 中，每个 (object_id, cam_XX) 对被视为独立的数据集条目，
  extrinsics/intrinsics/images/masks 从对应的 cam 子目录读取，
  joint_params/joint_angles 从 scene 根目录读取。

Mask 命名规则：
  joint k (0-indexed) → mask part_id = k + 2
  最大 part_id (= n_joints + 2) = 静态 root/base → 并入 Slot 0

Slot 分配：
  Slot 0 = 静态背景 + 静态 root 部件
  Slot k (k=1..n_joints) = joint_{k-1} 的运动部件

标量归一化：物理关节角度/位移逐关节逐场景归一化到 [-1, 1]
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str, target_w: int, target_h: int) -> torch.Tensor:
    """Load image, resize to (target_w, target_h), return [3, H, W] ∈ [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return TF.ToTensor()(img)   # [3, H, W]


def _load_mask(path: str, target_w: int, target_h: int) -> torch.Tensor:
    """Load binary mask PNG (0/255), resize, return [H, W] float32 ∈ {0,1}."""
    if not os.path.exists(path):
        return torch.zeros(target_h, target_w)
    mask = Image.open(path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    t = TF.ToTensor()(mask)[0]   # [H, W], values in [0,1]
    return (t > 0.5).float()


def _load_depth(path: str, target_w: int, target_h: int) -> Optional[torch.Tensor]:
    """Load depth as .npy (float32 meters). Resize bilinearly to target. None if missing."""
    if not os.path.exists(path):
        return None
    d = np.load(path).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)        # [1,1,h,w]
    t = F.interpolate(t, size=(target_h, target_w),
                      mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0)                            # [H, W]


def _pad_or_crop_tracks(arr: np.ndarray, target_n: int, fill: float = 0.0) -> np.ndarray:
    """Pad or crop along the second axis (track axis) to target_n.

    arr: [S, N, ...] or [N, ...]
    Returns array with track axis = target_n.
    """
    track_axis = 1 if arr.ndim >= 3 else 0
    cur = arr.shape[track_axis]
    if cur == target_n:
        return arr
    if cur > target_n:
        sl = [slice(None)] * arr.ndim
        sl[track_axis] = slice(0, target_n)
        return arr[tuple(sl)]
    # pad
    pad_shape = list(arr.shape)
    pad_shape[track_axis] = target_n - cur
    pad = np.full(pad_shape, fill, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=track_axis)


def _adjust_intrinsics(K: np.ndarray,
                       orig_w: int, orig_h: int,
                       new_w: int, new_h: int) -> np.ndarray:
    """Scale intrinsics for image resize."""
    K = K.copy().astype(np.float32)
    K[0, 0] *= new_w / orig_w   # fx
    K[1, 1] *= new_h / orig_h   # fy
    K[0, 2] *= new_w / orig_w   # cx
    K[1, 2] *= new_h / orig_h   # cy
    return K


def _normalize_scalars(values: list[float], motion_type: str,
                       scene_radius: float = 1.0) -> list[float]:
    """
    Map physical joint values → normalised motion scalar S ∈ [-1, 1], centred on
    rest, using the SAME convention as the SDF renderer / paper (Eq.S4-S5):

        revolute : S = (angle_rad   − rest) / (2π)      ← render: angle = 2π·S
        prismatic: S = (translation − rest) / (2·r)     ← render: trans = 2r·S
        static   : 0

    This keeps the kinematic-supervision target and the differentiable ray
    transform geometrically consistent (an absolute physical scale), unlike the
    old per-joint max-deviation normalisation which conflicted with the renderer.

    values[0] is the rest-state value (frame 0 = canonical rest pose) → S=0.
    """
    import math
    rest = values[0]
    if motion_type == "revolute":
        denom = 2.0 * math.pi
    elif motion_type == "prismatic":
        denom = 2.0 * scene_radius
    else:
        return [0.0] * len(values)
    return [(v - rest) / denom for v in values]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArticulatedDataset(Dataset):
    """
    Args:
        data_root:    root dir with {object_id} subdirectories
        target_size:  resize images & masks to target_size × target_size
        num_frames:   subsample to this many frames (None = all)
        max_parts:    max slots P (default 8; must be > max joints in dataset)
        phase:        "1" GT masks available; "2" pseudo-masks (SAM2)
        split:        "train", "val", or "all" (default "all")
        val_ratio:    fraction of scenes held out for val (default 0.15)
        split_seed:   random seed for deterministic scene shuffle (default 42)
        exclude_cams: set of camera names to skip, e.g. {"cam_00"}.
                      cam_00 is the back-view in PartNet-Mobility multi-cam data
                      and typically shows only 1 part (70% of scenes), making it
                      a low-quality training signal. Default: {"cam_00"}.
    """

    def __init__(
        self,
        data_root: str,
        target_size: int = 518,
        num_frames: Optional[int] = None,
        max_parts: int = 8,
        phase: str = "1",
        split: str = "all",
        val_ratio: float = 0.15,
        split_seed: int = 42,
        exclude_cams: Optional[set] = None,
        max_tracks: int = 4096,
        patch_size: int = 14,
    ):
        super().__init__()
        self.data_root   = Path(data_root)
        self.target_size = target_size
        self.num_frames  = num_frames
        self.max_parts   = max_parts
        self.phase       = phase
        self.max_tracks  = max_tracks
        self.patch_size  = patch_size
        # cam_00 is the back-view in PartNet-Mobility; excluded by default.
        self._exclude_cams: set = exclude_cams if exclude_cams is not None else {"cam_00"}
        # Instance variables (NOT class variables) to avoid cross-instance clobbering
        self._orig_w: int = 640
        self._orig_h: int = 480

        def _has_images(cam_dir: Path) -> bool:
            img_dir = cam_dir / "images"
            if not img_dir.exists():
                return False
            return bool(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        def _cam_subdirs(scene: Path) -> list[Path]:
            """Return sorted cam_XX subdirs if present, else empty list."""
            cams = sorted([
                c for c in scene.iterdir()
                if c.is_dir() and c.name.startswith("cam_") and _has_images(c)
            ])
            return cams

        def _n_joints(scene: Path) -> int:
            """Return number of joints in scene; -1 if unreadable."""
            try:
                return len(json.load(open(scene / "joint_params.json")))
            except Exception:
                return -1

        def _is_valid_scene(d: Path) -> bool:
            return (
                d.is_dir()
                and (d / "joint_params.json").exists()
                and (_has_images(d) or len(_cam_subdirs(d)) > 0)
                and 0 < _n_joints(d) <= max_parts - 1
            )

        # ── Collect all valid scene-root directories ───────────────────────
        # A scene root is valid if it has joint_params.json AND either:
        #   • a direct images/ dir (format A), or
        #   • at least one cam_XX/ subdir with images (format B)
        # Scenes with more joints than max_parts-1 are silently skipped.
        #
        # Two directory layouts are supported transparently:
        #   • flat:      data_root/{object_id}/...
        #   • categorised: data_root/{category}/{object_id}/...   (PartNet-Mobility
        #     data_processed groups scenes under category folders)
        valid_roots = sorted([
            d for d in self.data_root.iterdir() if _is_valid_scene(d)
        ])
        if len(valid_roots) == 0:
            # Descend one level: treat immediate children as category dirs.
            for cat in sorted(self.data_root.iterdir()):
                if cat.is_dir():
                    valid_roots.extend(
                        sorted(s for s in cat.iterdir() if _is_valid_scene(s))
                    )
        assert len(valid_roots) > 0, f"No valid scenes in {data_root}"

        # ── Train/val split at scene-root level (prevents data leakage) ────
        rng = np.random.default_rng(split_seed)
        idx = np.arange(len(valid_roots))
        rng.shuffle(idx)
        n_val = max(1, int(len(valid_roots) * val_ratio))
        val_set   = set(idx[:n_val].tolist())
        train_set = set(idx[n_val:].tolist())

        if split == "all":
            selected_roots = valid_roots
        elif split == "val":
            selected_roots = [valid_roots[i] for i in sorted(val_set)]
        else:  # "train"
            selected_roots = [valid_roots[i] for i in sorted(train_set)]

        assert len(selected_roots) > 0, (
            f"Split '{split}' produced 0 scenes from {len(valid_roots)} total"
        )

        # ── Expand scene roots → (scene_root, cam_dir_or_None) entries ─────
        # Format A → (scene_root, None)
        # Format B → (scene_root, cam_dir) for each cam, skipping excluded cams
        self.entries: list[tuple[Path, Path | None]] = []
        for root in selected_roots:
            if _has_images(root):
                self.entries.append((root, None))
            else:
                for cam in _cam_subdirs(root):
                    if cam.name not in self._exclude_cams:
                        self.entries.append((root, cam))

        assert len(self.entries) > 0, (
            f"No entries after expanding split '{split}'"
        )

        # Autodetect original image resolution from first available image
        first_img = None
        for scene_root, cam_dir in self.entries:
            src = cam_dir if cam_dir is not None else scene_root
            candidates = (list((src / "images").glob("*.jpg")) +
                          list((src / "images").glob("*.png")))
            if candidates:
                first_img = candidates[0]
                break
        if first_img is not None:
            w, h = Image.open(str(first_img)).size
            self._orig_w = w   # instance variable, not class variable
            self._orig_h = h

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        scene_root, cam_dir = self.entries[idx]
        # cam_dir: None for format-A, a Path like .../cam_00 for format-B
        # Per-cam data (images/extrinsics/intrinsics/masks) lives in:
        data_dir = cam_dir if cam_dir is not None else scene_root
        H = W = self.target_size

        # ── Frame list ────────────────────────────────────────────────────
        img_dir   = data_dir / "images"
        img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        assert len(img_paths) > 0, f"No images in {img_dir}"
        frame_ids = [p.stem for p in img_paths]

        if self.num_frames is not None and len(frame_ids) > self.num_frames:
            sel = np.linspace(0, len(frame_ids) - 1,
                              self.num_frames, dtype=int).tolist()
            frame_ids = [frame_ids[i] for i in sel]
            img_paths = [img_paths[i] for i in sel]
        S = len(frame_ids)

        # ── Images [S, 3, H, W] ───────────────────────────────────────────
        images = torch.stack([
            _load_image(str(p), W, H) for p in img_paths
        ])   # [S, 3, H, W]

        # ── Intrinsics [3, 3] ─────────────────────────────────────────────
        K_orig = np.loadtxt(str(data_dir / "intrinsics.txt")).astype(np.float32)
        K_new  = _adjust_intrinsics(K_orig,
                                    self._orig_w, self._orig_h, W, H)
        intrinsics = torch.from_numpy(K_new)   # [3, 3]

        # ── Extrinsics [S, 4, 4] cam-to-world ────────────────────────────
        extr_dir = data_dir / "extrinsics"
        extrinsics = torch.stack([
            torch.from_numpy(
                np.loadtxt(str(extr_dir / f"{fid}.txt")).astype(np.float32)
            )
            for fid in frame_ids
        ])   # [S, 4, 4]

        # ── Joint params (always from scene root) ─────────────────────────
        with open(scene_root / "joint_params.json") as f:
            jp_raw = json.load(f)   # {"joint_0": {...}, "joint_1": {...}, ...}

        # Sort by joint name so index is consistent
        joint_keys   = sorted(jp_raw.keys())    # ["joint_0", "joint_1", ...]
        n_joints     = len(joint_keys)

        assert n_joints <= self.max_parts - 1, (
            f"{scene_root.name}: {n_joints} joints > max_parts-1={self.max_parts-1}. "
            f"Increase max_parts."
        )

        # ── GT scalars per joint per frame ────────────────────────────────
        with open(scene_root / "joint_angles.json") as f:
            angles_raw = json.load(f)  # {"000": {"joint_0": val, ...}, ...}

        type_map = {"static": 0, "prismatic": 1, "revolute": 2}
        gt_motion_type = torch.zeros(self.max_parts, dtype=torch.long)
        gt_axis        = torch.zeros(self.max_parts, 3)
        gt_pivot       = torch.zeros(self.max_parts, 3)
        gt_scalars     = torch.zeros(self.max_parts, S)

        for k, jkey in enumerate(joint_keys):
            p     = k + 1    # Slot index (Slot 0 = static)
            jdata = jp_raw[jkey]

            gt_motion_type[p] = type_map.get(jdata["type"], 0)
            ax = torch.tensor(jdata["axis"], dtype=torch.float32)
            gt_axis[p] = F.normalize(ax.unsqueeze(0), dim=-1).squeeze(0)
            gt_pivot[p] = torch.tensor(jdata.get("pivot", [0.0, 0.0, 0.0]),
                                        dtype=torch.float32)

            # Raw angle values for this joint across selected frames
            # Handle both nested {"000": {"joint_0": 0.5}} and flat {"000": 0.5}
            raw_angles = []
            for fid in frame_ids:
                frame_val = angles_raw.get(fid, 0.0)
                if isinstance(frame_val, dict):
                    raw_angles.append(float(frame_val.get(jkey, 0.0)))
                else:
                    # Flat format: single scalar per frame; assign to joint_0 only
                    raw_angles.append(float(frame_val) if k == 0 else 0.0)
            norm_angles = _normalize_scalars(raw_angles, jdata["type"])
            gt_scalars[p, :] = torch.tensor(norm_angles, dtype=torch.float32)

        # ── Part masks [S, max_parts, H, W] ──────────────────────────────
        # Two naming conventions are supported:
        #
        #  (1) part_index.json present (PartNet-Mobility data_processed):
        #      mask files are {fid}_{seg_id}.png where seg_id is SAPIEN's
        #      absolute per_scene_id. part_index.json (scene-level, built by
        #      scripts/build_partnet_part_index.py) maps joint_name → seg_id.
        #      Dynamic slot p=k+1 ← joint_keys[k]'s seg_id; every other mask
        #      id on disk (base / extra static links / stray light ids) → Slot 0.
        #
        #  (2) no part_index.json (legacy convention):
        #      joint k → mask part_id = k+2;  static root → n_joints+2 → Slot 0.
        masks_dir  = data_dir / "part_masks"
        part_masks = torch.zeros(S, self.max_parts, H, W)
        # Pure static-body silhouette (root/base links only, NO background) — used
        # by the SDF render loss so the big solid static part is rendered+supervised.
        # slot0 of part_masks stays bg∪static for the router/segmentation target.
        static_body_mask = torch.zeros(S, H, W)
        has_static_mask  = False

        part_index = None
        pidx_path = scene_root / "part_index.json"
        if pidx_path.exists():
            try:
                part_index = json.load(open(pidx_path))
            except Exception:
                part_index = None

        if part_index is not None and masks_dir.exists():
            # seg_id → dynamic slot index (1..n_joints) via joint_keys ordering
            joint_id = part_index.get("joint_id", {})
            id_to_slot = {}
            for k, jkey in enumerate(joint_keys):
                sid = joint_id.get(jkey)
                if sid is not None:
                    id_to_slot[int(sid)] = k + 1

            for s, fid in enumerate(frame_ids):
                slot_union = torch.zeros(H, W)
                for mpath in masks_dir.glob(f"{fid}_*.png"):
                    try:
                        seg_id = int(mpath.stem.split("_")[1])
                    except Exception:
                        continue
                    p = id_to_slot.get(seg_id)
                    if p is None:
                        continue   # base / extra static links / stray ids → Slot 0
                    mk = _load_mask(str(mpath), W, H)
                    part_masks[s, p] = mk
                    slot_union = (slot_union + mk).clamp(0, 1)

                # Pure static-body silhouette = union of static_ids masks (no bg).
                for sid in part_index.get("static_ids", []):
                    mp = masks_dir / f"{fid}_{int(sid)}.png"
                    if mp.exists():
                        sm = _load_mask(str(mp), W, H)
                        static_body_mask[s] = (static_body_mask[s] + sm).clamp(0, 1)
                if static_body_mask[s].sum() > 0:
                    has_static_mask = True

                # Slot 0 = base part (pure static body, NO background) — paper-aligned.
                # Background belongs to no slot; the router's bg-sink absorbs it.
                # (If static_ids yields no pixels, slot 0 is empty and all background
                #  flows to the sink — never reintroduce bg into slot 0.)
                part_masks[s, 0] = static_body_mask[s].clone()

        elif masks_dir.exists():
            for s, fid in enumerate(frame_ids):
                slot_union = torch.zeros(H, W)

                # Dynamic slots: joint k → part_id k+2
                for k in range(n_joints):
                    p       = k + 1
                    part_id = k + 2
                    mpath   = masks_dir / f"{fid}_{part_id}.png"
                    mk      = _load_mask(str(mpath), W, H)   # [H, W]
                    part_masks[s, p] = mk
                    slot_union = (slot_union + mk).clamp(0, 1)

                # Static root mask (part_id = n_joints + 2)
                root_id   = n_joints + 2
                mpath_root = masks_dir / f"{fid}_{root_id}.png"
                root_mask  = _load_mask(str(mpath_root), W, H)

                # Pure static-body silhouette = static root mask (no bg).
                static_body_mask[s] = root_mask.clamp(0, 1)
                if static_body_mask[s].sum() > 0:
                    has_static_mask = True

                # Slot 0 = base part (pure static root, NO background) — paper-aligned.
                # Background belongs to no slot; the router's bg-sink absorbs it.
                part_masks[s, 0] = root_mask.clamp(0, 1)

        # ── Pseudo-masks for Phase 2 (SAM2 output) ───────────────────────
        # Expected layout: {data_dir}/pseudo_masks/{fid}_{slot_id}.png
        # Same naming convention as GT part_masks.
        # "has_pseudo_masks" is only True when the directory exists and has files.
        pseudo_masks = torch.zeros(S, self.max_parts, H, W)
        has_pseudo = False
        if self.phase == "2":
            pseudo_dir = data_dir / "pseudo_masks"
            if pseudo_dir.exists():
                any_found = False
                for s, fid in enumerate(frame_ids):
                    for k in range(n_joints):
                        p       = k + 1
                        part_id = k + 2
                        mpath   = pseudo_dir / f"{fid}_{part_id}.png"
                        mk      = _load_mask(str(mpath), W, H)
                        pseudo_masks[s, p] = mk
                        if mk.sum() > 0:
                            any_found = True
                    # Slot 0 pseudo-mask: invert union of dynamic pseudo-masks
                    dyn_union = pseudo_masks[s, 1:n_joints+1].sum(0).clamp(0, 1)
                    pseudo_masks[s, 0] = (1.0 - dyn_union).clamp(0, 1)
                has_pseudo = any_found

        # ── Depth [S, H, W] ──────────────────────────────────────────────
        # Layout: {data_dir}/depth/{fid}.npy (float32, world meters).
        # Missing → zeros + has_depth=False (graceful for early-stage training).
        depth_dir = data_dir / "depth"
        depth_frames: list[torch.Tensor] = []
        any_depth = False
        for fid in frame_ids:
            d = _load_depth(str(depth_dir / f"{fid}.npy"), W, H)
            if d is None:
                depth_frames.append(torch.zeros(H, W))
            else:
                depth_frames.append(d)
                any_depth = True
        depth = torch.stack(depth_frames)                          # [S, H, W]

        # ── Precomputed motion data ──────────────────────────────────────
        # Layout: {data_dir}/motion_cache.npz produced by
        #   scripts/precompute_motion_data.py
        # Contents (all numpy):
        #   tracks_2d_norm  [S_full, N_raw, 2]   pixel coords / (W,H) ∈ [0,1]
        #   tracks_3d       [S_full, N_raw, 3]   world coords (meters)
        #   tracks_vis      [S_full, N_raw]      bool/float ∈ {0,1}
        #   motion_mask     [P, H_p, W_p]        first-frame patch pseudo-label
        #   track_part_label[N_raw, P]           per-track soft assignment
        #   frame_ids       [S_full]             string ids matching disk frames
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        N_t = self.max_tracks
        P   = self.max_parts

        tracks_2d        = torch.zeros(S, N_t, 2)
        tracks_3d        = torch.zeros(S, N_t, 3)
        tracks_vis       = torch.zeros(S, N_t)
        motion_mask      = torch.zeros(P, H_p, W_p)
        track_part_label = torch.zeros(N_t, P)
        has_motion_data  = False

        cache_path = data_dir / "motion_cache.npz"
        if cache_path.exists():
            try:
                cache = np.load(str(cache_path), allow_pickle=True)
                cache_fids = [str(x) for x in cache["frame_ids"].tolist()]
                fid2idx    = {f: i for i, f in enumerate(cache_fids)}

                # Map currently selected frame_ids → cache indices (graceful skip)
                sel_idx = [fid2idx[f] for f in frame_ids if f in fid2idx]
                if len(sel_idx) == S:
                    raw_t2d = cache["tracks_2d_norm"][sel_idx]   # [S, N_raw, 2]
                    raw_t3d = cache["tracks_3d"][sel_idx]        # [S, N_raw, 3]
                    raw_vis = cache["tracks_vis"][sel_idx]       # [S, N_raw]
                    raw_lbl = cache["track_part_label"]          # [N_raw, P_cache]
                    raw_msk = cache["motion_mask"]               # [P_cache, h, w]

                    # Pad/crop track axis to N_t
                    raw_t2d = _pad_or_crop_tracks(raw_t2d, N_t, fill=0.0)
                    raw_t3d = _pad_or_crop_tracks(raw_t3d, N_t, fill=0.0)
                    raw_vis = _pad_or_crop_tracks(raw_vis, N_t, fill=0.0)
                    raw_lbl = _pad_or_crop_tracks(raw_lbl, N_t, fill=0.0)

                    # Convert normalized 2D → pixel coords in current resolution
                    tracks_2d  = torch.from_numpy(raw_t2d).float()
                    tracks_2d[..., 0] *= W
                    tracks_2d[..., 1] *= H
                    tracks_3d  = torch.from_numpy(raw_t3d).float()
                    tracks_vis = torch.from_numpy(raw_vis).float()

                    # Pad/crop part axis to P
                    P_cache = raw_lbl.shape[1]
                    if P_cache < P:
                        pad = np.zeros((N_t, P - P_cache), dtype=raw_lbl.dtype)
                        raw_lbl = np.concatenate([raw_lbl, pad], axis=1)
                    elif P_cache > P:
                        raw_lbl = raw_lbl[:, :P]
                    track_part_label = torch.from_numpy(raw_lbl).float()

                    # Resize motion_mask to (P, H_p, W_p) if cache resolution differs
                    P_cache, h_c, w_c = raw_msk.shape
                    mm = torch.from_numpy(raw_msk).float().unsqueeze(0)  # [1,P_c,h,w]
                    if (h_c, w_c) != (H_p, W_p):
                        mm = F.interpolate(mm, size=(H_p, W_p),
                                           mode="bilinear", align_corners=False)
                    mm = mm.squeeze(0)                                    # [P_c,H_p,W_p]
                    if P_cache < P:
                        motion_mask[:P_cache] = mm
                    else:
                        motion_mask = mm[:P]

                    has_motion_data = True
            except Exception:
                # Corrupt cache: fall back to zero placeholders.
                has_motion_data = False

        # ── Timestamps ────────────────────────────────────────────────────
        if S == 1:
            timestamps = torch.zeros(1)
        else:
            timestamps = torch.linspace(0.0, 1.0, S)

        return {
            "images":           images,           # [S, 3, H, W]
            "extrinsics":       extrinsics,        # [S, 4, 4]
            "intrinsics":       intrinsics,        # [3, 3]
            "timestamps":       timestamps,        # [S]
            "part_masks":       part_masks,        # [S, P, H, W]
            "static_body_mask": static_body_mask,  # [S, H, W]  pure static body (no bg)
            "has_static_mask":  has_static_mask,   # bool
            "pseudo_masks":     pseudo_masks,      # [S, P, H, W]  (zeros if phase!="2")
            "has_pseudo_masks": has_pseudo,        # bool
            "depth":            depth,             # [S, H, W]      (zeros if missing)
            "has_depth":        any_depth,         # bool
            "tracks_2d":        tracks_2d,         # [S, N_t, 2]    pixel coords (zeros if missing)
            "tracks_3d":        tracks_3d,         # [S, N_t, 3]    world meters (zeros if missing)
            "tracks_vis":       tracks_vis,        # [S, N_t]       (zeros if missing)
            "motion_mask":      motion_mask,       # [P, H_p, W_p]  (zeros if missing)
            "track_part_label": track_part_label,  # [N_t, P]       (zeros if missing)
            "has_motion_data":  has_motion_data,   # bool
            "gt_motion_type":   gt_motion_type,    # [P]  long
            "gt_axis":          gt_axis,           # [P, 3]
            "gt_pivot":         gt_pivot,          # [P, 3]
            "gt_scalars":       gt_scalars,        # [P, S]
            "has_pose":         True,
            "scene_id":         (
                f"{scene_root.name}/{cam_dir.name}"
                if cam_dir is not None else scene_root.name
            ),
            "n_active_parts":   n_joints + 1,
        }
