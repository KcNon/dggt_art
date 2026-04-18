#!/usr/bin/env python3
"""Visualize precomputed motion_mask / tracks for validation.

For a given (scene, cam) with ``motion_cache.npz`` already produced by
``precompute_motion_data.py``, this script writes a PNG showing:

  * frame 0 RGB with per-track colored dots (color = argmax over P parts)
  * the per-part motion_mask as a P-panel grid alongside
  * GT part masks overlaid for reference (if available)

Usage::

    python scripts/vis_motion_mask.py \\
        --cam_dir /data2/cyt/data_root_refine/100191/cam_01 \\
        --out /tmp/motion_vis/100191_cam01.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


DISTINCT_COLORS = np.array([
    [ 80,  80,  80],  # 0 static base (grey)
    [255,  80,  80],  # 1 red
    [ 80, 255,  80],  # 2 green
    [ 80,  80, 255],  # 3 blue
    [255, 255,  80],  # 4 yellow
    [255,  80, 255],  # 5 magenta
    [ 80, 255, 255],  # 6 cyan
    [255, 180,  50],  # 7 orange
], dtype=np.uint8)


def compare_to_gt(part_masks_first, motion_mask):
    """Compute per-slot IoU between binarized motion_mask (patch res) vs GT.

    part_masks_first: [P, H, W]  first frame
    motion_mask:      [P, H_p, W_p]
    Returns: list of IoU per slot (argmax-based binarization).
    """
    P, H, W = part_masks_first.shape
    P2, H_p, W_p = motion_mask.shape
    # Upsample motion_mask to (H, W) for comparison
    mm = torch.from_numpy(motion_mask).unsqueeze(0)
    mm = F.interpolate(mm, size=(H, W), mode="bilinear", align_corners=False)
    mm = mm.squeeze(0).numpy()
    pred = mm.argmax(0)                          # [H, W]

    # GT argmax
    gt = part_masks_first.argmax(0)              # [H, W]

    ious = []
    for k in range(min(P, P2)):
        p_m = (pred == k)
        g_m = (gt == k)
        inter = (p_m & g_m).sum()
        union = (p_m | g_m).sum()
        ious.append(inter / max(union, 1))
    return ious


def render_vis(cam_dir: Path, out_path: Path):
    cache = np.load(str(cam_dir / "motion_cache.npz"), allow_pickle=True)
    frame_ids      = [str(x) for x in cache["frame_ids"].tolist()]
    tracks_2d_norm = cache["tracks_2d_norm"]         # [S, N, 2]
    t_vis          = cache["tracks_vis"]             # [S, N]
    track_label    = cache["track_part_label"]       # [N, P]
    motion_mask    = cache["motion_mask"]            # [P, H_p, W_p]

    # Load frame 0
    img_dir = cam_dir / "images"
    f0 = frame_ids[0]
    for ext in (".jpg", ".png"):
        p = img_dir / f"{f0}{ext}"
        if p.exists():
            img = np.asarray(Image.open(p).convert("RGB"))
            break
    H, W = img.shape[:2]

    # Track overlay
    overlay = img.copy()
    px = (tracks_2d_norm[0, :, 0] * W).astype(int)
    py = (tracks_2d_norm[0, :, 1] * H).astype(int)
    assign = track_label.argmax(-1)                    # [N]
    for n in range(len(px)):
        if t_vis[0, n] < 0.5:
            continue
        x, y = px[n], py[n]
        if 2 <= x < W - 2 and 2 <= y < H - 2:
            color = DISTINCT_COLORS[assign[n] % len(DISTINCT_COLORS)]
            overlay[y - 2:y + 3, x - 2:x + 3] = color

    # motion_mask argmax upsampled
    mm_t = torch.from_numpy(motion_mask).unsqueeze(0)
    mm_up = F.interpolate(mm_t, size=(H, W), mode="bilinear",
                          align_corners=False).squeeze(0).numpy()
    mm_arg = mm_up.argmax(0)
    mm_vis = DISTINCT_COLORS[mm_arg % len(DISTINCT_COLORS)]  # [H, W, 3]

    # Blend mm over original image
    blend = (img * 0.45 + mm_vis * 0.55).astype(np.uint8)

    # Try GT comparison
    iou_str = ""
    masks_dir = cam_dir / "part_masks"
    if masks_dir.exists():
        with open(cam_dir.parent / "joint_params.json") as f:
            import json
            n_joints = len(json.load(f))
        P = motion_mask.shape[0]
        gt = np.zeros((P, H, W), dtype=np.float32)
        slot_union = np.zeros((H, W), dtype=np.float32)
        # Cap at P-1 to leave slot 0 for static base when n_joints ≥ P.
        for k in range(min(n_joints, P - 1)):
            p = k + 1
            pid = k + 2
            mp = masks_dir / f"{f0}_{pid}.png"
            if mp.exists():
                mk = np.asarray(Image.open(mp).convert("L")) / 255.0
                mk = (mk > 0.5).astype(np.float32)
                if mk.shape != (H, W):
                    from PIL import Image as PI
                    mk = np.asarray(
                        PI.fromarray((mk * 255).astype(np.uint8))
                        .resize((W, H), PI.NEAREST)) / 255.0
                    mk = (mk > 0.5).astype(np.float32)
                gt[p] = mk
                slot_union = np.clip(slot_union + mk, 0, 1)
        root_pid = n_joints + 2
        rp = masks_dir / f"{f0}_{root_pid}.png"
        root = np.zeros((H, W), dtype=np.float32)
        if rp.exists():
            root = np.asarray(Image.open(rp).convert("L")) / 255.0
            root = (root > 0.5).astype(np.float32)
            if root.shape != (H, W):
                from PIL import Image as PI
                root = np.asarray(
                    PI.fromarray((root * 255).astype(np.uint8))
                    .resize((W, H), PI.NEAREST)) / 255.0
                root = (root > 0.5).astype(np.float32)
        gt[0] = np.clip((1 - slot_union) + root, 0, 1)
        ious = compare_to_gt(gt, motion_mask)
        iou_str = "IoU per slot: " + ", ".join(f"{v:.2f}" for v in ious)

    # Compose: [original | tracks | blended]
    out_img = np.concatenate([img, overlay, blend], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img).save(str(out_path))
    print(f"Saved {out_path}")
    if iou_str:
        print(iou_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    render_vis(Path(args.cam_dir), Path(args.out))


if __name__ == "__main__":
    main()
