"""
Smoke-test iTACOSimDataset:
  • print one __getitem__ payload's shape/dtype per field
  • for 3 scenes, dump frame 0 RGB | part_mask overlay | dyn mask side-by-side
  • check axis/pivot project to plausible image location
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.itaco_sim_dataset import iTACOSimDataset


def _project(K, R_w2c, t_w2c, p_w):
    """OpenGL convention: cam looks at -Z, depth = -z_cam, pixel y flipped."""
    p_cam = R_w2c @ p_w + t_w2c
    z = float(p_cam[2])
    if z >= -1e-6:        # behind camera (in OpenGL, in front is z < 0)
        return None
    depth = -z
    u = K[0, 0] * p_cam[0] / depth + K[0, 2]
    v = K[1, 1] * (-p_cam[1]) / depth + K[1, 2]   # OpenGL +Y up → -v in pixel
    return float(u), float(v), depth


def overlay_mask(rgb_uint8: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5):
    out = rgb_uint8.copy()
    m = mask > 0.5
    if m.any():
        out[m] = (alpha * np.array(color) + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def main():
    ds = iTACOSimDataset(
        target_size=518, num_frames=8, max_parts=8,
        categories=["Box", "Laptop", "StorageFurniture"],
        random_start=False,
    )
    print(f"\n[smoke] dataset size = {len(ds)}")
    out_dir = Path("/data2/cyt/eval/smoke_itaco_sim")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(6, len(ds))
    for i in range(n_show):
        b = ds[i]
        scene = b["scene_id"]
        print(f"\n--- {scene} ---")
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:18s}  shape={list(v.shape)}  dtype={v.dtype}")
            else:
                print(f"  {k:18s}  = {v}")

        # ----- visualize -----
        rgb = (b["images"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)   # [H,W,3]
        # Active part = slot 1
        mask_active = b["part_masks"][0, 1].numpy()
        mask_all = (b["part_masks"][0, 1:].sum(0).numpy() > 0.5).astype(np.float32)
        bg = b["part_masks"][0, 0].numpy()
        # Motion mask (patch-res) → upsample to img res
        H, W = rgb.shape[:2]
        mm = b["motion_mask"][1].numpy()
        mm_img = np.array(Image.fromarray((mm * 255).astype(np.uint8)).resize((W, H), Image.NEAREST))

        v_active = overlay_mask(rgb, mask_active, color=(255, 50, 50), alpha=0.5)
        v_all    = overlay_mask(rgb, mask_all,    color=(50, 255, 50), alpha=0.4)
        v_mm     = overlay_mask(rgb, (mm_img > 60).astype(np.float32),
                                color=(50, 50, 255), alpha=0.5)

        # ----- project pivot to frame 0 (sanity check) -----
        c2w0 = b["extrinsics"][0].numpy()
        w2c0 = np.linalg.inv(c2w0)
        Rw, tw = w2c0[:3, :3], w2c0[:3, 3]
        K = b["intrinsics"].numpy()
        pv = b["gt_pivot"][1].numpy()
        ax = b["gt_axis"][1].numpy()
        proj_p = _project(K, Rw, tw, pv)
        proj_a = _project(K, Rw, tw, pv + 0.1 * ax)
        print(f"  active_seg pixels @ t0: {int(mask_active.sum())}  "
              f"pivot_world={pv}  axis_world={ax}")
        print(f"  pivot →px {proj_p}   pivot+0.1*axis →px {proj_a}")

        # Draw pivot dot if in frame
        if proj_p is not None:
            u, v, z = proj_p
            if 0 <= u < W and 0 <= v < H and z > 0:
                ui, vi = int(u), int(v)
                v_active[max(0,vi-3):vi+4, max(0,ui-3):ui+4] = (255, 255, 0)

        canvas = np.concatenate([rgb, v_active, v_all, v_mm], axis=1)
        out = out_dir / f"{i:02d}_{scene.replace('/','_')}.png"
        Image.fromarray(canvas).save(out)
        print(f"  saved {out}")

    print(f"\nDone. Visualisations → {out_dir}")


if __name__ == "__main__":
    main()
