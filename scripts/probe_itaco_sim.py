"""
Three-in-one pre-loader probe for iTACO sim_data:
  (1) camera_pose quat order (xyzw vs wxyz) — verify via depth back-projection
      stationary check on bg pixels (segment value == 1).
  (2) MonST3R dynamic_mask file naming + frame indexing.
  (3) segment_id ↔ active-actor mapping (Plan A: actor with biggest pose var).

Read-only. Prints one block per scene.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cyt/video2articulation/sim_data/partnet_mobility")
PREP = Path("/data2/cyt/video2articulation/sim_data/exp_results/preprocessing")


def quat_wxyz_to_R(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def quat_xyzw_to_R(q):
    x,y,z,w = q
    return quat_wxyz_to_R([w,x,y,z])


def cam_pose_to_c2w(pose7, quat_order):
    """pose7: [T,7] xyz+quat. Returns [T,4,4] c2w."""
    T = pose7.shape[0]
    M = np.zeros((T,4,4), dtype=np.float64); M[:,3,3]=1
    fn = quat_wxyz_to_R if quat_order=="wxyz" else quat_xyzw_to_R
    for t in range(T):
        M[t,:3,:3] = fn(pose7[t,3:7])
        M[t,:3, 3] = pose7[t,:3]
    return M


def back_project_bg(rgb_idx, depth_uint16, K, c2w, seg, bg_label=1):
    """
    Sample a few bg pixels, back-project to world via c2w; return points.
    """
    H,W = depth_uint16.shape
    # 50 random bg pixels
    yy, xx = np.where(seg == bg_label)
    if len(yy)==0:
        return None
    n = min(50, len(yy))
    idx = np.random.RandomState(0).choice(len(yy), n, replace=False)
    py, px = yy[idx], xx[idx]
    z = depth_uint16[py, px].astype(np.float64) / 1000.0  # meters
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (px - cx) * z / fx
    Y = (py - cy) * z / fy
    Z = z
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)   # [n,4]
    pts_w = (c2w @ pts_cam.T).T[:, :3]                      # [n,3]
    return pts_w


def probe_scene(cat, inst, jdir, view):
    print(f"\n{'='*70}\n  {cat}/{inst}/{jdir}/{view}\n{'='*70}")
    raw_j = ROOT / cat / inst / jdir
    raw_v = raw_j / view
    prep_v = PREP / cat / inst / jdir / view

    K = np.load(raw_v / "intrinsics.npy").astype(np.float64)
    cp = np.load(raw_v / "camera_pose.npy").astype(np.float64)   # [T,7]
    T = cp.shape[0]
    print(f"  T={T}  K=[fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}]")

    # ---- (1) quat order probe via depth back-projection stationarity ----
    print("\n  (1) Quat order probe (background world-points should be ~stationary)")
    # use first and last frame
    t0, t1 = 0, T-1
    seg0 = np.load(raw_v / "segment" / f"{t0:06d}.npz")["a"]
    seg1 = np.load(raw_v / "segment" / f"{t1:06d}.npz")["a"]
    d0   = np.load(raw_v / "depth"   / f"{t0:06d}.npz")["a"]
    d1   = np.load(raw_v / "depth"   / f"{t1:06d}.npz")["a"]

    bg_label = 1 if (seg0 == 1).sum() > 100 else int(np.bincount(seg0.flatten()).argmax())
    print(f"      using bg_label = {bg_label}  (count_t0={(seg0==bg_label).sum()})")

    for order in ("wxyz", "xyzw"):
        c2w = cam_pose_to_c2w(cp, order)
        # camera_pose is most likely camera-to-world; if it's world-to-camera we need inv.
        # Try both interpretations:
        for label, M in [("c2w-direct", c2w),
                          ("inv (w2c→c2w)", np.linalg.inv(c2w))]:
            p0 = back_project_bg(t0, d0, K, M[t0], seg0, bg_label)
            p1 = back_project_bg(t1, d1, K, M[t1], seg1, bg_label)
            if p0 is None or p1 is None:
                continue
            # pick same number of points (smaller); use approximate matching by random subset
            n = min(len(p0), len(p1))
            d_centroid = np.linalg.norm(p0[:n].mean(0) - p1[:n].mean(0))
            d_scatter  = float(p0[:n].std(0).mean())
            print(f"      order={order:5s} {label:18s} bg_centroid_drift={d_centroid:.3f} m   bg_scatter={d_scatter:.3f}")

    # ---- (2) MonST3R dynamic_mask names ----
    print("\n  (2) MonST3R dynamic_mask listing")
    if (prep_v / "monst3r").exists():
        names = sorted(p.name for p in (prep_v / "monst3r").glob("dynamic_mask_*.png"))
        idxs  = sorted(int(p.name.split("_")[-1].split(".")[0]) for p in (prep_v / "monst3r").glob("dynamic_mask_*.png"))
        print(f"      n={len(names)}  index range = [{min(idxs)}, {max(idxs)}]  total_video_frames={T}")
        print(f"      first 3 names: {names[:3]}")
        # likely the indices are sparse keyframes (subset of [0..T-1]).
        if max(idxs) < T:
            stride = T / max(1, len(idxs))
            print(f"      ratio T/n_masks = {stride:.2f}  → looks like every ~{int(round(stride))}-th frame")
    else:
        print("      [missing]")

    # ---- (3) segment_id ↔ active actor mapping ----
    print("\n  (3) Per-actor pose variance (Plan A active-part detection)")
    with open(raw_j / "actor_pose.pkl", "rb") as f:
        ap = pickle.load(f)
    act_names = sorted(ap.keys())
    var_scores = {}
    for k, lst in ap.items():
        arr = np.stack(lst)              # [T, 7]
        trans_std = float(arr[:, :3].std(0).sum())
        rot_std   = float(arr[:, 3:7].std(0).sum())
        var_scores[k] = trans_std + rot_std
    sorted_actors = sorted(var_scores.items(), key=lambda kv: -kv[1])
    print(f"      actor variance ranking: {[(k,f'{v:.4f}') for k,v in sorted_actors]}")

    seg_uniq = np.unique(seg0)
    print(f"      segment unique values @ t0: {seg_uniq.tolist()}")
    # actor name suffix (actor_X) -> we expect X to match a segment value
    actor_to_seg_id = {}
    for k in act_names:
        try:
            x = int(k.split("_")[1])
            actor_to_seg_id[k] = x if x in seg_uniq else None
        except Exception:
            actor_to_seg_id[k] = None
    matched = sum(1 for v in actor_to_seg_id.values() if v is not None)
    print(f"      actor→segment hit-rate: {matched}/{len(act_names)}  map={actor_to_seg_id}")

    active_actor = sorted_actors[0][0]
    active_seg   = actor_to_seg_id.get(active_actor)
    print(f"      → ACTIVE actor = {active_actor}, predicted active segment_id = {active_seg}")
    if active_seg is not None:
        cnt0 = int((seg0 == active_seg).sum())
        cnt1 = int((seg1 == active_seg).sum())
        print(f"      pixel count of active_seg: t0={cnt0} ({100*cnt0/seg0.size:.1f}%)  t1={cnt1} ({100*cnt1/seg1.size:.1f}%)")


def main():
    samples = [
        ("Box",            "100129", "joint_0_bg", "view_0"),
        ("Laptop",         None,     None,         "view_0"),
        ("StorageFurniture", None,   None,         "view_0"),
        ("USB",            None,     None,         "view_0"),  # has slider
    ]
    for cat, inst, jdir, view in samples:
        inst_root = ROOT / cat
        if not inst_root.exists():
            print(f"\n[skip] {cat} not found"); continue
        if inst is None:
            inst = sorted(p.name for p in inst_root.iterdir() if p.is_dir())[0]
        if jdir is None:
            jdir = sorted(p.name for p in (inst_root/inst).iterdir()
                          if p.is_dir() and p.name.startswith("joint_"))[0]
        try:
            probe_scene(cat, inst, jdir, view)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {cat}/{inst}/{jdir}/{view}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
