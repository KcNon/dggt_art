"""
Read-only inspection of iTACO sim_data mystery files:
  segment/*.npz, actor_pose.pkl, joint_id_list.txt, gt_joint_value.npy, qpos.npy
Picks one (Cat, instance, joint, view) sample and dumps every relevant field.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cyt/video2articulation/sim_data/partnet_mobility")
PREP = Path("/data2/cyt/video2articulation/sim_data/exp_results/preprocessing")


def _h(arr):
    if isinstance(arr, np.ndarray):
        return f"ndarray shape={arr.shape} dtype={arr.dtype} min={float(arr.min()) if arr.size else 'n/a'} max={float(arr.max()) if arr.size else 'n/a'}"
    return f"{type(arr).__name__}: {arr}"


def inspect_one(cat: str, inst: str, joint_dir: str, view: str):
    print(f"\n{'='*70}\n  {cat}/{inst}/{joint_dir}/{view}\n{'='*70}")
    raw_j = ROOT / cat / inst / joint_dir
    raw_v = raw_j / view
    prep_v = PREP / cat / inst / joint_dir / view

    # --- joint-level files ---
    print("\n[meta.json]")
    print(json.dumps(json.loads((raw_j / "meta.json").read_text()), indent=2))

    print("\n[joint_id_list.txt]")
    print((raw_j / "joint_id_list.txt").read_text())

    gt = np.load(raw_j / "gt_joint_value.npy")
    print(f"\n[gt_joint_value.npy]  {_h(gt)}\n  first 5: {gt[:5]}\n  last  5: {gt[-5:]}")

    qpos = np.load(raw_j / "qpos.npy")
    print(f"\n[qpos.npy]  {_h(qpos)}")
    if qpos.ndim == 2:
        print(f"  qpos[0]:  {qpos[0]}")
        print(f"  qpos[-1]: {qpos[-1]}")

    print("\n[actor_pose.pkl]")
    with open(raw_j / "actor_pose.pkl", "rb") as f:
        ap = pickle.load(f)
    print(f"  type: {type(ap).__name__}")
    if isinstance(ap, dict):
        for k, v in ap.items():
            print(f"  key={k!r}: {_h(v) if isinstance(v, np.ndarray) else type(v).__name__}")
            if isinstance(v, np.ndarray) and v.ndim <= 2 and v.size <= 30:
                print(f"      values: {v}")
            elif isinstance(v, np.ndarray):
                print(f"      first row: {v[0] if v.ndim>=1 else v}")
                print(f"      last  row: {v[-1] if v.ndim>=1 else v}")
            elif isinstance(v, list) and len(v) <= 5:
                for i, item in enumerate(v):
                    print(f"      [{i}]: {_h(item) if isinstance(item, np.ndarray) else item}")
            elif isinstance(v, list):
                print(f"      list len={len(v)} sample[0]: {_h(v[0]) if isinstance(v[0], np.ndarray) else type(v[0]).__name__}")
    elif isinstance(ap, (list, tuple)):
        print(f"  len={len(ap)}; first elem type {type(ap[0]).__name__}")
        if hasattr(ap[0], "shape"):
            print(f"  first elem: {_h(ap[0])}")

    # --- view-level files ---
    print("\n[camera_pose.npy]")
    cp = np.load(raw_v / "camera_pose.npy")
    print(f"  {_h(cp)}")
    print(f"  pose[0]:\n{cp[0] if cp.ndim==3 else cp}")

    print("\n[intrinsics.npy]")
    ki = np.load(raw_v / "intrinsics.npy")
    print(f"  {_h(ki)}")
    print(f"  K:\n{ki}")

    print("\n[depth/000000.npz]")
    dz = np.load(raw_v / "depth" / "000000.npz")
    print(f"  keys: {list(dz.keys())}")
    for k in dz.keys():
        print(f"  '{k}': {_h(dz[k])}")

    print("\n[segment/000000.npz]")
    sz = np.load(raw_v / "segment" / "000000.npz", allow_pickle=True)
    print(f"  keys: {list(sz.keys())}")
    for k in sz.keys():
        v = sz[k]
        print(f"  '{k}': {_h(v)}")
        if isinstance(v, np.ndarray) and v.dtype != object:
            uniq = np.unique(v.flatten())[:20]
            print(f"    unique[:20]: {uniq}")
        elif isinstance(v, np.ndarray) and v.dtype == object:
            try:
                obj = v.item()
                print(f"    .item() type: {type(obj).__name__}, content sample: {repr(obj)[:300]}")
            except Exception as e:
                print(f"    .item() failed: {e}")

    # rgb count
    rgb_dir = raw_v / "rgb"
    if rgb_dir.exists():
        n_rgb = len(list(rgb_dir.glob("*.jpg")))
        print(f"\n[rgb/]  n_frames={n_rgb}")

    # preprocessing
    print(f"\n[preprocessing exists?]  monst3r={prep_v/'monst3r'} -> {(prep_v/'monst3r').exists()}")
    print(f"                         video_segment_reverse={prep_v/'video_segment_reverse'} -> {(prep_v/'video_segment_reverse').exists()}")
    if (prep_v / "monst3r").exists():
        dm = sorted((prep_v / "monst3r").glob("dynamic_mask_*.png"))
        print(f"  dynamic_mask_*.png count = {len(dm)}")


def main():
    # Pick 3 different categories to ensure we don't miss schema variants
    samples = [
        ("Box",            "100129", "joint_0_bg", "view_0"),
        ("Laptop",         None,     None,         "view_0"),  # auto-pick first
        ("StorageFurniture", None,   None,         "view_0"),
    ]
    for cat, inst, jdir, view in samples:
        inst_root = ROOT / cat
        if not inst_root.exists():
            print(f"\n[skip] {cat} not found")
            continue
        if inst is None:
            inst = sorted(p.name for p in inst_root.iterdir() if p.is_dir())[0]
        if jdir is None:
            jdir = sorted(p.name for p in (inst_root/inst).iterdir() if p.is_dir() and p.name.startswith("joint_"))[0]
        try:
            inspect_one(cat, inst, jdir, view)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {cat}/{inst}/{jdir}/{view}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
