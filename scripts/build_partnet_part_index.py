"""
build_partnet_part_index.py — 为 data_processed 下每个场景生成 part_index.json

背景
----
新数据 (process.py 生成) 的 part_masks 文件名 {frame}_{seg_id}.png 用的是
SAPIEN 渲染时的全局 per_scene_id (绝对 link id)，而非 ArticulatedDataset
约定的 joint_k -> k+2。seg_id 与 joint 的对应关系在生成时没有保存。

本脚本通过「精确复刻 process.py 的全局加载顺序」来无重渲染地复原映射：
  * per_scene_id 是一个不重置的全局递增计数器；
  * process.py 在一个共享 Scene 里依 extracted_ids.json 顺序加载所有模型，
    每个模型消耗 (link 数 + 4 个相机) 个 id，开头 2 个平行光占 id 1,2；
  * 因此只要按相同顺序重放 (load URDF + add 4 cam + remove)，即可复现每个
    link 被烧进 mask 文件名的那个 seg_id。

对每个 scene 写出 {scene_root}/part_index.json:
  {
    "version": 1,
    "n_joints": 4,
    "joint_id": {"joint_0": 83, "joint_1": 84, ...},  # joint_name -> 烧进mask的seg_id
    "static_ids": [81, 82],                             # 非active-joint子件 -> 静态
    "validated": true                                   # 该scene磁盘上的动态id是否都已命中
  }

loader 侧:  joint k (按 sorted(joint_params.keys()) 的下标) -> slot k+1,
            其 mask 文件 = {frame}_{joint_id[name]}.png;
            其余所有 {frame}_*.png -> 并入静态 slot 0。

用法:
  python scripts/build_partnet_part_index.py \
      --dataset_root  /data2/lza/partnet-Mobility/dataset \
      --processed_root /data2/lza/partnet-Mobility/data_processed \
      --ids_json      /data2/lza/partnet-Mobility/extracted_ids.json
"""

import argparse
import glob
import json
import os

import numpy as np
import sapien


def disk_mask_ids(scene_dir: str) -> set:
    """该 scene 全部 cam 的 part_masks 里出现过的 seg_id 集合。"""
    s = set()
    for f in glob.glob(os.path.join(scene_dir, "cam_*", "part_masks", "*.png")):
        try:
            s.add(int(os.path.basename(f).split("_")[1].split(".")[0]))
        except Exception:
            pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="原始 PartNet-Mobility URDF 根目录 (含 {id}/mobility.urdf)")
    ap.add_argument("--processed_root", required=True,
                    help="process.py 输出根目录 data_processed")
    ap.add_argument("--ids_json", required=True,
                    help="extracted_ids.json，决定加载顺序")
    ap.add_argument("--dry_run", action="store_true",
                    help="只校验不写文件")
    args = ap.parse_args()

    sapien.set_log_level("off")

    # ── 精确复刻 process.py 的 create_scene ────────────────────────────────
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 1, -1], [0.8, 0.8, 0.8], shadow=True)
    scene.add_directional_light([-1, -1, -1], [0.5, 0.5, 0.5], shadow=False)

    with open(args.ids_json, "r", encoding="utf-8") as f:
        category_ids = json.load(f)

    n_written = n_validated = n_partial = n_no_disk = 0
    failures = []

    for category, model_ids in category_ids.items():
        for model_id in model_ids:
            mid = str(model_id)
            urdf = os.path.join(args.dataset_root, mid, "mobility.urdf")
            if not os.path.exists(urdf):
                # process.py 在此处直接 continue，不消耗任何 id
                continue

            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            try:
                art = loader.load(urdf)
            except Exception:
                continue
            if art is None:
                continue

            # —— 关键：复刻 process.py 的 id 消耗 ——
            # 1) articulation 的 link 此刻已拿到 per_scene_id
            active_joints = art.get_active_joints()
            joint_id = {}
            for j in active_joints:
                cl = j.get_child_link()
                joint_id[j.name] = int(cl.entity.per_scene_id)
            all_link_ids = set(int(l.entity.per_scene_id) for l in art.get_links())

            # 2) process.py 在 load 之后为每个模型创建 4 个相机 (也占 id)
            cams = [
                scene.add_camera(name=f"c{i}", width=32, height=24,
                                 fovy=np.deg2rad(35), near=0.1, far=100)
                for i in range(4)
            ]

            scene_dir = os.path.join(args.processed_root, category, mid)
            if os.path.isdir(scene_dir):
                disk = disk_mask_ids(scene_dir)
                if disk:
                    dyn_ids = set(joint_id.values())
                    static_ids = sorted(all_link_ids - dyn_ids)
                    # 校验: 磁盘上属于动态关节的那些 id 是否都能被本场景解释
                    matched_dyn = dyn_ids & disk
                    missing_dyn = dyn_ids - disk   # 该关节件在所有视角都不可见(罕见)
                    stray = disk - all_link_ids    # 杂散(灯光/背景) -> 归静态

                    validated = (len(stray) == 0)
                    if not validated and len(disk & all_link_ids) >= 1 and len(dyn_ids & disk) == len(dyn_ids - missing_dyn):
                        # 有杂散但所有可见动态件都命中 -> 仍可用
                        n_partial += 1
                        validated_flag = True
                    elif validated:
                        n_validated += 1
                        validated_flag = True
                    else:
                        failures.append((category, mid, sorted(disk), sorted(all_link_ids)))
                        validated_flag = False

                    part_index = {
                        "version": 1,
                        "n_joints": len(active_joints),
                        "joint_id": joint_id,            # name -> seg_id
                        "static_ids": static_ids,        # 非动态 link 的 seg_id
                        "validated": bool(validated_flag),
                    }
                    if not args.dry_run:
                        with open(os.path.join(scene_dir, "part_index.json"), "w") as wf:
                            json.dump(part_index, wf, indent=2)
                        n_written += 1
                else:
                    n_no_disk += 1

            for c in cams:
                scene.remove_camera(c)
            scene.remove_articulation(art)

    print("=" * 60)
    print(f"写出 part_index.json   : {n_written}")
    print(f"  精确匹配             : {n_validated}")
    print(f"  含杂散但动态全命中   : {n_partial}")
    print(f"磁盘有目录但无mask     : {n_no_disk}")
    print(f"真正错配(未写validated): {len(failures)}")
    for c, m, d, l in failures[:20]:
        print(f"  FAIL {c}/{m} disk={d[:10]} links={l[:10]}")


if __name__ == "__main__":
    main()
