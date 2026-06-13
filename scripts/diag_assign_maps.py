"""
Diagnose phase-1a feed-forward segmentation: load a checkpoint, run inference on a
few scenes, and dump the predicted assignment (argmax over P part slots + bg sink)
next to GT, plus per-slot mass / dead-slot / IoU stats.

Tells us whether the IoU plateau is: bg-sink eating foreground, slot collapse/
duplication, dead slots, or just noisy matching.

Usage:
  python scripts/diag_assign_maps.py --ckpt .../ckpt_020000.pth --scenes 40453 100202
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from dggt.models.art_vggt import ArtVGGT
from datasets.articulated_dataset import ArticulatedDataset
from dggt.utils.dead_slot_gating import detect_dead_slots, compute_slot_mass
from dggt.utils.hungarian_matching import batch_hungarian_match

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--scenes", nargs="+", default=["40453", "100202"])
ap.add_argument("--res", type=int, default=518)
ap.add_argument("--out", default="/data5/lza/checkpoint/Art/overfit")
args = ap.parse_args()
dev = "cuda"
os.makedirs(args.out, exist_ok=True)

# Palette: slot id -> RGB. bg (last) = black.
PAL = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [0, 0, 0],  # bg
], dtype=np.uint8)

ds = ArticulatedDataset(data_root="/data2/lza/partnet-Mobility/data_processed",
                        target_size=args.res, num_frames=8, max_parts=8, phase="1",
                        split="all", exclude_cams=set())

print(f"loading {args.ckpt}")
sd = torch.load(args.ckpt, map_location="cpu")
step = sd.get("step", "?")
model = ArtVGGT(img_size=args.res, patch_size=14, embed_dim=1024, num_slots=8,
                n_gaussians=256, scene_radius=1.0, use_camera_head=False).to(dev)
missing, unexpected = model.load_state_dict(sd["model"], strict=False)
print(f"  step={step}  missing={len(missing)} unexpected={len(unexpected)}")
if missing:  print("  e.g. missing:", missing[:5])
model.eval()

for scene in args.scenes:
    ents = [i for i in range(len(ds)) if ds.entries[i][0].name == scene]
    if not ents:
        print(f"scene {scene} not found"); continue
    batch = default_collate([ds[ents[0]]])
    images = batch["images"].to(dev); extr = batch["extrinsics"].to(dev)
    intr = batch["intrinsics"].to(dev); ts = batch["timestamps"].to(dev)
    H = W = args.res
    with torch.no_grad():
        preds = model(images, extr, intr, ts)
    assign = preds["assign_maps"]              # [1,P,Hp,Wp]
    bg = preds.get("bg_map")                    # [1,1,Hp,Wp] or None
    P = assign.shape[1]
    mass = compute_slot_mass(assign)[0]         # [P]
    is_dead = detect_dead_slots(assign)[0]      # [P]
    bg_mass = bg[0].sum().item() if bg is not None else 0.0
    tot = assign[0].sum().item() + bg_mass + 1e-6

    # GT frame-0 masks + Hungarian match
    gt = (batch["part_masks"][:, 0] > 0.5).float().to(dev)   # [1,P_gt,H,W]
    up = F.interpolate(assign, (H, W), mode="bilinear", align_corners=False)
    matches = batch_hungarian_match(up, gt)
    pred_idx, gt_idx = matches[0]

    # argmax seg incl bg
    if bg is not None:
        bgu = F.interpolate(bg, (H, W), mode="bilinear", align_corners=False)
        seg_src = torch.cat([up, bgu], dim=1)   # [1,P+1,H,W]
        bg_label = P
    else:
        seg_src = up; bg_label = P
    pred_lab = seg_src[0].argmax(0).cpu().numpy()     # [H,W] in 0..P (P=bg)

    # GT label map (disjoint modal masks); default bg
    gt_lab = np.full((H, W), P, dtype=np.int64)
    for gi in range(gt.shape[1]):
        gt_lab[(gt[0, gi] > 0.5).cpu().numpy()] = gi

    # remap pred slot ids -> gt ids for consistent coloring (matched pairs)
    remap = {int(pi): int(gi) for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist())}
    pred_lab_c = np.full_like(pred_lab, P)
    for pi in range(P):
        pred_lab_c[pred_lab == pi] = remap.get(pi, pi)
    pred_lab_c[pred_lab == P] = P  # bg stays bg

    # per matched-pair IoU
    print(f"\n=== scene {scene}  (P={P}) ===")
    print(f"  slot mass frac: " + " ".join(f"s{p}={mass[p].item()/tot:.3f}" for p in range(P))
          + f"  | bg={bg_mass/tot:.3f}")
    print(f"  dead slots: {[p for p in range(P) if is_dead[p]]}")
    ious = []
    for pi, gi in zip(pred_idx.tolist(), gt_idx.tolist()):
        pm = (pred_lab == pi); gm = (gt_lab == gi)
        inter = np.logical_and(pm, gm).sum(); uni = np.logical_or(pm, gm).sum()
        if gm.sum() < 10:  continue
        iou = inter / (uni + 1e-6); ious.append(iou)
        print(f"    pred s{pi} -> gt {gi}: IoU={iou:.3f}  (gt px={int(gm.sum())}, pred px={int(pm.sum())})")
    print(f"  mIoU={np.mean(ious) if ious else 0:.3f}")

    # panel: input | GT | pred
    img0 = (images[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_rgb = PAL[gt_lab]
    pred_rgb = PAL[pred_lab_c]
    panel = np.concatenate([img0, gt_rgb, pred_rgb], axis=1)
    path = os.path.join(args.out, f"diag_seg_{scene}_step{step}.png")
    Image.fromarray(panel).save(path)
    print(f"  saved {path}  (input | GT | pred)")
