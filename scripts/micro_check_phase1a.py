"""Quick check that the mask_loss base-pair fix lets slot0 (base part) come alive.
Overfit phase-1a mask loss on ONE scene; print slot0 mass + base IoU per step."""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from dggt.models.art_vggt import ArtVGGT
from datasets.articulated_dataset import ArticulatedDataset
from dggt.utils.dead_slot_gating import compute_slot_mass
import train_art

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="40453")
ap.add_argument("--res", type=int, default=252)   # 18*14, small for speed
ap.add_argument("--steps", type=int, default=80)
args = ap.parse_args()
dev = "cuda"

ds = ArticulatedDataset(data_root="/data2/lza/partnet-Mobility/data_processed",
                        target_size=args.res, num_frames=4, max_parts=8, phase="1",
                        split="all", exclude_cams=set())
i = next(i for i in range(len(ds)) if ds.entries[i][0].name == args.scene)
batch = default_collate([ds[i]])
imgs = batch["images"].to(dev); extr = batch["extrinsics"].to(dev)
intr = batch["intrinsics"].to(dev); ts = batch["timestamps"].to(dev)

model = ArtVGGT(img_size=args.res, patch_size=14, embed_dim=1024, num_slots=8,
                n_gaussians=256, scene_radius=1.0, use_camera_head=False).to(dev)
model.set_phase("1a", warmup=True); model.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

cfg = argparse.Namespace(phase="1a", w_type=0.0, w_axis=0.0, w_pivot=0.0, w_scalar=0.0,
    w_dead_opacity=0.0, w_render=0.0, w_render_global=0.0, w_bbox=0.0, w_pose_enc=0.0,
    l1_sparsity=0.0, l1_sparsity_warmup=0.0, w_motion_mask=0.0, w_motion_track=0.0,
    motion_warmup_steps=1, sdf_beta=0.1, sdf_rays=64, sdf_frames=2, scene_radius=1.0)

gt0 = (batch["part_masks"][0, 0, 0] > 0.5).float().to(dev)   # base part mask, frame0
print(f"scene {args.scene}: base(gt0) px={int(gt0.sum())}/{args.res**2}")
print("step | mask  | slot0_mass_frac bg_frac | base_IoU")
for s in range(args.steps):
    opt.zero_grad()
    preds = model(imgs, extr, intr, ts)
    loss, ld = train_art.compute_loss(preds, batch, s, cfg, is_warmup=True, head=model.sdf_head)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if s % 10 == 0 or s == args.steps - 1:
        with torch.no_grad():
            am = preds["assign_maps"]; bg = preds.get("bg_map")
            mass = compute_slot_mass(am)[0]
            tot = am[0].sum() + (bg[0].sum() if bg is not None else 0) + 1e-6
            up = F.interpolate(am, (args.res, args.res), mode="bilinear", align_corners=False)
            src = torch.cat([up, F.interpolate(bg, (args.res, args.res), mode="bilinear",
                             align_corners=False)], 1) if bg is not None else up
            lab = src[0].argmax(0)
            s0 = (lab == 0)
            inter = (s0.float() * gt0).sum(); uni = ((s0.float() + gt0) > 0).float().sum()
            iou = (inter / (uni + 1e-6)).item()
            bgf = (bg[0].sum() / tot).item() if bg is not None else 0
            print(f"{s:4d} | {ld['mask'].item():.3f} | {mass[0].item()/tot.item():.3f}  {bgf:.3f} | {iou:.3f}")
print("DONE")
