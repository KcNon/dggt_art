"""Phase-1 SDF pipeline smoke test: overfit one scene, check loss drops, dump a render."""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from dggt.models.art_vggt import ArtVGGT
from datasets.articulated_dataset import ArticulatedDataset
import train_art
from dggt.render.sdf_volume import generate_rays, render_rays_static

dev = "cuda"
IMG = 126   # 9*14, small for speed
ds = ArticulatedDataset(
    data_root="/data2/lza/partnet-Mobility/data_processed",
    target_size=IMG, num_frames=2, max_parts=8, phase="1",
    split="all", exclude_cams=set(),
)
# pick a multi-joint scene
idx = next(i for i in range(len(ds)) if "100202" in ds.entries[i][0].name)
batch = default_collate([ds[idx]])
print("scene:", batch["scene_id"], "n_active:", int(batch["n_active_parts"][0]))

model = ArtVGGT(img_size=IMG, patch_size=14, embed_dim=1024, num_slots=8,
                n_gaussians=8, scene_radius=1.0, use_camera_head=False).to(dev)
model.set_phase("1b", warmup=False)
model.train()

cfg = argparse.Namespace(
    phase="1b", w_type=0.5, w_axis=0.5, w_pivot=0.1, w_scalar=0.3,
    w_dead_opacity=0.0, w_render=1.0, w_render_global=0.0, w_bbox=0.5,
    w_pose_enc=0.0, l1_sparsity=0.01, l1_sparsity_warmup=0.0,
    w_motion_mask=0.0, w_motion_track=0.0, motion_warmup_steps=1,
    sdf_beta=0.1, sdf_rays=256, sdf_frames=2, scene_radius=1.0,
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

images=batch["images"].to(dev); extr=batch["extrinsics"].to(dev)
intr=batch["intrinsics"].to(dev); ts=batch["timestamps"].to(dev)

print("step | total   render  mask    bbox   | finite")
for s in range(25):
    opt.zero_grad()
    preds = model(images, extr, intr, ts)
    loss, ld = train_art.compute_loss(preds, batch, s, cfg, is_warmup=False, head=model.sdf_head)
    fin = torch.isfinite(loss).item()
    if not fin:
        print(f"{s:4d} | NON-FINITE loss, abort"); sys.exit(1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if s % 2 == 0 or s == 24:
        print(f"{s:4d} | {ld['total'].item():.4f}  {ld['render'].item():.4f}  "
              f"{ld['mask'].item():.4f}  {ld['bbox'].item():.4f} | {fin}")

# Dump a render of the moving parts at frame 0
model.eval()
with torch.no_grad():
    preds = model(images, extr, intr, ts)
    planes = preds["planes"][0].float()
    bc, bs = preds["bbox_center"][0].float(), preds["bbox_size"][0].float()
    from dggt.utils.dead_slot_gating import detect_dead_slots
    is_dead = detect_dead_slots(preds["assign_maps"])[0]
    alive = ~is_dead
    c2w = extr[0,0].float(); K = intr[0].float()
    ys,xs = torch.meshgrid(torch.arange(float(IMG)),torch.arange(float(IMG)),indexing="ij")
    pix = torch.stack([xs.reshape(-1),ys.reshape(-1)],-1).to(dev)
    ro,rd = generate_rays(c2w,K,pix)
    out = render_rays_static(model.sdf_head, planes, bc, bs, alive, ro, rd, beta=0.1, n_samples=48)
    op = out["opacity"].reshape(IMG,IMG)
    print("render opacity: min %.3f max %.3f  fg %d/%d"%(op.min(),op.max(),(op>0.5).sum(),op.numel()))
    print("rgb finite:", torch.isfinite(out["rgb"]).all().item())
print("SMOKE OK")
