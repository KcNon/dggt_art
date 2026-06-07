"""
Single-scene multi-view per-part SDF overfit (paper-style losses).

Renders ALL parts COMPOSITED (occlusion-aware) and supervises each part's
occlusion-aware opacity vs its (modal) GT mask + composite RGB — matching
train_art.sdf_render_loss. (Rendering parts in isolation vs modal GT masks
wrongly penalises occluded regions and tanks IoU on self-occluding parts.)

Losses per step, over K (view,stage) pairs with balanced ray sampling:
  • per-part mask L2 (fg-weighted): occlusion-aware part_opacity vs GT part mask
  • composite RGB L2 on foreground   + LPIPS (perceptual, dense, periodic)
  • eikonal (finite-difference SDF grad → clean distance field)
Coarse-to-fine: 1/β linearly annealed (soft → sharp). Articulation = GT.
"""
import os, sys, math, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data._utils.collate import default_collate
from datasets.articulated_dataset import ArticulatedDataset
from dggt.heads.hexaplane_sdf_head import HexaPlaneSDFHead
from dggt.render.sdf_volume import generate_rays, render_rays_static
import lpips as lpips_lib

dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="100202")
ap.add_argument("--res", type=int, default=128)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--out", default="/data5/lza/checkpoint/Art/overfit")
ap.add_argument("--lpips", action="store_true")
ap.add_argument("--w_lpips", type=float, default=0.5)
ap.add_argument("--kvs", type=int, default=4)
ap.add_argument("--rays", type=int, default=2048)
ap.add_argument("--w_eik", type=float, default=0.1)
ap.add_argument("--w_depth", type=float, default=1.0, help="GT depth supervision (key for untextured concave objects)")
ap.add_argument("--geom_only", action="store_true", help="geometry only: mask L2 + depth L1 + eikonal, NO RGB/LPIPS")
ap.add_argument("--only_part", type=int, default=-1, help="isolate a single moving part by its index in the moving list (diagnostic)")
ap.add_argument("--no_base", action="store_true", help="exclude the static base part (slot 0); render movable parts only (legacy diagnostic)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = args.res

ds = ArticulatedDataset(data_root="/data2/lza/partnet-Mobility/data_processed",
                        target_size=R, num_frames=8, max_parts=8, phase="1",
                        split="all", exclude_cams=set())
ents = [i for i in range(len(ds)) if ds.entries[i][0].name == args.scene]
assert len(ents) > 0, f"scene {args.scene} not found"
views = [ds[i] for i in ents]
V = len(views); S = views[0]["images"].shape[0]
n_active = int(views[0]["n_active_parts"][0]) if torch.is_tensor(views[0]["n_active_parts"]) else int(views[0]["n_active_parts"])
moving = list(range(1, n_active))
if args.only_part >= 0:
    assert args.only_part < len(moving), f"only_part {args.only_part} out of range (moving has {len(moving)})"
    moving = [moving[args.only_part]]
# parts = base (slot 0, pure static body — paper-aligned) + movable parts.
# Rendering the big solid base supplies the strongest geometric signal.
parts = ([0] + moving) if not args.no_base else list(moving)
print(f"scene {args.scene}: views={V} stages={S} parts={parts} (base={'yes' if 0 in parts else 'no'})")

imgs  = torch.stack([v["images"] for v in views]).to(dev)
extr  = torch.stack([v["extrinsics"] for v in views]).to(dev)
intr  = torch.stack([v["intrinsics"] for v in views]).to(dev)
pmask = torch.stack([(v["part_masks"] > 0.5).float() for v in views]).to(dev)
depths = torch.stack([v["depth"] for v in views]).to(dev)        # [V,S,R,R] camera z-depth
gt_axis  = views[0]["gt_axis"].to(dev)
gt_pivot = views[0]["gt_pivot"].to(dev)
gt_scal  = views[0]["gt_scalars"].to(dev)
gt_mtype = views[0]["gt_motion_type"].to(dev)
mprob_full = torch.zeros(8, 3, device=dev)
for p in range(8):
    if gt_mtype[p] > 0:
        mprob_full[p, int(gt_mtype[p])] = 1.0

P = len(parts)
# Per-part articulation arrays (aligned with planes/bbox order). The base part
# (index 0) has mprob_full[0]=[0,0,0] → static (no ray transform), scalar 0.
mprob_m = mprob_full[parts]            # [P,3]
axis_m  = gt_axis[parts]               # [P,3]
pivot_m = gt_pivot[parts]             # [P,3]
scal_m  = gt_scal[parts]               # [P,S]
aliveP  = torch.ones(P, dtype=torch.bool, device=dev)

head = HexaPlaneSDFHead(dim_in=48, num_slots=P, plane_res=64, plane_ch=32, dim_patch=0).to(dev)
latent = torch.nn.Parameter(torch.randn(1, P, 48, device=dev) * 0.1)
bb_c = torch.nn.Parameter(torch.zeros(P, 3, device=dev))
bb_s = torch.nn.Parameter(torch.full((P, 3), 0.9, device=dev))
opt = torch.optim.Adam(list(head.parameters()) + [latent, bb_c, bb_s], lr=3e-3)
perc = None
if args.lpips:
    perc = lpips_lib.LPIPS(net="vgg").to(dev); perc.eval()
    for q in perc.parameters(): q.requires_grad_(False)
    print("LPIPS enabled")

ys, xs = torch.meshgrid(torch.arange(float(R)), torch.arange(float(R)), indexing="ij")
pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1).to(dev)
allidx = torch.arange(R * R, device=dev)

def beta_at(step):
    f = step / max(args.steps - 1, 1)
    inv = (1/0.3) + f * ((1/0.02) - (1/0.3))
    return 1.0 / inv

def render_all(planes, v, t, beta, sel):
    """Composite render of ALL moving parts (occlusion-aware)."""
    ro, rd = generate_rays(extr[v, t], intr[v], pix[sel])
    return render_rays_static(
        head, planes, bb_c, bb_s.clamp(min=0.05), aliveP, ro, rd,
        beta=beta, n_samples=64,
        motion_probs=mprob_m, axis=axis_m, pivot=pivot_m, scalar=scal_m[:, t],
        scene_radius=1.0)

def sample_rays(v, t, n, fg_frac=0.5):
    union = pmask[v, t, parts].sum(0).clamp(0, 1).reshape(-1)
    fg = torch.nonzero(union > 0.5, as_tuple=False).squeeze(-1)
    n_fg = min(int(n * fg_frac), fg.numel())
    chunks = []
    if n_fg > 0:
        chunks.append(fg[torch.randint(fg.numel(), (n_fg,), device=dev)])
    chunks.append(torch.randint(R * R, (n - n_fg,), device=dev))
    return torch.cat(chunks)

def eikonal(planes, n=512, eps=0.01):
    tot = torch.zeros((), device=dev)
    for pl in range(P):
        x = torch.rand(n, 3, device=dev) * 2 - 1
        comps = []
        for d in range(3):
            off = torch.zeros_like(x); off[:, d] = eps
            sp, _ = head.query(planes[pl], x + off)
            sm, _ = head.query(planes[pl], x - off)
            comps.append((sp - sm) / (2 * eps))
        grad = torch.cat(comps, dim=-1)
        tot = tot + ((grad.norm(dim=-1) - 1) ** 2).mean()
    return tot / P

print("step | loss   | mIoU  | PSNR  | eik   | beta")
for it in range(args.steps):
    opt.zero_grad()
    beta = beta_at(it)
    planes = head.decode_planes(latent)[0]
    loss = torch.zeros((), device=dev)
    for _ in range(args.kvs):
        v = torch.randint(V, (1,)).item(); t = torch.randint(S, (1,)).item()
        sel = sample_rays(v, t, args.rays)
        out = render_all(planes, v, t, beta, sel)
        pop = out["part_opacity"].clamp(1e-4, 1-1e-4)         # [N,P] occlusion-aware
        gti = imgs[v, t].reshape(3, -1).transpose(0, 1)[sel]
        union = pmask[v, t, parts].sum(0).clamp(0, 1).reshape(-1)[sel]
        for p in range(P):
            gtm = pmask[v, t, parts[p]].reshape(-1)[sel]
            w_px = torch.where(gtm > 0.5, 10.0, 1.0)
            loss = loss + (w_px * (pop[:, p] - gtm) ** 2).mean()
        fg = union > 0.5
        if fg.any() and not args.geom_only:
            loss = loss + F.mse_loss(out["rgb"][fg], gti[fg])
        # depth: render expected ray-distance vs GT (camera z-depth → ray dist)
        if args.w_depth > 0:
            u = pix[sel][:, 0]; vv = pix[sel][:, 1]; K = intr[v]
            raylen = torch.sqrt(((u-K[0,2])/K[0,0])**2 + ((vv-K[1,2])/K[1,1])**2 + 1)
            t_gt = depths[v, t].reshape(-1)[sel] * raylen
            dm = fg & (t_gt > 0)
            if dm.any():
                loss = loss + args.w_depth * (out["depth"][:, 0][dm] - t_gt[dm]).abs().mean()
    loss = loss / args.kvs
    loss = loss + args.w_eik * eikonal(planes)
    if perc is not None and not args.geom_only and it % 4 == 0:
        v = torch.randint(V, (1,)).item(); t = torch.randint(S, (1,)).item()
        out = render_all(planes, v, t, beta, allidx)
        union = pmask[v, t, parts].sum(0).clamp(0, 1).reshape(-1)
        ri = (out["rgb"] * union[:, None]).reshape(R, R, 3).permute(2, 0, 1)[None] * 2 - 1
        gti = imgs[v, t].reshape(3, -1).transpose(0, 1)
        gi = (gti * union[:, None]).reshape(R, R, 3).permute(2, 0, 1)[None] * 2 - 1
        loss = loss + args.w_lpips * perc(ri, gi).mean()
    loss.backward(); opt.step()

    if it % 100 == 0 or it == args.steps - 1:
        with torch.no_grad():
            leik = eikonal(planes)
            ious, psnrs = [], []
            for v2 in range(V):
                for t2 in [0, S//2, S-1]:                    # avg over stages
                    out = render_all(planes, v2, t2, beta, allidx)
                    pop = out["part_opacity"]
                    for p in range(P):
                        gtm = pmask[v2, t2, parts[p]].reshape(-1)
                        if gtm.sum() < 10:                    # skip near-empty (fully occluded)
                            continue
                        inter = ((pop[:, p] > 0.5) & (gtm > 0.5)).sum().float()
                        uni = ((pop[:, p] > 0.5) | (gtm > 0.5)).sum().float()
                        ious.append((inter/(uni+1e-6)).item())
                    union = pmask[v2, t2, parts].sum(0).clamp(0, 1).reshape(-1)
                    fg = union > 0.5
                    if fg.any():
                        gti = imgs[v2, t2].reshape(3, -1).transpose(0, 1)
                        mse = F.mse_loss(out["rgb"][fg], gti[fg])
                        psnrs.append((-10*torch.log10(mse+1e-8)).item())
        print(f"{it:4d} | {loss.item():.4f} | {np.mean(ious):.3f} | {np.mean(psnrs):5.2f} | {leik.item():.3f} | {beta:.3f}")

# Dump reconstruction (view 0, stages 0/mid/last)
with torch.no_grad():
    planes = head.decode_planes(latent)[0]
    rows = []
    for t in [0, S//2, S-1]:
        out = render_all(planes, 0, t, beta_at(args.steps-1), allidx)
        if args.geom_only:
            # [render opacity | GT union mask | render depth | GT depth]  (grayscale)
            ro_op = out["opacity"].reshape(R, R, 1).repeat(1, 1, 3)
            gt_un = pmask[0, t, parts].sum(0).clamp(0, 1).reshape(R, R, 1).repeat(1, 1, 3)
            u = pix[:, 0]; vv = pix[:, 1]; K = intr[0]
            raylen = torch.sqrt(((u-K[0,2])/K[0,0])**2 + ((vv-K[1,2])/K[1,1])**2 + 1)
            dmax = depths[0, t].max().clamp(min=1e-3)
            ro_d = (out["depth"].reshape(-1) / raylen / dmax).reshape(R, R, 1).repeat(1, 1, 3)
            gt_d = (depths[0, t] / dmax).reshape(R, R, 1).repeat(1, 1, 3)
            rows.append(torch.cat([ro_op, gt_un, ro_d, gt_d], 1))
        else:
            recon = out["rgb"].reshape(R, R, 3)
            gt = imgs[0, t].reshape(3, -1).transpose(0, 1).reshape(R, R, 3)
            rows.append(torch.cat([recon, gt], 1))
    grid = torch.cat(rows, 0)
    Image.fromarray((grid.clamp(0,1).cpu().numpy()*255).astype("uint8")).save(f"{args.out}/recon_{args.scene}.png")
    label = "渲染轮廓|GT mask|渲染深度|GT深度" if args.geom_only else "重建|GT"
    print("saved", f"{args.out}/recon_{args.scene}.png  ({label}, 行=stage 0/mid/last)")
print("OVERFIT DONE")
