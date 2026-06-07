"""Decisive representation test: fit ONE static frame (stage 0) of a real scene
with a SINGLE hexa-plane VolSDF field over the full AABB, from all views.
No articulation, no per-part, no occlusion — pure "can the representation +
renderer reconstruct my data". Supervises full-image silhouette + depth + RGB."""
import os, sys, argparse, torch, numpy as np
sys.path.insert(0, "/home/ziang/code/dggt_art")
import torch.nn.functional as F
from PIL import Image
from datasets.articulated_dataset import ArticulatedDataset
from dggt.heads.hexaplane_sdf_head import HexaPlaneSDFHead
from dggt.render.sdf_volume import generate_rays, render_rays_static

dev="cuda"
ap=argparse.ArgumentParser()
ap.add_argument("--scene", default="40453")
ap.add_argument("--res", type=int, default=128)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--stage", type=int, default=0)
ap.add_argument("--rays", type=int, default=4096)
ap.add_argument("--w_depth", type=float, default=1.0)
ap.add_argument("--w_eik", type=float, default=0.1)
ap.add_argument("--out", default="/data5/lza/checkpoint/Art/overfit_static")
args=ap.parse_args()
os.makedirs(args.out, exist_ok=True); R=args.res

ds=ArticulatedDataset(data_root="/data2/lza/partnet-Mobility/data_processed",
                      target_size=R,num_frames=8,max_parts=8,phase="1",split="all",exclude_cams=set())
ents=[i for i in range(len(ds)) if ds.entries[i][0].name==args.scene]
views=[ds[i] for i in ents]; V=len(views); t=args.stage
print(f"scene {args.scene}: views={V} stage={t}")
imgs =torch.stack([v["images"] for v in views]).to(dev)        # [V,S,3,R,R]
extr =torch.stack([v["extrinsics"] for v in views]).to(dev)
intr =torch.stack([v["intrinsics"] for v in views]).to(dev)
pmask=torch.stack([(v["part_masks"]>0.5).float() for v in views]).to(dev)  # [V,S,P,R,R]
depths=torch.stack([v["depth"] for v in views]).to(dev)        # [V,S,R,R]
na=int(views[0]["n_active_parts"][0]) if torch.is_tensor(views[0]["n_active_parts"]) else int(views[0]["n_active_parts"])
# full-object silhouette = ANY active part (static body + all moving)
sil=pmask[:,t,:na].sum(1).clamp(0,1)                            # [V,R,R]
print("silhouette area per view:", [int(sil[v].sum()) for v in range(V)])

head=HexaPlaneSDFHead(dim_in=48,num_slots=1,plane_res=64,plane_ch=32,dim_patch=0).to(dev)
latent=torch.nn.Parameter(torch.randn(1,1,48,device=dev)*0.1)
center=torch.zeros(1,3,device=dev); size=torch.ones(1,3,device=dev)  # full AABB [-1,1]^3
alive=torch.ones(1,dtype=torch.bool,device=dev)
opt=torch.optim.Adam(list(head.parameters())+[latent],lr=3e-3)

ys,xs=torch.meshgrid(torch.arange(float(R)),torch.arange(float(R)),indexing="ij")
pix=torch.stack([xs.reshape(-1),ys.reshape(-1)],-1).to(dev)
allidx=torch.arange(R*R,device=dev)
def beta_at(s):
    f=s/max(args.steps-1,1); inv=(1/0.3)+f*((1/0.02)-(1/0.3)); return 1.0/inv
def eik(planes,n=512,eps=0.01):
    x=torch.rand(n,3,device=dev)*2-1; comps=[]
    for d in range(3):
        o=torch.zeros_like(x);o[:,d]=eps
        sp,_=head.query(planes[0],x+o); sm,_=head.query(planes[0],x-o)
        comps.append((sp-sm)/(2*eps))
    g=torch.cat(comps,-1); return ((g.norm(dim=-1)-1)**2).mean()

print("step | loss   | IoU   | PSNR  | dep   | beta")
for it in range(args.steps):
    opt.zero_grad(); beta=beta_at(it)
    planes=head.decode_planes(latent)[0]
    v=torch.randint(V,(1,)).item()
    s=sil[v].reshape(-1); fg=torch.nonzero(s>0.5).squeeze(-1)
    nfg=min(args.rays//2,fg.numel())
    sel=torch.cat([fg[torch.randint(fg.numel(),(nfg,),device=dev)],
                   torch.randint(R*R,(args.rays-nfg,),device=dev)])
    ro,rd=generate_rays(extr[v,t],intr[v],pix[sel])
    out=render_rays_static(head,planes,center,size,alive,ro,rd,beta=beta,n_samples=96,scene_radius=1.0)
    op=out["opacity"][:,0].clamp(1e-4,1-1e-4)
    tgt=s[sel]
    w=torch.where(tgt>0.5,5.0,1.0)
    loss=(w*(op-tgt)**2).mean()
    fgm=tgt>0.5
    gti=imgs[v,t].reshape(3,-1).transpose(0,1)[sel]
    if fgm.any():
        loss=loss+F.mse_loss(out["rgb"][fgm],gti[fgm])
    if args.w_depth>0:
        u=pix[sel][:,0];vv=pix[sel][:,1];K=intr[v]
        rl=torch.sqrt(((u-K[0,2])/K[0,0])**2+((vv-K[1,2])/K[1,1])**2+1)
        tg=depths[v,t].reshape(-1)[sel]*rl; dm=fgm&(tg>0)
        if dm.any(): loss=loss+args.w_depth*(out["depth"][:,0][dm]-tg[dm]).abs().mean()
    loss=loss+args.w_eik*eik(planes)
    loss.backward(); opt.step()
    if it%100==0 or it==args.steps-1:
        with torch.no_grad():
            ious,psnrs=[],[]
            for v2 in range(V):
                ro,rd=generate_rays(extr[v2,t],intr[v2],pix)
                o=render_rays_static(head,planes,center,size,alive,ro,rd,beta=beta,n_samples=96,scene_radius=1.0)
                op=o["opacity"][:,0]; g=sil[v2].reshape(-1)
                inter=((op>0.5)&(g>0.5)).sum().float();uni=((op>0.5)|(g>0.5)).sum().float()
                ious.append((inter/(uni+1e-6)).item())
                fgm=g>0.5
                if fgm.any():
                    gi=imgs[v2,t].reshape(3,-1).transpose(0,1)
                    mse=F.mse_loss(o["rgb"][fgm],gi[fgm]); psnrs.append((-10*torch.log10(mse+1e-8)).item())
        print(f"{it:4d} | {loss.item():.4f} | {np.mean(ious):.3f} | {np.mean(psnrs):5.2f} | {beta:.3f}")

with torch.no_grad():
    planes=head.decode_planes(latent)[0]; rows=[]
    for v2 in range(V):
        ro,rd=generate_rays(extr[v2,t],intr[v2],pix)
        o=render_rays_static(head,planes,center,size,alive,ro,rd,beta=beta_at(args.steps-1),n_samples=96,scene_radius=1.0)
        rr=o["rgb"].reshape(R,R,3).clamp(0,1)
        gg=imgs[v2,t].permute(1,2,0)
        dmax=depths[v2,t].max().clamp(min=1e-3)
        K=intr[v2];u=pix[:,0];vv=pix[:,1]
        rl=torch.sqrt(((u-K[0,2])/K[0,0])**2+((vv-K[1,2])/K[1,1])**2+1)
        rd_=(o["depth"][:,0]/rl/dmax).reshape(R,R,1).repeat(1,1,3)
        gd=(depths[v2,t]/dmax).reshape(R,R,1).repeat(1,1,3)
        rows.append(torch.cat([rr,gg,rd_.clamp(0,1),gd.clamp(0,1)],1).cpu())
    grid=torch.cat(rows,0).numpy()
    Image.fromarray((grid*255).astype(np.uint8)).save(f"{args.out}/static_{args.scene}.png")
    print(f"saved {args.out}/static_{args.scene}.png  [render rgb|GT rgb|render depth|GT depth], rows=views")
print("STATIC OVERFIT DONE")
