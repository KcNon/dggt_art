"""Isolated renderer test: can SDF render fit ONE GT part silhouette? (validates render gradient)"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from datasets.articulated_dataset import ArticulatedDataset
from dggt.heads.hexaplane_sdf_head import HexaPlaneSDFHead
from dggt.render.sdf_volume import generate_rays, render_rays_static

dev="cuda"; IMG=126
ds=ArticulatedDataset(data_root="/data2/lza/partnet-Mobility/data_processed",
                      target_size=IMG,num_frames=2,max_parts=8,phase="1",split="all",exclude_cams=set())
idx=next(i for i in range(len(ds)) if "100202" in ds.entries[i][0].name)
b=default_collate([ds[idx]])
extr=b["extrinsics"][0].to(dev); intr=b["intrinsics"][0].to(dev)
# target = moving part slot 1 GT mask, frame 0
target=(b["part_masks"][0,0,1]>0.5).float().to(dev)   # [H,W]
print("target fg pixels:", int(target.sum()), "/", target.numel())

head=HexaPlaneSDFHead(dim_in=16,num_slots=1,plane_res=64,plane_ch=16,dim_patch=0).to(dev)
latent=torch.nn.Parameter(torch.randn(1,1,16,device=dev))   # learnable slot vec
center=torch.zeros(1,3,device=dev); size=torch.ones(1,3,device=dev)   # bbox covers [-1,1]^3
alive=torch.tensor([True],device=dev)
opt=torch.optim.Adam(list(head.parameters())+[latent],lr=5e-3)

ys,xs=torch.meshgrid(torch.arange(float(IMG)),torch.arange(float(IMG)),indexing="ij")
pix_all=torch.stack([xs.reshape(-1),ys.reshape(-1)],-1).to(dev)
tgt_flat=target.reshape(-1)

def iou(op):
    p=(op>0.5); g=(tgt_flat>0.5)
    return (p&g).sum().float()/((p|g).sum().float()+1e-6)

print("step | bce    | IoU")
for s in range(200):
    opt.zero_grad()
    # sample 2048 rays (half fg)
    fg=torch.nonzero(tgt_flat>0.5).squeeze(-1)
    sel=torch.cat([fg[torch.randint(fg.numel(),(1024,),device=dev)],
                   torch.randint(IMG*IMG,(1024,),device=dev)])
    planes=head.decode_planes(latent)[0]
    ro,rd=generate_rays(extr[0],intr,pix_all[sel])
    out=render_rays_static(head,planes,center,size,alive,ro,rd,beta=0.1,n_samples=48)
    op=out["opacity"][:,0].clamp(1e-5,1-1e-5)
    loss=F.binary_cross_entropy(op,tgt_flat[sel])
    loss.backward(); opt.step()
    if s%40==0 or s==199:
        with torch.no_grad():
            planes=head.decode_planes(latent)[0]
            ro,rd=generate_rays(extr[0],intr,pix_all)
            o=render_rays_static(head,planes,center,size,alive,ro,rd,beta=0.1,n_samples=48)["opacity"][:,0]
        print(f"{s:4d} | {loss.item():.4f} | {iou(o).item():.3f}")
print("RENDER OVERFIT DONE")
