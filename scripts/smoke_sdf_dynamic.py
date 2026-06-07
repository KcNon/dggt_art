"""Dynamic-render test: fit ONE moving part across frames via ray inverse-transform."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from datasets.articulated_dataset import ArticulatedDataset
from dggt.heads.hexaplane_sdf_head import HexaPlaneSDFHead
from dggt.render.sdf_volume import generate_rays, render_rays_static

dev="cuda"; IMG=126
ds=ArticulatedDataset(data_root="/data2/lza/partnet-Mobility/data_processed",
                      target_size=IMG,num_frames=8,max_parts=8,phase="1",split="all",exclude_cams=set())
idx=next(i for i in range(len(ds)) if "100202" in ds.entries[i][0].name)
b=default_collate([ds[idx]])
extr=b["extrinsics"][0].to(dev); intr=b["intrinsics"][0].to(dev)
S=b["images"].shape[1]
frames=[0, S//2, S-1]
# moving part = slot 1; GT axis/pivot from dataset
gt_axis=b["gt_axis"][0,1].to(dev); gt_pivot=b["gt_pivot"][0,1].to(dev)
masks=[(b["part_masks"][0,s,1]>0.5).float().to(dev) for s in frames]   # per-frame GT
print("part slot1 fg per frame:", [int(m.sum()) for m in masks])

head=HexaPlaneSDFHead(dim_in=16,num_slots=1,plane_res=64,plane_ch=16,dim_patch=0).to(dev)
latent=torch.nn.Parameter(torch.randn(1,1,16,device=dev))
# Use GT scalars (new angle/2π convention), FIXED — only the field is learned.
scal=b["gt_scalars"][0,1,frames].to(dev)
print("GT scalars @frames:", scal.cpu().numpy().round(3))
center=torch.zeros(1,3,device=dev); size=torch.ones(1,3,device=dev)
alive=torch.tensor([True],device=dev)
mprob=torch.tensor([[0,0,1.]],device=dev)  # revolute
opt=torch.optim.Adam(list(head.parameters())+[latent],lr=5e-3)

ys,xs=torch.meshgrid(torch.arange(float(IMG)),torch.arange(float(IMG)),indexing="ij")
pix_all=torch.stack([xs.reshape(-1),ys.reshape(-1)],-1).to(dev)

def iou(op,tgt):
    p=(op>0.5); g=(tgt.reshape(-1)>0.5)
    return (p&g).sum().float()/((p|g).sum().float()+1e-6)

print("step | loss   | IoU@f0  IoU@fmid IoU@flast | scal")
for it in range(250):
    opt.zero_grad()
    planes=head.decode_planes(latent)[0]
    loss=0
    for fi in range(len(frames)):
        tgt=masks[fi].reshape(-1)
        fg=torch.nonzero(tgt>0.5).squeeze(-1)
        sel=torch.cat([fg[torch.randint(fg.numel(),(800,),device=dev)],
                       torch.randint(IMG*IMG,(800,),device=dev)])
        ro,rd=generate_rays(extr[frames[fi]],intr,pix_all[sel])
        out=render_rays_static(head,planes,center,size,alive,ro,rd,beta=0.1,n_samples=48,
                               motion_probs=mprob,axis=gt_axis[None],pivot=gt_pivot[None],
                               scalar=scal[fi:fi+1],scene_radius=1.0)
        op=out["opacity"][:,0].clamp(1e-5,1-1e-5)
        loss=loss+F.binary_cross_entropy(op,tgt[sel])
    loss.backward(); opt.step()
    if it%50==0 or it==249:
        ious=[]
        with torch.no_grad():
            planes=head.decode_planes(latent)[0]
            for fi in range(len(frames)):
                ro,rd=generate_rays(extr[frames[fi]],intr,pix_all)
                o=render_rays_static(head,planes,center,size,alive,ro,rd,beta=0.1,n_samples=48,
                                     motion_probs=mprob,axis=gt_axis[None],pivot=gt_pivot[None],
                                     scalar=scal[fi:fi+1],scene_radius=1.0)["opacity"][:,0]
                ious.append(iou(o,masks[fi]).item())
        print(f"{it:4d} | {loss.item():.4f} | {ious[0]:.3f}    {ious[1]:.3f}    {ious[2]:.3f}    | {scal.detach().cpu().numpy().round(2)}")
print("DYNAMIC RENDER TEST DONE")
