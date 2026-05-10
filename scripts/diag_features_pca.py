"""
Diagnostic: PCA-visualize DINOv2 patch tokens vs Aggregator output tokens
on sim vs real scenes, to localize where the feature distribution breaks.

Output: a grid image per scene with [RGB | DINO PCA | AGG PCA] for frame 0.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dggt.models.art_vggt import ArtVGGT
from datasets.itaco_real_dataset import iTACORealDataset
from datasets.articulated_dataset import ArticulatedDataset


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = ckpt.get("cfg", {})
    model = ArtVGGT(
        img_size=518, patch_size=14, embed_dim=1024, num_slots=8,
        n_gaussians=saved_cfg.get("n_gaussians", 256),
        scene_radius=saved_cfg.get("scene_radius", 1.0),
        use_camera_head=True, stop_gradient_plucker=False,
        gradient_checkpointing=False,
        use_track_tokens=saved_cfg.get("use_track_tokens", False),
        num_frames_track=saved_cfg.get("num_frames", 8),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"[load] step={ckpt.get('step','?')}")
    return model


def pca_to_rgb(feat_2d, H_p, W_p):
    """feat_2d: [N_p, D] numpy. Returns [H_p, W_p, 3] uint8."""
    f = feat_2d - feat_2d.mean(axis=0, keepdims=True)
    # Top-3 PCs via SVD
    U, S, Vt = np.linalg.svd(f, full_matrices=False)
    pc = f @ Vt[:3].T            # [N, 3]
    # Normalize each channel to [0, 1]
    lo = np.percentile(pc, 2, axis=0)
    hi = np.percentile(pc, 98, axis=0)
    pc = np.clip((pc - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    img = (pc.reshape(H_p, W_p, 3) * 255).astype(np.uint8)
    return img


@torch.no_grad()
def run_one(model, batch, device, tag, out_root, idx):
    images = batch["images"].to(device)        # [1, S, 3, H, W]
    B, S, _, H, W = images.shape
    H_p, W_p = H // 14, W // 14

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        agg_list, _agg_with_tokens, dino_list, _img_feat, patch_start_idx, _slots = model.aggregator(images)

    image_tokens = agg_list[-1].float()        # [1, S, P_total, 2C]
    dino_tokens  = dino_list[-1].float()       # [1, S, P_total, C]

    # Per-layer collapse trace (frame 0, patch tokens only)
    layer_stds = []
    for li, lt in enumerate(agg_list):
        ft = lt[0, 0, patch_start_idx:].float()       # [N_p, 2C]
        layer_stds.append(float(torch.linalg.norm(ft, dim=1).std().item()))
    print(f"  per-layer norm-std: {[f'{s:.2f}' for s in layer_stds]}")

    agg0  = image_tokens[0, 0, patch_start_idx:].cpu().numpy()    # [N_p, 2C]
    dino0 = dino_tokens [0, 0, patch_start_idx:].cpu().numpy()    # [N_p, C]

    rgb = images[0, 0].permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    dino_pca = pca_to_rgb(dino0, H_p, W_p)
    agg_pca  = pca_to_rgb(agg0,  H_p, W_p)

    # Upsample PCA images to RGB resolution
    dino_pca = np.array(Image.fromarray(dino_pca).resize((W, H), Image.NEAREST))
    agg_pca  = np.array(Image.fromarray(agg_pca ).resize((W, H), Image.NEAREST))

    canvas = np.concatenate([rgb, dino_pca, agg_pca], axis=1)
    out = out_root / f"{tag}_{idx:03d}.png"
    Image.fromarray(canvas).save(out)
    print(f"  saved {out}")

    # Stats: norm of features, to detect distribution shift
    print(f"  dino norm  mean={np.linalg.norm(dino0,axis=1).mean():.3f}  std={np.linalg.norm(dino0,axis=1).std():.3f}")
    print(f"  agg  norm  mean={np.linalg.norm(agg0 ,axis=1).mean():.3f}  std={np.linalg.norm(agg0 ,axis=1).std():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sim_root",  default="/data2/cyt/data_root_refine")
    p.add_argument("--real_root", default="/data2/cyt/video2articulation/real_data")
    p.add_argument("--output_dir", default="/data2/cyt/eval/diag_pca")
    p.add_argument("--n_each", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda")
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, device)

    real_ds = iTACORealDataset(
        data_root=args.real_root, target_size=518, num_frames=args.num_frames,
        max_parts=8, max_tracks=1024, split="all", random_start=False,
    )
    sim_ds = ArticulatedDataset(
        data_root=args.sim_root, target_size=518, num_frames=args.num_frames,
        max_parts=8, phase="1", split="val", val_ratio=0.05,
        motion_cache_name="motion_cache_gt.npz", max_tracks=1024,
    )

    print(f"[real] {len(real_ds)} scenes  | [sim] {len(sim_ds)} scenes")

    real_loader = DataLoader(real_ds, batch_size=1, shuffle=False, num_workers=0)
    sim_loader  = DataLoader(sim_ds,  batch_size=1, shuffle=False, num_workers=0)

    print("\n=== REAL ===")
    for i, batch in enumerate(real_loader):
        if i >= args.n_each: break
        print(f"[real {i}]")
        run_one(model, batch, device, "real", out_root, i)
        torch.cuda.empty_cache()

    print("\n=== SIM ===")
    for i, batch in enumerate(sim_loader):
        if i >= args.n_each: break
        print(f"[sim {i}]")
        run_one(model, batch, device, "sim", out_root, i)
        torch.cuda.empty_cache()

    print(f"\nDone → {out_root}")


if __name__ == "__main__":
    main()
