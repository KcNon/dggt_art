"""
ArtVGGT — Feedforward Articulated Scene Transformer.

Wires together (per plan §一, §四):
  Aggregator        (unchanged VGGT encoder)
  CameraHead        (unchanged, optional)
  PartSlotRouter    (new Decoder)
  ArticulationHead  (unified kinematic + dynamics head)
  ArtGaussianHead   (new)

Forward inputs:
  images      [B, S, 3, H, W]
  extrinsics  [B, S, 4, 4]   cam-to-world (or None → CameraHead predicts)
  intrinsics  [B, 3, 3]      shared across frames
  timestamps  [B, S]         normalised to [0, 1]

Forward outputs (dict):
  slot_features       [B, P, D]
  assign_maps         [B, P, H_p, W_p]
  motion_type_logits  [B, P, 2]
  axis                [B, P, 3]
  pivot               [B, P, 3]
  scalars             [B, P, S]
  gs_mu               [B, P, N_g, 3]
  gs_rot              [B, P, N_g, 4]
  gs_scale            [B, P, N_g, 3]
  gs_color            [B, P, N_g, 3]
  gs_opacity          [B, P, N_g, 1]
  pose_enc            [B, S, 9]  (if CameraHead enabled and no GT extrinsics)
  plucker_rays        [B, S, N_patches, 6]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dggt.models.aggregator import Aggregator
from dggt.heads.camera_head import CameraHead
from dggt.heads.part_slot_router import PartSlotRouter
from dggt.heads.articulation_head import ArticulationHead
from dggt.heads.art_gaussian_head import ArtGaussianHead
from dggt.heads.track_encoder import TrackEncoder
from dggt.utils.plucker import compute_plucker_rays_patch, plucker_stop_gradient


class ArtVGGT(nn.Module):
    """
    Articulated-scene version of VGGT.

    Args:
        img_size:       input image size (square assumed)
        patch_size:     DINOv2 patch size
        embed_dim:      Aggregator embedding dimension
        num_slots:      number of part slots (P, default 8)
        n_gaussians:    canonical Gaussians per slot (default 256)
        scene_radius:   canonical bbox half-side for position / pivot decoding
        use_camera_head: whether to run CameraHead for pose estimation
        stop_gradient_plucker: whether to detach Plücker rays from CameraHead
                               (set True for Phase 2 when pose is predicted)
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_slots: int = 8,
        n_gaussians: int = 256,
        scene_radius: float = 1.0,
        use_camera_head: bool = True,
        stop_gradient_plucker: bool = False,
        gradient_checkpointing: bool = False,
        use_track_tokens: bool = False,
        num_frames_track: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_slots  = num_slots
        self.stop_gradient_plucker = stop_gradient_plucker
        self._gradient_checkpointing = gradient_checkpointing
        self.use_track_tokens = use_track_tokens

        dim_agg = 2 * embed_dim     # Aggregator produces frame+global concat → 2×D

        # ── Encoder ────────────────────────────────────────────────────────
        # num_slots passed so Aggregator initialises learnable slot tokens that
        # participate in every global attention block (cross-frame).
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_slots=num_slots,
        )
        self.patch_start_idx = self.aggregator.patch_start_idx  # = 5
        if gradient_checkpointing:
            self.aggregator.set_gradient_checkpointing(True)

        # ── Optional pose estimator ────────────────────────────────────────
        self.camera_head = CameraHead(dim_in=dim_agg) if use_camera_head else None

        # ── Articulation decoder stack ─────────────────────────────────────
        self.part_slot_router = PartSlotRouter(
            dim_agg=dim_agg,
            dim_dino=embed_dim,      # DINOv2 last-layer tokens, same dim as embed_dim
            dim_slot=embed_dim,
            num_slots=num_slots,
            patch_start_idx=self.patch_start_idx,
            patch_size=patch_size,
            use_track_tokens=use_track_tokens,
            dim_track=embed_dim,
        )

        # ── Track encoder (optional) ───────────────────────────────────────
        if use_track_tokens:
            self.track_encoder = TrackEncoder(
                num_frames=num_frames_track,
                img_size=img_size,
                dim_inner=512,
                dim_out=embed_dim,
                num_freq_bands=6,            # Fourier lift on xy
                dim_vis_feat=embed_dim,      # frame-0 DINOv2 sample as anchor
            )
        else:
            self.track_encoder = None

        self.articulation_head = ArticulationHead(
            dim_in=embed_dim,
            num_slots=num_slots,
            scene_radius=scene_radius,
        )

        self.gaussian_head = ArtGaussianHead(
            dim_in=embed_dim,
            num_slots=num_slots,
            n_gaussians=n_gaussians,
            scene_radius=scene_radius,
            dim_patch=3 * embed_dim,   # agg_last(2C) + dino_last(C) = 3C
            dim_proj=256,
        )

    # ------------------------------------------------------------------
    # Helper: decode pose encoding → extrinsics (cam-to-world 4×4)
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_enc_to_extrinsics(
        pose_enc: torch.Tensor,    # [B, S, 9]
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Convert CameraHead pose encoding to cam-to-world 4×4 extrinsics.

        pose_enc format: absT(3) | quaR(4) | FoV(2)
        pose_encoding_to_extri_intri returns cam-from-world [B*S, 3, 4].
        We invert to get cam-to-world [B, S, 4, 4].
        """
        from dggt.utils.pose_enc import pose_encoding_to_extri_intri
        from dggt.utils.geometry import closed_form_inverse_se3
        B, S, _ = pose_enc.shape

        # Returns world-to-cam [B*S, 3, 4] in OpenCV convention
        w2c_34, _ = pose_encoding_to_extri_intri(
            pose_enc.reshape(B * S, 1, 9),
            image_size_hw=image_size,
        )
        # Pad to 4×4
        bottom = torch.tensor([[0, 0, 0, 1]], dtype=w2c_34.dtype, device=w2c_34.device)
        bottom = bottom.unsqueeze(0).expand(B * S, -1, -1)   # [B*S, 1, 4]
        w2c_44 = torch.cat([w2c_34.reshape(B * S, 3, 4), bottom], dim=1)  # [B*S, 4, 4]

        # Invert: cam-to-world
        c2w_44 = closed_form_inverse_se3(w2c_44)   # [B*S, 4, 4]
        return c2w_44.reshape(B, S, 4, 4)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,                  # [B, S, 3, H, W]
        extrinsics: torch.Tensor | None,       # [B, S, 4, 4] or None
        intrinsics: torch.Tensor,              # [B, 3, 3]
        timestamps: torch.Tensor,              # [B, S]
        tracks_2d:  torch.Tensor | None = None,   # [B, S, N_t, 2] pixel coords
        tracks_vis: torch.Tensor | None = None,   # [B, S, N_t]
        disable_tracks: bool = False,             # val-time ablation: same ckpt, no track injection
    ) -> dict:

        if images.dim() == 4:
            images = images.unsqueeze(0)

        B, S, C, H, W = images.shape

        # ── 1. Aggregator (encoder) ────────────────────────────────────────
        (agg_tokens_list,
         img_tokens_list,
         dino_tokens_list,
         image_feature,
         patch_start_idx,
         slot_states) = self.aggregator(images)
        # slot_states: [B, P, embed_dim] — slot tokens refined through all
        # global attention blocks, carrying cross-frame/view context.
        # dino_tokens_list[-1]: [B, S, P_total, embed_dim] — DINOv2 last-layer features
        dino_tokens = dino_tokens_list[-1]

        # Use last-layer aggregated tokens as primary feature source
        # Shape: [B, S, P_total, 2*embed_dim]
        image_tokens = agg_tokens_list[-1]

        preds = {}

        # ── 2. Camera head (optional) ──────────────────────────────────────
        if self.camera_head is not None:
            with torch.amp.autocast("cuda", enabled=False):
                pose_enc_list = self.camera_head(agg_tokens_list)
            pose_enc = pose_enc_list[-1]           # [B, S, 9]
            preds["pose_enc"] = pose_enc

            if extrinsics is None:
                # Phase 2: use predicted pose
                extrinsics_used = self._pose_enc_to_extrinsics(
                    pose_enc, (images.shape[-2], images.shape[-1])
                )
                preds["predicted_extrinsics"] = extrinsics_used
            else:
                extrinsics_used = extrinsics
        else:
            assert extrinsics is not None, (
                "Either provide GT extrinsics or enable CameraHead"
            )
            extrinsics_used = extrinsics

        # ── 3. Plücker rays ────────────────────────────────────────────────
        # Broadcast intrinsics to [B, S, 3, 3]
        intrinsics_bs = intrinsics.unsqueeze(1).expand(B, S, 3, 3)

        plucker_rays = compute_plucker_rays_patch(
            extrinsics_used.float(),
            intrinsics_bs.float(),
            H, W,
            self.patch_size,
        )   # [B, S, N_patches, 6]

        if self.stop_gradient_plucker and extrinsics is None:
            # Phase 2: block pose-gradient from contaminating slot router
            plucker_rays = plucker_stop_gradient(plucker_rays)

        preds["plucker_rays"] = plucker_rays

        # ── 3b. Track encoder (optional) ───────────────────────────────────
        track_tokens = None
        track_mask   = None
        if self.use_track_tokens and not disable_tracks \
                and self.track_encoder is not None \
                and tracks_2d is not None and tracks_vis is not None:
            # Track mask: valid if visible in at least one frame
            track_mask = (tracks_vis.sum(dim=1) > 0).float()     # [B, N_t]

            # Frame-0 visual anchor: grid_sample DINOv2 last-layer patches at
            # each track's frame-0 pixel. Gives each track token a semantic
            # prior ("this track starts on a door patch" / "on a wheel patch"),
            # letting slot_track_fuse match semantics (not just geometry).
            vis_feat = None
            if getattr(self.track_encoder, "dim_vis_feat", 0) > 0:
                D_dino = dino_tokens.shape[-1]
                H_p = H // self.patch_size
                W_p = W // self.patch_size
                dino0 = dino_tokens[:, 0, patch_start_idx:, :]      # [B, N_p, D]
                fmap  = dino0.reshape(B, H_p, W_p, D_dino).permute(0, 3, 1, 2)
                # Normalise frame-0 track pixels to [-1, 1] for grid_sample
                gx = tracks_2d[:, 0, :, 0] / max(W - 1, 1) * 2.0 - 1.0
                gy = tracks_2d[:, 0, :, 1] / max(H - 1, 1) * 2.0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)    # [B,1,N,2]
                vis_feat = F.grid_sample(
                    fmap.float(), grid.float(),
                    mode="bilinear", padding_mode="border", align_corners=True,
                ).squeeze(2).transpose(1, 2)                         # [B, N, D]
                vis_feat = vis_feat.to(dino_tokens.dtype)

            track_tokens = self.track_encoder(
                tracks_2d, tracks_vis, vis_feat=vis_feat,
            )                                                        # [B, N_t, D]
            preds["track_mask_valid_frac"] = track_mask.mean().detach()

        # ── 4. PartSlotRouter (decoder) ────────────────────────────────────
        slot_features, assign_maps = self.part_slot_router(
            image_tokens,
            dino_tokens,
            plucker_rays,
            timestamps,
            img_hw=(H, W),
            slot_init=slot_states,   # slot tokens pre-enriched by Aggregator global blocks
            track_tokens=track_tokens,
            track_mask=track_mask,
        )
        preds["slot_features"] = slot_features   # [B, P, D]
        preds["assign_maps"]   = assign_maps      # [B, P, H_p, W_p]

        # ── 5. Articulation head ────────────────────────────────────────────
        art = self.articulation_head(slot_features, timestamps)
        preds.update({
            "motion_type_logits": art["motion_type_logits"],  # [B, P, 2]
            "axis":               art["axis"],                 # [B, P, 3]
            "pivot":              art["pivot"],                # [B, P, 3]
            "scalars":            art["scalars"],              # [B, P, S]
            "bbox_center":        art["bbox_center"],          # [B, P, 3]
            "bbox_size":          art["bbox_size"],            # [B, P, 3]
        })

        # ── 6. Gaussian head (mu constrained by predicted bbox) ─────────────
        # First-frame patch features: concat Aggregator last-layer (2C) and
        # DINOv2 last-layer (C) → [B, N_p, 3C=3072].
        # Special tokens (camera + register, patch_start_idx=5) are excluded
        # so only spatial patch tokens are used for the masked pooling.
        agg_frame0  = image_tokens[:, 0, patch_start_idx:, :]   # [B, N_p, 2C]
        dino_frame0 = dino_tokens[:,  0, patch_start_idx:, :]   # [B, N_p,  C]
        patch_feats_frame0 = torch.cat([agg_frame0, dino_frame0], dim=-1)  # [B, N_p, 3C]

        gs = self.gaussian_head(
            slot_features,
            bbox_center=art["bbox_center"],
            bbox_size=art["bbox_size"],
            patch_feats_frame0=patch_feats_frame0,
            assign_maps=assign_maps,
        )
        preds["gs_mu"]      = gs["mu"]       # [B, P, N_g, 3]
        preds["gs_rot"]     = gs["rot"]      # [B, P, N_g, 4]
        preds["gs_scale"]   = gs["scale"]    # [B, P, N_g, 3]
        preds["gs_color"]   = gs["color"]    # [B, P, N_g, 3]
        preds["gs_opacity"] = gs["opacity"]  # [B, P, N_g, 1]

        return preds

    # ------------------------------------------------------------------
    # Convenience: freeze / unfreeze parameter groups
    # ------------------------------------------------------------------
    def set_phase(self, phase: str, warmup: bool = False):
        """
        Configure parameter freezing for each training phase.

        phase: "1a", "1b", or "2"
        warmup: if True (Phase 1a early stage), freeze kinematic + gaussian heads
        """
        # DINOv2 always frozen (inside aggregator.patch_embed)
        for p in self.aggregator.patch_embed.parameters():
            p.requires_grad_(False)

        if phase in ("1a", "1b"):
            for p in self.aggregator.parameters():
                p.requires_grad_(True)
            for mod in [self.camera_head, self.part_slot_router,
                        self.articulation_head, self.gaussian_head,
                        self.track_encoder]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)
            # During warmup: freeze articulation and gaussian heads
            if warmup:
                for mod in [self.articulation_head, self.gaussian_head]:
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad_(False)

        elif phase == "2":
            all_layers = (
                list(self.aggregator.frame_blocks) +
                list(self.aggregator.global_blocks)
            )
            n_freeze = max(0, len(all_layers) - 4)
            for layer in all_layers[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad_(False)
            for layer in all_layers[n_freeze:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for mod in [self.camera_head, self.part_slot_router,
                        self.articulation_head, self.gaussian_head,
                        self.track_encoder]:
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)
        else:
            raise ValueError(f"Unknown phase: {phase!r}. Expected '1a', '1b', or '2'.")

        # Always re-freeze DINOv2
        for p in self.aggregator.patch_embed.parameters():
            p.requires_grad_(False)
