#!/bin/bash
# ============================================================
# Phase D — sim→real domain fine-tune.
#
# Resumes from Phase C ckpt_100000 (sim assign_iou=0.77, real fg_iou=0.18).
# Diagnosis (doc/change.md "Phase D 诊断"):
#   - DINOv2 (frozen) features identical on sim/real.
#   - Aggregator output norm-std on real ≈ 6 vs sim ≈ 21 — feature collapse.
#   - Per-layer trace: layers 14-23 overfit sim; layers 0-13 cross-domain stable.
#   - Phase C training itself made real worse (std 13 → 6).
#
# Strategy:
#   - dataset_type = mixed (iTACO sim + iTACO real, real_mix_ratio = 0.3)
#   - phase = 2 with n_unfreeze_blocks = 10 (only late aggregator + heads)
#   - DINOv2 + first 14 frame/global blocks frozen (preserve cross-domain priors)
#   - lr = 5e-6 (gentle, avoid catastrophic forgetting of sim distribution)
#   - No render / kin loss for real batches (has_kin_gt=False branch handles it)
#   - tracks unused on sim_data side (has_motion_data=False); real motion still
#     works via MonST3R coarse motion_mask supervision.
# ============================================================

PHASE_D_DIR="/data2/cyt/checkpoints/art_v20_phase_d"
TRAIN_SCRIPT="/home/yuantao/code/dggt_art/train_art.py"
RESUME_CKPT="/data2/cyt/checkpoints/art_v20_phase_c/ckpt_100000.pth"

mkdir -p "${PHASE_D_DIR}"
echo "[$(date)] Launching Phase D (mixed sim+real fine-tune) from ${RESUME_CKPT}"
echo "         output → ${PHASE_D_DIR}/train.log"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port=29505 \
    "${TRAIN_SCRIPT}" \
    --data_root                  /data2/cyt/video2articulation/sim_data \
    --real_data_root             /data2/cyt/video2articulation/real_data \
    --dataset_type               mixed \
    --real_mix_ratio             0.3 \
    --output_dir                 "${PHASE_D_DIR}" \
    --phase                      2 \
    --n_unfreeze_blocks          10 \
    --num_frames                 8 \
    --img_size                   518 \
    --batch_size                 1 \
    --lr                         5e-6 \
    --weight_decay               1e-4 \
    --grad_clip                  0.5 \
    --total_steps                130000 \
    --resume                     "${RESUME_CKPT}" \
    --reset_scheduler \
    --w_type                     0.3 \
    --w_axis                     0.5 \
    --w_pivot                    0.2 \
    --w_scalar                   0.1 \
    --w_render                   0.1 \
    --w_render_global            0.0 \
    --w_bbox                     0.05 \
    --w_dead_opacity             0.05 \
    --w_pose_enc                 0.05 \
    --w_pseudo_mask              0.05 \
    --l1_sparsity                0.005 \
    --l1_sparsity_warmup         0.0 \
    --w_motion_mask              0.5 \
    --w_motion_track             0.0 \
    --motion_warmup_steps        0 \
    --use_track_tokens \
    --track_encoder_lr           1e-5 \
    --freeze_track_warmup_steps  0 \
    --max_tracks_per_sample      1024 \
    --motion_cache_name          motion_cache_gt.npz \
    --log_interval               50 \
    --save_interval              2500 \
    --val_interval               500 \
    --warmup_iou_threshold       0.0 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers                2 \
    >> "${PHASE_D_DIR}/train.log" 2>&1

echo "[$(date)] Phase D finished (exit code $?)"
