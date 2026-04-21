#!/bin/bash
# ============================================================
# Phase 1b + motion-aux-loss ablation run.
# Mirrors scripts/launch_phase1b.sh but:
#   - resumes from ckpt_012000 (same starting point as the GPU4-7 baseline)
#   - uses GPUs 0-3 (the baseline occupies 4-7)
#   - enables motion_mask / motion_track aux losses using the GT-mask
#     pseudo-labels produced by scripts/precompute_motion_from_mask.py
# Output goes to a separate directory so checkpoints do not clash.
# ============================================================

PHASE1B_MOTION_DIR="/data2/cyt/checkpoints/art_v20_phase1b_motion"
TRAIN_SCRIPT="/home/yuantao/code/dggt_art/train_art.py"
RESUME_CKPT="/data2/cyt/checkpoints/art_v20_phase1b/ckpt_012000.pth"

mkdir -p "${PHASE1B_MOTION_DIR}"
echo "[$(date)] Launching Phase 1b + motion aux from ${RESUME_CKPT}"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29503 \
    "${TRAIN_SCRIPT}" \
    --data_root             /data2/cyt/data_root_refine \
    --output_dir            "${PHASE1B_MOTION_DIR}" \
    --phase                 1b \
    --num_frames            6 \
    --img_size              518 \
    --batch_size            1 \
    --lr                    2e-5 \
    --weight_decay          1e-4 \
    --grad_clip             0.5 \
    --total_steps           50000 \
    --resume                "${RESUME_CKPT}" \
    --w_type                0.5 \
    --w_axis                1.0 \
    --w_pivot               0.3 \
    --w_scalar              0.2 \
    --w_render              0.3 \
    --w_render_global       0.1 \
    --w_bbox                0.1 \
    --w_dead_opacity        0.1 \
    --w_pose_enc            0.1 \
    --l1_sparsity           0.01 \
    --l1_sparsity_warmup    0.0 \
    --motion_cache_name     motion_cache_gt.npz \
    --w_motion_mask         0.3 \
    --w_motion_track        0.3 \
    --motion_warmup_steps   5000 \
    --log_interval          50 \
    --save_interval         3000 \
    --val_interval          500 \
    --warmup_iou_threshold  0.75 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers           4 \
    >> "${PHASE1B_MOTION_DIR}/train.log" 2>&1

echo "[$(date)] Phase 1b + motion finished (exit code $?)"
