#!/bin/bash
# ============================================================
# Launch Phase 1a training (art_v20, new dataset + slot-in-Aggregator arch)
# Usage: bash scripts/launch_phase1a.sh
# ============================================================

PHASE1A_DIR="/data2/cyt/checkpoints/art_v20_phase1a"
TRAIN_SCRIPT="/home/yuantao/code/dggt_art/train_art.py"

mkdir -p "${PHASE1A_DIR}"

echo "[$(date)] Launching Phase 1a → ${PHASE1A_DIR}/train.log"

CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun \
    --nproc_per_node=5 \
    --master_port=29503 \
    "${TRAIN_SCRIPT}" \
    --data_root             /data2/cyt/data_root_refine \
    --output_dir            "${PHASE1A_DIR}" \
    --phase                 1a \
    --num_frames            8 \
    --img_size              518 \
    --batch_size            1 \
    --lr                    1e-4 \
    --weight_decay          1e-4 \
    --grad_clip             1.0 \
    --total_steps           50000 \
    --w_type                0.5 \
    --w_axis                0.0 \
    --w_pivot               0.0 \
    --w_scalar              0.0 \
    --w_render              0.0 \
    --w_bbox                0.0 \
    --w_dead_opacity        0.05 \
    --w_pose_enc            0.1 \
    --l1_sparsity           0.005 \
    --l1_sparsity_warmup    0.0 \
    --log_interval          50 \
    --save_interval         3000 \
    --val_interval          500 \
    --warmup_iou_threshold  0.8 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers           5 \
    --exclude_cam \
    --reset_scheduler \
    >> "${PHASE1A_DIR}/train.log" 2>&1

echo "[$(date)] Phase 1a finished (exit code $?)"
