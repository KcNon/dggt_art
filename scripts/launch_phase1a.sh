#!/bin/bash
# ============================================================
# Launch Phase 1a training (art_v20, new dataset + slot-in-Aggregator arch)
# Usage: bash scripts/launch_phase1a.sh
# ============================================================

# NOTE: /data2 (2.9G free) and / (6.4G free) are nearly full — pick a disk with
# room before a long run; each checkpoint (model+optimizer) can exceed 1GB.
# Fresh run for the new architecture (slot0=base + router bg-sink). The old
# ckpt_016000.pth predates these (no bg_token, old slot0=bg∪static semantics), so
# it is NOT resumed; we also use a clean dir so it isn't clobbered.
PHASE1A_DIR="/data5/lza/checkpoint/Art/phase1a_bgsink_warmup"
TRAIN_SCRIPT="/home/ziang/code/dggt_art/train_art.py"

mkdir -p "${PHASE1A_DIR}"

echo "[$(date)] Launching Phase 1a → ${PHASE1A_DIR}/train.log"

# GPUs 0,1,3 free; GPU 2 held by another user's pcdet job (skip it).
CUDA_VISIBLE_DEVICES=0,1,3 torchrun \
    --nproc_per_node=3 \
    --master_port=29507 \
    "${TRAIN_SCRIPT}" \
    --data_root             /data2/lza/partnet-Mobility/data_processed \
    --output_dir            "${PHASE1A_DIR}" \
    --phase                 1a \
    --num_frames            8 \
    --img_size              518 \
    --batch_size            1 \
    --lr                    1e-4 \
    --weight_decay          1e-4 \
    --grad_clip             1.0 \
    --total_steps           20000 \
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
    --save_interval         2000 \
    --val_interval          500 \
    --warmup_iou_threshold  0.7 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers           3 \
    --exclude_cams \
    --reset_scheduler \
    >> "${PHASE1A_DIR}/train.log" 2>&1

echo "[$(date)] Phase 1a finished (exit code $?)"
