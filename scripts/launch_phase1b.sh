#!/bin/bash
# ============================================================
# Launch Phase 1b (SDF rendering + articulation), resuming the trained
# Phase-1a segmentation prior. Weights-only resume (--reset_step) keeps the
# router/segmentation but starts step/optimizer/scheduler fresh.
#
# The loss-side SDF/RGB query MLPs are excluded from DDP's reducer (used outside
# the DDP forward graph → would be "marked ready twice") and their grads are
# all-reduced manually in train_art.py. That lets us keep full 518 + 8 frames +
# gradient checkpointing on 4 GPUs.
# Usage: bash scripts/launch_phase1b.sh
# ============================================================

PHASE1A_CKPT="/data5/lza/checkpoint/Art/phase1a_bgsink_warmup/ckpt_020000.pth"
PHASE1B_DIR="/data5/lza/checkpoint/Art/phase1b_bgsink"
TRAIN_SCRIPT="/home/ziang/code/dggt_art/train_art.py"

# nerfacc JIT-compiles its CUDA backend on first phase-1b call → needs ninja on PATH,
# CUDA 12.1 toolkit, and gcc-11 host compiler. Set explicitly so the script is robust
# under nohup / non-interactive shells (no `conda activate`). See memory nerfacc-build-env.
export PATH="/home/ziang/miniconda3/envs/dggt/bin:${PATH}"
export CUDA_HOME=/usr/local/cuda-12.1
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

mkdir -p "${PHASE1B_DIR}"
echo "[$(date)] Launching Phase 1b from ${PHASE1A_CKPT} → ${PHASE1B_DIR}/train.log"

# Use 3 GPUs (0,1,2); GPU 3 left free per user request.
# Absolute torchrun so the script works under nohup/non-interactive shells (no conda activate).
TORCHRUN=/home/ziang/miniconda3/envs/dggt/bin/torchrun
CUDA_VISIBLE_DEVICES=0,1,2 "${TORCHRUN}" \
    --nproc_per_node=3 \
    --master_port=29508 \
    "${TRAIN_SCRIPT}" \
    --data_root             /data2/lza/partnet-Mobility/data_processed \
    --resume                "${PHASE1A_CKPT}" \
    --reset_step \
    --reset_scheduler \
    --output_dir            "${PHASE1B_DIR}" \
    --phase                 1b \
    --num_frames            8 \
    --img_size              518 \
    --batch_size            1 \
    --lr                    2e-5 \
    --lr_warmup_steps       500 \
    --weight_decay          1e-4 \
    --grad_clip             0.5 \
    --total_steps           30000 \
    --w_type                0.5 \
    --w_axis                0.5 \
    --w_pivot               0.1 \
    --w_scalar              0.3 \
    --w_render              1.0 \
    --w_render_global       0.0 \
    --w_depth               0.0 \
    --w_bbox                0.5 \
    --w_dead_opacity        0.1 \
    --w_pose_enc            0.0 \
    --l1_sparsity           0.01 \
    --l1_sparsity_warmup    0.0 \
    --log_interval          50 \
    --save_interval         3000 \
    --val_interval          999999 \
    --val_max_batches       64 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers           3 \
    --exclude_cams \
    >> "${PHASE1B_DIR}/train.log" 2>&1

echo "[$(date)] Phase 1b finished (exit code $?)"
