#!/bin/bash
# ============================================================
# Auto-launch Phase 1b once Phase 1a warmup checkpoint appears.
# Polls Phase 1a torchrun PID to avoid OOM from overlapping jobs.
# Usage: bash scripts/launch_phase1b.sh <phase1a_torchrun_pid>
# ============================================================

PHASE1A_PID="${1:-}"
PHASE1A_DIR="/data2/cyt/checkpoints/art_v20_phase1a"
PHASE1B_DIR="/data2/cyt/checkpoints/art_v20_phase1b"
TRAIN_SCRIPT="/home/yuantao/code/dggt_art/train_art.py"

# ── Wait for warmup checkpoint ─────────────────────────────────────────────
echo "[$(date)] Watching for Phase 1a warmup checkpoint in ${PHASE1A_DIR} ..."
while true; do
    WARMUP_CKPT=$(ls "${PHASE1A_DIR}"/ckpt_warmup_*.pth 2>/dev/null | sort | tail -1)
    if [ -n "${WARMUP_CKPT}" ]; then
        echo "[$(date)] Found warmup checkpoint: ${WARMUP_CKPT}"
        break
    fi
    sleep 30
done

# ── Wait for Phase 1a process to fully exit (avoids GPU OOM) ──────────────
if [ -n "${PHASE1A_PID}" ]; then
    echo "[$(date)] Waiting for Phase 1a (PID ${PHASE1A_PID}) to exit..."
    while kill -0 "${PHASE1A_PID}" 2>/dev/null; do
        sleep 10
    done
    echo "[$(date)] Phase 1a exited. Waiting 15s for CUDA memory release..."
    sleep 15
fi

# ── Launch Phase 1b ───────────────────────────────────────────────────────
mkdir -p "${PHASE1B_DIR}"
echo "[$(date)] Launching Phase 1b from ${WARMUP_CKPT}"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    "${TRAIN_SCRIPT}" \
    --data_root             /data2/cyt/data_root_refine \
    --output_dir            "${PHASE1B_DIR}" \
    --phase                 1b \
    --num_frames            6 \
    --img_size              518 \
    --batch_size            1 \
    --lr                    2e-5 \
    --weight_decay          1e-4 \
    --grad_clip             0.5 \
    --total_steps           50000 \
    --resume                /data2/cyt/checkpoints/art_v20_phase1b/ckpt_012000.pth \
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
    --log_interval          50 \
    --save_interval         3000 \
    --val_interval          500 \
    --warmup_iou_threshold  0.75 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers           4 \
    >> "${PHASE1B_DIR}/train.log" 2>&1

echo "[$(date)] Phase 1b finished (exit code $?)"
