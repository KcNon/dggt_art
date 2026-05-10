#!/bin/bash
# ============================================================
# Phase C — TrackEncoder + track-stream injection in PartSlotRouter.
# Resumes from Phase 1a ckpt_021000 (mask IoU plateaued at ~0.76).
#
# Phase 1a 在 0.76 见顶,直接跳过 warmup gate 进 Phase 1b:
#   - set_phase("1b", warmup=False) 解冻所有头(含 kin + gaussian)
#   - compute_loss 不再短路,kin + render + bbox + pose_enc 全部接入
#   - 同时开启 motion_mask / motion_track aux loss(复用 motion_cache_gt.npz)
#   - Track-warmup 显式关闭(start_step=21000 早已过任何合理边界)
#
# Loss 权重对齐 launch_phase1b_motion.sh(已验证过的 Phase 1b 配置),
# 额外加上 --use_track_tokens 把轨迹作为前向输入注入 PartSlotRouter。
# ============================================================

PHASE_C_DIR="/data2/cyt/checkpoints/art_v20_phase_c"
TRAIN_SCRIPT="/home/yuantao/code/dggt_art/train_art.py"
RESUME_CKPT="/data2/cyt/checkpoints/art_v20_phase_c/ckpt_050000.pth"

mkdir -p "${PHASE_C_DIR}"
echo "[$(date)] Launching Phase C (phase=1b + tracks) from ${RESUME_CKPT} → ${PHASE_C_DIR}/train.log"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nproc_per_node=4 \
    --master_port=29504 \
    "${TRAIN_SCRIPT}" \
    --data_root                  /data2/cyt/data_root_refine \
    --output_dir                 "${PHASE_C_DIR}" \
    --phase                      1b \
    --num_frames                 8 \
    --img_size                   518 \
    --batch_size                 1 \
    --lr                         2e-5 \
    --weight_decay               1e-4 \
    --grad_clip                  0.5 \
    --total_steps                100000 \
    --resume                     "${RESUME_CKPT}" \
    --reset_scheduler \
    --w_type                     0.5 \
    --w_axis                     1.0 \
    --w_pivot                    0.3 \
    --w_scalar                   0.2 \
    --w_render                   0.3 \
    --w_render_global            0.1 \
    --w_bbox                     0.1 \
    --w_dead_opacity             0.1 \
    --w_pose_enc                 0.1 \
    --l1_sparsity                0.01 \
    --l1_sparsity_warmup         0.0 \
    --motion_cache_name          motion_cache_gt.npz \
    --w_motion_mask              0.3 \
    --w_motion_track             0.3 \
    --motion_warmup_steps        5000 \
    --use_track_tokens \
    --track_encoder_lr           3e-4 \
    --freeze_track_warmup_steps  0 \
    --max_tracks_per_sample      1024 \
    --log_interval               50 \
    --save_interval              5000 \
    --val_interval               500 \
    --warmup_iou_threshold       0.8 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers                4 \
    >> "${PHASE_C_DIR}/train.log" 2>&1

echo "[$(date)] Phase C finished (exit code $?)"
