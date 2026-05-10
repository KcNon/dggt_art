#!/bin/bash
# ============================================================
# DRAFT — Real-data fine-tune (NOT YET RUNNABLE).
#
# Diagnosis (see doc/change.md "Phase D 诊断"):
#   - DINOv2 (frozen) features identical on sim and real (norm-std == 0).
#   - Aggregator output norm-std on sim ≈ 21, on real ≈ 6  →  feature collapse.
#   - Per-layer trace: divergence starts at layer 14/24.
#     Layers 0-13 are domain-stable; layers 14-23 overfit sim statistics.
#
# Fine-tune strategy: unfreeze only last ~10 aggregator blocks + heads,
# train on real_data with weak supervision (no GT kin):
#   - SAM2 fg-mask BCE / Hungarian against slot alphas
#   - MonST3R dynamic mask cross-entropy (motion vs static)
#   - Track motion loss using sim-track-pretrained TrackEncoder
#   - NO render loss, NO kin loss (no GT)
#
# Required code changes before running:
#   1. train_art.py:  add --dataset_type {sim,real}; route to iTACORealDataset.
#   2. train_art.py:  add --n_unfreeze_blocks N flag; pass into set_phase.
#   3. art_vggt.py::set_phase():  accept n_unfreeze_blocks for phase "2".
#   4. compute_loss path:  branch on dataset_type — drop kin/render/bbox/pose_enc
#      when GT is absent; keep mask/track/motion losses.
#   5. (Optional) Add color-jitter / blur augmentation to ArticulatedDataset to
#      narrow the sim/real domain gap simultaneously.
#
# These are non-trivial; review and approve before launching.
# ============================================================

FT_DIR="/data2/cyt/checkpoints/art_v20_ft_real"
TRAIN_SCRIPT="/home/yuantao/code/dggt_art/train_art.py"
RESUME_CKPT="/data2/cyt/checkpoints/art_v20_phase_c/ckpt_100000.pth"

mkdir -p "${FT_DIR}"
echo "[$(date)] DRAFT — Real fine-tune from ${RESUME_CKPT} → ${FT_DIR}/train.log"

# Single GPU (4 real scenes, batch=1, num_frames=4).
CUDA_VISIBLE_DEVICES=4 python \
    "${TRAIN_SCRIPT}" \
    --data_root                  /data2/cyt/video2articulation/real_data \
    --dataset_type               real \
    --output_dir                 "${FT_DIR}" \
    --phase                      2 \
    --n_unfreeze_blocks          10 \
    --num_frames                 4 \
    --img_size                   518 \
    --batch_size                 1 \
    --lr                         5e-6 \
    --weight_decay               1e-4 \
    --grad_clip                  0.5 \
    --total_steps                5000 \
    --resume                     "${RESUME_CKPT}" \
    --reset_scheduler \
    --w_type                     0.0 \
    --w_axis                     0.0 \
    --w_pivot                    0.0 \
    --w_scalar                   0.0 \
    --w_render                   0.0 \
    --w_render_global            0.0 \
    --w_bbox                     0.0 \
    --w_dead_opacity             0.05 \
    --w_pose_enc                 0.0 \
    --l1_sparsity                0.005 \
    --w_motion_mask              0.5 \
    --w_motion_track             0.3 \
    --motion_warmup_steps        0 \
    --use_track_tokens \
    --max_tracks_per_sample      1024 \
    --log_interval               20 \
    --save_interval              500 \
    --val_interval               200 \
    --gradient_checkpointing \
    --use_bf16 \
    --num_workers                2 \
    >> "${FT_DIR}/train.log" 2>&1

echo "[$(date)] DRAFT FT finished (exit code $?)"
