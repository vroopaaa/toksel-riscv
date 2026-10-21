#!/usr/bin/env bash
# Dedicated Launcher for Stage Breakdown across Batch Sizes
cd "/home/roopi/ISCAS 27/Saliency_Deepak"
export PYTHONPATH=".:Saliency-Objectness-Guided-Sparse-ViT:$PYTHONPATH"

python3 hardware_analysis/profile_large_batch_breakdown.py \
    --device cuda \
    --backbone "vit_small_patch16_224" \
    --runs 20 \
    --warmup 5 \
    "$@"
