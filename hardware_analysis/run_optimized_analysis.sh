#!/usr/bin/env bash
# Launcher for comparing Original vs. Optimized Samplers (Method 1 & Method 2)
cd "/home/roopi/ISCAS 27/Saliency_Deepak"

python3 hardware_analysis/benchmark_optimized.py \
    --device cuda \
    --image-size 512 \
    --backbone "vit_small_patch16_224" \
    --warmup 25 \
    --runs 100 \
    "$@"
