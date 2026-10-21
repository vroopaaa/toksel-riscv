#!/usr/bin/env bash
# Launcher for Deep Hardware & Kernel Profiling Suite
cd "/home/roopi/ISCAS 27/Saliency_Deepak"

python3 hardware_analysis/deep_kernel_profiler.py \
    --device cuda \
    --backbone "vit_small_patch16_224" \
    "$@"
