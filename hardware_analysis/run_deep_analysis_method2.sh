#!/usr/bin/env bash
# Dedicated Launcher for Method 2 Deep Hardware Analysis
cd "/home/roopi/ISCAS 27/Saliency_Deepak"

python3 hardware_analysis/deep_kernel_profiler_method2.py \
    --device cuda \
    --backbone "vit_small_patch16_224" \
    "$@"
