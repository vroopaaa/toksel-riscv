#!/usr/bin/env bash
# Quick launcher for Hardware & Latency Analysis
# Run this inside your activated virtual environment with torch/timm installed.

DEVICE="cuda"
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ">> [WARNING] CUDA not detected in current Python environment. Falling back to CPU."
    DEVICE="cpu"
fi

python3 hardware_analysis/profile_hardware.py \
    --device "$DEVICE" \
    --image-size 512 \
    --backbone "vit_small_patch16_224" \
    --warmup 25 \
    --runs 100 \
    "$@"
