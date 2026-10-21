#!/usr/bin/env bash
# ==============================================================================
# Runner for Sparse-ViT Native In-Model Latency Profiler
#
# Measures latency directly inside the model forward pass via:
#   `outputs = model(images, return_timings=True)`
#
# Compares Original Sparse-ViT vs Sparse-ViT + Zero-Loop Max-Pool NMS.
# ==============================================================================

set -e

# Change to the root workspace directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Set up Python Path
export PYTHONPATH=".:Saliency-Objectness-Guided-Sparse-ViT:$PYTHONPATH"

# Detect Python binary with torch installed
PYTHON_CMD="python3"
if python -c "import torch" 2>/dev/null; then
    PYTHON_CMD="python"
elif python3 -c "import torch" 2>/dev/null; then
    PYTHON_CMD="python3"
fi

# Detect CUDA or CPU
DEVICE="cuda"
if ! $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ">> [INFO] CUDA not available in current environment; falling back to CPU."
    DEVICE="cpu"
fi

echo "================================================================================"
echo " Starting Native In-Model Benchmark with $PYTHON_CMD on $DEVICE"
echo "================================================================================"

$PYTHON_CMD hardware_analysis/test_native_timings.py \
    --device "$DEVICE" \
    --batch-sizes 1 12 \
    --image-size 512 \
    --K 128 \
    --warmup 10 \
    --runs 30 \
    "$@"
