#!/usr/bin/env bash
cd "/home/roopi/ISCAS 27/Saliency_Deepak"
export PYTHONPATH=".:Saliency-Objectness-Guided-Sparse-ViT:$PYTHONPATH"
python3 hardware_analysis/profile_embedding_substages.py "$@"
PYTHON_BIN="python3"
if ! $PYTHON_BIN -c "import torch" &>/dev/null; then
    if [ -f "/home/roopi/anaconda3/envs/env1/bin/python" ]; then
        PYTHON_BIN="/home/roopi/anaconda3/envs/env1/bin/python"
    fi
fi

$PYTHON_BIN hardware_analysis/profile_embedding_substages.py "$@"
