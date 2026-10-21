#!/usr/bin/env bash
cd "/home/roopi/ISCAS 27/Saliency_Deepak"
export PYTHONPATH=".:Saliency-Objectness-Guided-Sparse-ViT:$PYTHONPATH"
python3 hardware_analysis/test_sampler_batch_scaling.py
