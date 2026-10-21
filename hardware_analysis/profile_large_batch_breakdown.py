import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

# Ensure repository root and workspace are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_DIR, WORKSPACE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.sparse_vit_det import SparseViTDetector
from hardware_analysis.optimized_samplers import MaxPoolFixationSampler


@torch.no_grad()
def profile_model_stages(
    model: SparseViTDetector,
    batch_size: int = 12,
    image_size: int = 512,
    device: str = "cuda",
    warmup: int = 10,
    runs: int = 25
) -> Dict[str, float]:
    """
    Measures the exact latency (in milliseconds) of every individual stage
    in SparseViTDetector using high-precision CUDA events.
    """
    model.eval()
    H = W = image_size
    images = torch.randn(batch_size, 3, H, W, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(images)
    torch.cuda.synchronize()

    # Storage for stage timings
    stage_times = {
        "1. CNN Feature Extractor (MobileNetV3)": [],
        "2. Fixation Sampler": [],
        "3. Local Patch Extractor (grid_sample)": [],
        "4. Patch Embedding & Pos/Score Proj": [],
        "5. Transformer Blocks (12 Layers)": [],
        "6. Detection Head": [],
        "Total Pipeline (End-to-End)": []
    }

    # CUDA Events
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_cnn = torch.cuda.Event(enable_timing=True)
    ev_sampler = torch.cuda.Event(enable_timing=True)
    ev_patch_ext = torch.cuda.Event(enable_timing=True)
    ev_embed = torch.cuda.Event(enable_timing=True)
    ev_transformer = torch.cuda.Event(enable_timing=True)
    ev_head = torch.cuda.Event(enable_timing=True)

    for _ in range(runs):
        ev_start.record()

        # Stage 1: Feature Extractor
        X, S, O = model.feature_extractor(images)
        ev_cnn.record()

        # Stage 2: Fixation Sampler
        coords, norm_coords, fixation_scores, F_map = model.fixation_sampler(S, O, image_size=(H, W))
        ev_sampler.record()

        # Stage 3: Patch Extractor (grid_sample)
        patches = model.patch_extractor(images, coords)
        ev_patch_ext.record()

        # Stage 4: Patch Embedding & Pos/Score Injections + Global Token Projection
        foveal_tokens = model.extract_patch_embeddings(patches)
        grid_pos_embed = model.vit.pos_embed[:, 1:, :]
        r_pos = model.pos_interpolator(norm_coords, grid_pos_embed)
        score_tokens = model.score_embed(fixation_scores)
        foveal_tokens = foveal_tokens + r_pos + score_tokens

        g_feat = model.global_token_proj(X)
        g_tokens = g_feat.flatten(2).transpose(1, 2)
        g_tokens = g_tokens + model.global_pos_embed
        tokens = torch.cat([g_tokens, foveal_tokens], dim=1)
        tokens = model.vit.pos_drop(tokens)
        ev_embed.record()

        # Stage 5: Transformer Blocks
        for block in model.vit.blocks:
            tokens = block(tokens)
        tokens = model.vit.norm(tokens)
        ev_transformer.record()

        # Stage 6: Detection Head
        out_foveal_tokens = tokens[:, model.num_global_tokens:]
        pred_logits, pred_boxes = model.detection_head(out_foveal_tokens, norm_coords)
        ev_head.record()

        torch.cuda.synchronize()

        # Record elapsed times
        t_cnn = ev_start.elapsed_time(ev_cnn)
        t_sampler = ev_cnn.elapsed_time(ev_sampler)
        t_patch_ext = ev_sampler.elapsed_time(ev_patch_ext)
        t_embed = ev_patch_ext.elapsed_time(ev_embed)
        t_transformer = ev_embed.elapsed_time(ev_transformer)
        t_head = ev_transformer.elapsed_time(ev_head)
        t_total = ev_start.elapsed_time(ev_head)

        stage_times["1. CNN Feature Extractor (MobileNetV3)"].append(t_cnn)
        stage_times["2. Fixation Sampler"].append(t_sampler)
        stage_times["3. Local Patch Extractor (grid_sample)"].append(t_patch_ext)
        stage_times["4. Patch Embedding & Pos/Score Proj"].append(t_embed)
        stage_times["5. Transformer Blocks (12 Layers)"].append(t_transformer)
        stage_times["6. Detection Head"].append(t_head)
        stage_times["Total Pipeline (End-to-End)"].append(t_total)

    # Compute means
    return {k: float(np.mean(v)) for k, v in stage_times.items()}


def main():
    parser = argparse.ArgumentParser(description="Large Batch Stage Breakdown Profiler")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 100)
    print("       STAGE-BY-STAGE LATENCY BREAKDOWN: ORIGINAL SPARSE vs. METHOD 2 MAXPOOL")
    print("=" * 100)
    print(f">> Device       : {device} ({torch.cuda.get_device_name(0)})")
    print(f">> Image Size   : {args.image_size}x{args.image_size}")
    print(f">> ViT Backbone : {args.backbone}")
    print(f">> Iterations   : {args.runs} timed runs ({args.warmup} warmup)")
    print("=" * 100)

    # Instantiate Models
    print(">> Initializing Original Sparse-ViT...")
    model_orig = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)

    print(">> Initializing Method 2 (MaxPool NMS) Sparse-ViT...")
    model_maxpool = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)
    model_maxpool.fixation_sampler = MaxPoolFixationSampler(
        K=128, patch_size=16, image_size=(args.image_size, args.image_size)
    ).to(device)

    test_batches = [1, 4, 8, 12]

    for B in test_batches:
        print(f"\n" + "#" * 100)
        print(f"                               BATCH SIZE = {B:2d}")
        print("#" * 100)

        t_orig = profile_model_stages(model_orig, batch_size=B, image_size=args.image_size, device=args.device, warmup=args.warmup, runs=args.runs)
        t_pool = profile_model_stages(model_maxpool, batch_size=B, image_size=args.image_size, device=args.device, warmup=args.warmup, runs=args.runs)

        print(f"\n{'Pipeline Stage':<42} | {'Original (ms)':<15} | {'Method 2 (ms)':<15} | {'Delta (ms)':<12} | {'Verdict / Ratio':<20}")
        print("-" * 115)

        for stage_name in t_orig.keys():
            v_orig = t_orig[stage_name]
            v_pool = t_pool[stage_name]
            delta = v_orig - v_pool
            if "Transformer" in stage_name:
                ratio_str = f"Diff: {abs(delta):.2f} ms (~Identical!)"
            elif "Sampler" in stage_name:
                ratio_str = f"{v_orig / max(v_pool, 1e-4):.1f}x faster"
            elif "Total" in stage_name:
                ratio_str = f"Speedup: {v_orig / max(v_pool, 1e-4):.2f}x"
            else:
                ratio_str = f"Diff: {abs(delta):.2f} ms"

            print(f"{stage_name:<42} | {v_orig:8.2f} ms      | {v_pool:8.2f} ms      | {delta:+7.2f} ms   | {ratio_str}")

    print("\n" + "=" * 100)
    print(">> Benchmark completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()
