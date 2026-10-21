#!/usr/bin/env python3
"""
Micro-Profiler & Granular Decomposition for Stage 4:
Patch Embedding, Continuous Positional Encoding, and Dual-Resolution Token Assembly.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_DIR, WORKSPACE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.sparse_vit_det import SparseViTDetector


@torch.no_grad()
def measure_substages(device="cuda", runs=50, warmup=15):
    print("=" * 80)
    print("      MICRO-PROFILING: STAGE 4 (EMBEDDING & POSITIONAL INJECTIONS)")
    print("=" * 80)
    print(f">> Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f">> Running {runs} timed iterations ({warmup} warmup)...")
    print("=" * 80)

    model = SparseViTDetector(num_classes=80, K=128, image_size=(512, 512)).to(device)
def benchmark_stage4_substages(batch_size=1, runs=50, warmup=15, image_size=512, K=128, device="cuda"):
    model = SparseViTDetector(num_classes=80, K=K, image_size=(image_size, image_size)).to(device)
    model.eval()

    # Pre-generate inputs matching the exact shapes from earlier stages
    B = 1
    K = 128
    B = batch_size
    D = model.embed_dim
    images = torch.randn(B, 3, 512, 512, device=device)
    
    # Run through feature extractor and sampler once to get real inputs
    images = torch.randn(B, 3, image_size, image_size, device=device)

    # Run through initial stages to get authentic intermediate feature tensors
    X, S, O = model.feature_extractor(images)
    coords, norm_coords, fixation_scores, F_map = model.fixation_sampler(S, O)
    patches = model.patch_extractor(images, coords)

    grid_pos_embed = model.vit.pos_embed[:, 1:, :]
    G = model.grid_size
    grid_embed = grid_pos_embed.view(1, G, G, D).permute(0, 3, 1, 2).expand(B, D, G, G)

    sample_grid_x = 2.0 * norm_coords[..., 0] - 1.0
    sample_grid_y = 2.0 * norm_coords[..., 1] - 1.0
    sample_grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).view(B, K, 1, 2)

    # Warmup
    for _ in range(warmup):
        _ = model.extract_patch_embeddings(patches)
        _ = model.pos_interpolator(norm_coords, grid_pos_embed)
        _ = F.grid_sample(grid_embed, sample_grid, mode='bicubic', align_corners=False)
        _ = model.pos_interpolator.fourier_encoder(norm_coords)
        _ = model.score_embed(fixation_scores)
        _ = model.global_token_proj(X)
    torch.cuda.synchronize()

    # Individual Timers
    times = {
        "1. Pure Patch Embedding Conv2d + LayerNorm": [],
        "2. Bicubic Positional Grid Interpolation": [],
        "3. 2D Fourier Positional MLP (Sin/Cos + 2 Linear)": [],
        "4. Fixation Score Embedding MLP": [],
        "5. Global Context Token Proj (AvgPool + 1x1 Conv)": [],
        "6. Tensor Reshaping & Concatenation (torch.cat)": []
        "1. Pure Patch Embedding (Conv2d + LayerNorm)": [],
        "2. Continuous Positional Spline Interpolation (Bicubic)": [],
        "3. 2D Fourier Positional MLP (Sin/Cos Bands + 2-layer MLP)": [],
        "4. Fixation Score MLP Projection (Scalar -> Dim)": [],
        "5. Global Background Context Projection (AvgPool + 1x1 Conv)": [],
        "6. Dual-Res Sequence Assembly (torch.cat + pos_drop)": []
    }

    evs = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_patch = torch.cuda.Event(enable_timing=True)
    ev_bicubic = torch.cuda.Event(enable_timing=True)
    ev_fourier = torch.cuda.Event(enable_timing=True)
    ev_score = torch.cuda.Event(enable_timing=True)
    ev_gproj = torch.cuda.Event(enable_timing=True)
    ev_concat = torch.cuda.Event(enable_timing=True)

    # Separate out the bicubic interpolation and Fourier parts from PositionalInterpolator
    grid_embed = grid_pos_embed.view(1, 14, 14, D).permute(0, 3, 1, 2).expand(B, D, 14, 14)
    sample_grid_x = 2.0 * norm_coords[..., 0] - 1.0
    sample_grid_y = 2.0 * norm_coords[..., 1] - 1.0
    sample_grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).view(B, K, 1, 2)

    for _ in range(runs):
        evs[0].record()
        ev_start.record()

        # 1. Patch Embedding Conv2d
        # 1. Pure Patch Embedding
        foveal_tokens = model.extract_patch_embeddings(patches)
        evs[1].record()
        ev_patch.record()

        # 2. Bicubic Grid Sample
        interpolated = torch.nn.functional.grid_sample(
        # 2. Bicubic Positional Spline Grid Interpolation
        interpolated = F.grid_sample(
            grid_embed, sample_grid, mode='bicubic', padding_mode='border', align_corners=False
        ).squeeze(-1).permute(0, 2, 1)
        evs[2].record()
        ev_bicubic.record()

        # 3. Fourier Positional MLP
        # 3. 2D Fourier Positional MLP
        fourier_pos = model.pos_interpolator.fourier_encoder(norm_coords)
        r_pos = interpolated + model.pos_interpolator.alpha * fourier_pos
        evs[3].record()
        ev_fourier.record()

        # 4. Score Embedding MLP
        score_tokens = model.score_embed(fixation_scores)
        foveal_tokens = foveal_tokens + r_pos + score_tokens
        evs[4].record()
        ev_score.record()

        # 5. Global Token Proj
        g_feat = model.global_token_proj(X)
        g_tokens = g_feat.flatten(2).transpose(1, 2) + model.global_pos_embed
        evs[5].record()
        ev_gproj.record()

        # 6. Concat
        # 6. Concat & Pos Drop
        tokens = torch.cat([g_tokens, foveal_tokens], dim=1)
        tokens = model.vit.pos_drop(tokens)
        evs[6].record()
        ev_concat.record()

        torch.cuda.synchronize()

        times["1. Pure Patch Embedding Conv2d + LayerNorm"].append(evs[0].elapsed_time(evs[1]))
        times["2. Bicubic Positional Grid Interpolation"].append(evs[1].elapsed_time(evs[2]))
        times["3. 2D Fourier Positional MLP (Sin/Cos + 2 Linear)"].append(evs[2].elapsed_time(evs[3]))
        times["4. Fixation Score Embedding MLP"].append(evs[3].elapsed_time(evs[4]))
        times["5. Global Context Token Proj (AvgPool + 1x1 Conv)"].append(evs[4].elapsed_time(evs[5]))
        times["6. Tensor Reshaping & Concatenation (torch.cat)"].append(evs[5].elapsed_time(evs[6]))
        times["1. Pure Patch Embedding (Conv2d + LayerNorm)"].append(ev_start.elapsed_time(ev_patch))
        times["2. Continuous Positional Spline Interpolation (Bicubic)"].append(ev_patch.elapsed_time(ev_bicubic))
        times["3. 2D Fourier Positional MLP (Sin/Cos Bands + 2-layer MLP)"].append(ev_bicubic.elapsed_time(ev_fourier))
        times["4. Fixation Score MLP Projection (Scalar -> Dim)"].append(ev_fourier.elapsed_time(ev_score))
        times["5. Global Background Context Projection (AvgPool + 1x1 Conv)"].append(ev_score.elapsed_time(ev_gproj))
        times["6. Dual-Res Sequence Assembly (torch.cat + pos_drop)"].append(ev_gproj.elapsed_time(ev_concat))

    print(f"{'Sub-Module Operation':<52} | {'Measured Latency (ms)':<22} | {'Share (%)'}")
    print("-" * 88)
    
    total_ms = sum(np.mean(v) for v in times.values())
    # Bilinear comparison test for Positional Interpolator
    bilinear_times = []
    ev_b0 = torch.cuda.Event(enable_timing=True)
    ev_b1 = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        ev_b0.record()
        _ = F.grid_sample(grid_embed, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        ev_b1.record()
        torch.cuda.synchronize()
        bilinear_times.append(ev_b0.elapsed_time(ev_b1))

    total_stage4 = sum(float(np.mean(v)) for v in times.values())
    bicubic_mean = float(np.mean(times["2. Continuous Positional Spline Interpolation (Bicubic)"]))
    bilinear_mean = float(np.mean(bilinear_times))

    print(f"\n>> BATCH SIZE B = {B} (K = {K} Patches, Embed Dim D = {D})")
    print(f"{'Sub-Module Operation':<62} | {'Measured Latency (ms)':<22} | {'Share (%)'}")
    print("-" * 98)

    for k, v in times.items():
        m = float(np.mean(v))
        s = float(np.std(v))
        pct = (m / total_ms) * 100.0
        print(f"{k:<52} | {m:6.3f} +/- {s:5.3f} ms   | {pct:5.1f}%")
        pct = (m / total_stage4) * 100.0
        print(f"{k:<62} | {m:6.3f} +/- {s:5.3f} ms   | {pct:5.1f}%")

    print("-" * 88)
    print(f"{'TOTAL MEASURED STAGE 4 TIME':<52} | {total_ms:6.3f} ms               | 100.0%")
    print("=" * 88)
    print("-" * 98)
    print(f"{'TOTAL COMBINED STAGE 4 LATENCY':<62} | {total_stage4:6.3f} ms               | 100.0%")
    print("=" * 98)
    print(f"  [Ablation Insight] Positional Interpolator Grid Sample Mode:")
    print(f"   - Current:  Bicubic Spline mode   -> {bicubic_mean:6.3f} ms ({bicubic_mean / total_stage4 * 100:4.1f}% of Stage 4)")
    print(f"   - Option:   Bilinear mode         -> {bilinear_mean:6.3f} ms ({bicubic_mean / max(bilinear_mean, 1e-4):.1f}x faster grid_sample)")
    print(f"   - Projected Stage 4 with Bilinear -> {total_stage4 - bicubic_mean + bilinear_mean:6.3f} ms ({total_stage4 / max(total_stage4 - bicubic_mean + bilinear_mean, 1e-4):.2f}x speedup for Stage 4)")
    print("=" * 98)


def main():
    parser = argparse.ArgumentParser(description="Micro-Profile Stage 4 Sub-Operations")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--K", type=int, default=128)
    args = parser.parse_args()

    print("=" * 98)
    print("      DEEP MICRO-PROFILING: STAGE 4 (EMBEDDING, POSITIONAL & MLP INJECTIONS)")
    print("=" * 98)
    print(f">> Device        : {args.device} ({torch.cuda.get_device_name(0)})")
    print(f">> Image Size    : {args.image_size}x{args.image_size}")
    print(f">> Timed Runs    : {args.runs} iterations ({args.warmup} warmup)")
    print("=" * 98)

    for b in args.batch_sizes:
        benchmark_stage4_substages(
            batch_size=b,
            runs=args.runs,
            warmup=args.warmup,
            image_size=args.image_size,
            K=args.K,
            device=args.device
        )


if __name__ == "__main__":
    measure_substages()
    main()
