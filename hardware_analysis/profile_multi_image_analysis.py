import os
import sys
import argparse
import time
import math
import gc
from typing import Dict, List, Tuple, Optional

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

from models.baseline_vit_det import BaselineViTDetector
from models.sparse_vit_det import SparseViTDetector
from hardware_analysis.optimized_samplers import MaxPoolFixationSampler


def generate_diverse_batch(batch_size: int, image_size: int, device: str, seed: int) -> torch.Tensor:
    """
    Generates a diverse synthetic image batch with varied spatial structures,
    contrasts, and Gaussian blobs (simulating real foreground objects).
    This forces distinct saliency/objectness peak landscapes on every run.
    """
    torch.manual_seed(seed)
    H = W = image_size
    
    # 1. Base texture (different frequency / contrast)
    base = torch.randn(batch_size, 3, H, W, device=device) * 0.2 + 0.5
    
    # 2. Inject random foreground Gaussian blobs to mimic real object saliency
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    num_blobs = np.random.randint(2, 6)
    for _ in range(num_blobs):
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        sigma = np.random.uniform(0.1, 0.35)
        blob = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        base += blob.unsqueeze(0).unsqueeze(0) * np.random.uniform(0.3, 0.8)

    return torch.clamp(base, 0.0, 1.0)


@torch.no_grad()
def profile_dense_stages_multi_images(
    model: BaselineViTDetector,
    batch_size: int = 1,
    image_size: int = 512,
    device: str = "cuda",
    warmup: int = 5,
    num_images: int = 30
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Measures latency for BaselineViTDetector across `num_images` distinct input batches.
    Gracefully catches Out-Of-Memory errors on memory-constrained GPUs.
    """
    model.eval()
    H = W = image_size
    
    try:
        torch.cuda.empty_cache()
        gc.collect()

        # Warmup
        for w in range(warmup):
            dummy = generate_diverse_batch(batch_size, H, device, seed=1000 + w)
            _ = model(dummy)
        torch.cuda.synchronize()

        stage_times = {
            "1. Patch Embed Conv2d (1024 patches)": [],
            "2. Positional Embedding Interpolation": [],
            "3. Transformer Blocks (12 Layers, 1024 tokens)": [],
            "4. Detection Head (1024 tokens)": [],
            "Total Pipeline (End-to-End)": []
        }

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_patch = torch.cuda.Event(enable_timing=True)
        ev_pos = torch.cuda.Event(enable_timing=True)
        ev_trans = torch.cuda.Event(enable_timing=True)
        ev_head = torch.cuda.Event(enable_timing=True)

        for run_idx in range(num_images):
            images = generate_diverse_batch(batch_size, H, device, seed=2000 + run_idx)

            ev_start.record()

            # 1. Standard dense patch embedding
            tokens = model.vit.patch_embed.proj(images)
            tokens = tokens.flatten(2).transpose(1, 2)
            if hasattr(model.vit.patch_embed, 'norm') and model.vit.patch_embed.norm is not None:
                tokens = model.vit.patch_embed.norm(tokens)
            ev_patch.record()

            # 2. Add interpolated 2D positional embeddings
            pos_embed = model.interpolate_pos_embed(tokens)
            tokens = tokens + pos_embed
            tokens = model.vit.pos_drop(tokens)
            ev_pos.record()

            # 3. Dense Transformer Blocks (1024 tokens)
            for block in model.vit.blocks:
                tokens = block(tokens)
            tokens = model.vit.norm(tokens)
            ev_trans.record()

            # 4. Dense Detection Head
            norm_coords = model.get_dense_grid_coords(batch_size, images.device)
            pred_logits, pred_boxes = model.detection_head(tokens, norm_coords)
            ev_head.record()

            torch.cuda.synchronize()

            stage_times["1. Patch Embed Conv2d (1024 patches)"].append(ev_start.elapsed_time(ev_patch))
            stage_times["2. Positional Embedding Interpolation"].append(ev_patch.elapsed_time(ev_pos))
            stage_times["3. Transformer Blocks (12 Layers, 1024 tokens)"].append(ev_pos.elapsed_time(ev_trans))
            stage_times["4. Detection Head (1024 tokens)"].append(ev_trans.elapsed_time(ev_head))
            stage_times["Total Pipeline (End-to-End)"].append(ev_start.elapsed_time(ev_head))

        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in stage_times.items()}

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return None
        raise e


@torch.no_grad()
def profile_stages_multi_images(
    model: SparseViTDetector,
    batch_size: int = 12,
    image_size: int = 512,
    device: str = "cuda",
    warmup: int = 10,
    num_images: int = 50
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Measures latency (mean +/- std in ms) across `num_images` distinct input batches.
    Each trial generates a fresh image batch to ensure cold cache and diverse saliency maps.
    """
    model.eval()
    H = W = image_size

    try:
        torch.cuda.empty_cache()
        gc.collect()

        # Warmup with dummy batches
        for w in range(warmup):
            dummy = generate_diverse_batch(batch_size, H, device, seed=1000 + w)
            _ = model(dummy)
        torch.cuda.synchronize()

        stage_times = {
            "1. CNN Feature Extractor (MobileNetV3)": [],
            "2. Fixation Sampler": [],
            "3. Local Patch Extractor (grid_sample)": [],
            "4. Patch Embedding & Pos/Score Proj": [],
            "   ├─ 4a. Pure Patch Embed (Conv2d + LN)": [],
            "   ├─ 4b. Bicubic Pos Grid Interpolation": [],
            "   ├─ 4c. 2D Fourier Positional MLP": [],
            "   ├─ 4d. Fixation Score MLP": [],
            "   └─ 4e. Global Token Proj & Assembly": [],
            "5. Transformer Blocks (12 Layers)": [],
            "6. Detection Head": [],
            "Total Pipeline (End-to-End)": []
        }

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_cnn = torch.cuda.Event(enable_timing=True)
        ev_sampler = torch.cuda.Event(enable_timing=True)
        ev_patch_ext = torch.cuda.Event(enable_timing=True)
        ev_s4_patch = torch.cuda.Event(enable_timing=True)
        ev_s4_bicubic = torch.cuda.Event(enable_timing=True)
        ev_s4_fourier = torch.cuda.Event(enable_timing=True)
        ev_s4_score = torch.cuda.Event(enable_timing=True)
        ev_embed = torch.cuda.Event(enable_timing=True)
        ev_transformer = torch.cuda.Event(enable_timing=True)
        ev_head = torch.cuda.Event(enable_timing=True)

        for run_idx in range(num_images):
            # Generate a FRESH image batch for each run
            images = generate_diverse_batch(batch_size, H, device, seed=2000 + run_idx)

            ev_start.record()

            # Stage 1: CNN Feature Extractor
            X, S, O = model.feature_extractor(images)
            ev_cnn.record()

            # Stage 2: Fixation Sampler
            coords, norm_coords, fixation_scores, F_map = model.fixation_sampler(S, O, image_size=(H, W))
            ev_sampler.record()

            # Stage 3: Patch Extractor (grid_sample)
            patches = model.patch_extractor(images, coords)
            ev_patch_ext.record()

            # Stage 4: Patch Embedding & Pos/Score Injections + Global Token Projection
            # 4a. Pure Patch Embedding Conv2d + LayerNorm
            foveal_tokens = model.extract_patch_embeddings(patches)
            ev_s4_patch.record()

            # 4b. Bicubic Positional Grid Interpolation
            grid_pos_embed = model.vit.pos_embed[:, 1:, :]
            r_pos = model.pos_interpolator(norm_coords, grid_pos_embed)
            grid_embed = grid_pos_embed.view(1, model.grid_size, model.grid_size, model.embed_dim).permute(0, 3, 1, 2)
            sample_grid_x = 2.0 * norm_coords[..., 0] - 1.0
            sample_grid_y = 2.0 * norm_coords[..., 1] - 1.0
            K_curr = norm_coords.shape[1]
            sample_grid = torch.stack([sample_grid_x, sample_grid_y], dim=-1).view(batch_size, K_curr, 1, 2)
            grid_embed_expanded = grid_embed.expand(batch_size, model.embed_dim, model.grid_size, model.grid_size)
            interpolated = torch.nn.functional.grid_sample(
                grid_embed_expanded,
                sample_grid,
                mode='bicubic',
                padding_mode='border',
                align_corners=False
            ).squeeze(-1).permute(0, 2, 1)
            ev_s4_bicubic.record()

            # 4c. 2D Fourier Frequency Positional MLP
            fourier_pos = model.pos_interpolator.fourier_encoder(norm_coords)
            r_pos = interpolated + model.pos_interpolator.alpha * fourier_pos
            ev_s4_fourier.record()

            # 4d. Fixation Score Embedding MLP
            score_tokens = model.score_embed(fixation_scores)
            foveal_tokens = foveal_tokens + r_pos + score_tokens
            ev_s4_score.record()

            # 4e. Global Token Proj & Assembly
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

            stage_times["1. CNN Feature Extractor (MobileNetV3)"].append(ev_start.elapsed_time(ev_cnn))
            stage_times["2. Fixation Sampler"].append(ev_cnn.elapsed_time(ev_sampler))
            stage_times["3. Local Patch Extractor (grid_sample)"].append(ev_sampler.elapsed_time(ev_patch_ext))
            stage_times["4. Patch Embedding & Pos/Score Proj"].append(ev_patch_ext.elapsed_time(ev_embed))
            stage_times["   ├─ 4a. Pure Patch Embed (Conv2d + LN)"].append(ev_patch_ext.elapsed_time(ev_s4_patch))
            stage_times["   ├─ 4b. Bicubic Pos Grid Interpolation"].append(ev_s4_patch.elapsed_time(ev_s4_bicubic))
            stage_times["   ├─ 4c. 2D Fourier Positional MLP"].append(ev_s4_bicubic.elapsed_time(ev_s4_fourier))
            stage_times["   ├─ 4d. Fixation Score MLP"].append(ev_s4_fourier.elapsed_time(ev_s4_score))
            stage_times["   └─ 4e. Global Token Proj & Assembly"].append(ev_s4_score.elapsed_time(ev_embed))
            stage_times["5. Transformer Blocks (12 Layers)"].append(ev_embed.elapsed_time(ev_transformer))
            stage_times["6. Detection Head"].append(ev_transformer.elapsed_time(ev_head))
            stage_times["Total Pipeline (End-to-End)"].append(ev_start.elapsed_time(ev_head))

        # Return (mean, std) for each stage
        return {
            k: (float(np.mean(v)), float(np.std(v)))
            for k, v in stage_times.items()
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return None
        raise e


def main():
    parser = argparse.ArgumentParser(description="Multi-Image Statistical GPU Latency Profiler with Dense ViT")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-images", type=int, default=30, help="Number of distinct image batches to evaluate")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 115)
    print("      STATISTICAL GPU BENCHMARK: DENSE ViT vs. ORIGINAL SPARSE-ViT vs. METHOD 2")
    print("=" * 115)
    print(f">> Device        : {device} ({torch.cuda.get_device_name(0)})")
    print(f">> Image Size    : {args.image_size}x{args.image_size}")
    print(f">> ViT Backbone  : {args.backbone}")
    print(f">> Distinct Runs : {args.num_images} fresh image batches per test ({args.warmup} warmup)")
    print("=" * 115)

    # 1. Initialize Dense Baseline ViT
    print(">> Initializing Normal Dense ViT (Baseline)...")
    model_dense = BaselineViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, image_size=(args.image_size, args.image_size)
    ).to(device)

    # 2. Initialize Original Sparse-ViT
    print(">> Initializing Original Sparse-ViT (K=128)...")
    model_orig = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)

    # 3. Initialize Method 2 (MaxPool NMS)
    print(">> Initializing Method 2 (MaxPool NMS) Sparse-ViT...")
    model_maxpool = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)
    model_maxpool.fixation_sampler = MaxPoolFixationSampler(
        K=128, patch_size=16, image_size=(args.image_size, args.image_size)
    ).to(device)

    test_batches = [1, 4, 8, 12]

    for B in test_batches:
        print("\n" + "#" * 115)
        print(f"                       BATCH SIZE = {B:2d} ({args.num_images} DISTINCT IMAGE BATCHES)")
        print("#" * 115)

        stats_dense = profile_dense_stages_multi_images(
            model_dense, batch_size=B, image_size=args.image_size, device=args.device,
            warmup=max(3, args.warmup // 2), num_images=args.num_images
        )
        stats_orig = profile_stages_multi_images(
            model_orig, batch_size=B, image_size=args.image_size, device=args.device,
            warmup=args.warmup, num_images=args.num_images
        )
        stats_pool = profile_stages_multi_images(
            model_maxpool, batch_size=B, image_size=args.image_size, device=args.device,
            warmup=args.warmup, num_images=args.num_images
        )

        # -------------------------------------------------------------
        # 1. Executive Summary Table for this Batch Size
        # -------------------------------------------------------------
        print("\n>> [Executive Summary: End-to-End Latency & Throughput]")
        print(f"{'Model Architecture':<36} | {'Tokens':<8} | {'Latency (Mean +/- Std)':<24} | {'Throughput (FPS)':<18} | {'Speedup vs Dense'}")
        print("-" * 115)

        dense_ms = stats_dense["Total Pipeline (End-to-End)"][0] if stats_dense is not None else None
        
        # Dense Row
        if stats_dense is not None:
            m, s = stats_dense["Total Pipeline (End-to-End)"]
            fps = B / (m / 1000.0)
            print(f"{'Normal Dense ViT (Baseline)':<36} | {'1024':<8} | {m:6.2f} +/- {s:4.2f} ms         | {fps:5.1f} FPS          | 1.00x (Baseline)")
        else:
            print(f"{'Normal Dense ViT (Baseline)':<36} | {'1024':<8} | {'OOM (Out-of-Memory)':<24} | {'---':<18} | OOM (>4GB VRAM)")

        # Original Sparse Row
        if stats_orig is not None:
            m_orig, s_orig = stats_orig["Total Pipeline (End-to-End)"]
            fps_orig = B / (m_orig / 1000.0)
            sp_orig = f"{dense_ms / m_orig:.2f}x" if dense_ms else "N/A (Dense OOM)"
            print(f"{'Original Sparse-ViT (K=128)':<36} | {'144':<8} | {m_orig:6.2f} +/- {s_orig:4.2f} ms         | {fps_orig:5.1f} FPS          | {sp_orig}")

        # Method 2 Row
        if stats_pool is not None:
            m_pool, s_pool = stats_pool["Total Pipeline (End-to-End)"]
            fps_pool = B / (m_pool / 1000.0)
            sp_pool = f"{dense_ms / m_pool:.2f}x" if dense_ms else "N/A (Dense OOM)"
            print(f"{'Method 2 MaxPool Sparse-ViT':<36} | {'144':<8} | {m_pool:6.2f} +/- {s_pool:4.2f} ms         | {fps_pool:5.1f} FPS          | {sp_pool}")

        # -------------------------------------------------------------
        # 2. Granular Stage Breakdown Table
        # -------------------------------------------------------------
        print("\n>> [Granular Stage Breakdown: Sparse-ViT Models]")
        print(f"{'Pipeline Stage':<40} | {'Original (Mean +/- Std)':<23} | {'Method 2 (Mean +/- Std)':<23} | {'Delta (ms)':<10} | {'Status'}")
        print("-" * 115)
        print(f"{'Pipeline Stage':<44} | {'Original (Mean +/- Std)':<23} | {'Method 2 (Mean +/- Std)':<23} | {'Delta (ms)':<10} | {'Status'}")
        print("-" * 122)

        for stage_name in stats_orig.keys():
            m_o, s_o = stats_orig[stage_name]
            m_p, s_p = stats_pool[stage_name]
            delta = m_o - m_p

            orig_str = f"{m_o:6.2f} +/- {s_o:4.2f} ms"
            pool_str = f"{m_p:6.2f} +/- {s_p:4.2f} ms"

            if "Transformer" in stage_name:
                status = f"Diff: {abs(delta):.2f} ms (Identical within 1-sigma)"
            elif "Sampler" in stage_name:
            elif "2. Fixation Sampler" in stage_name:
                speedup = m_o / max(m_p, 1e-4)
                status = f"{speedup:.1f}x faster sampler"
            elif "Total" in stage_name:
                speedup = m_o / max(m_p, 1e-4)
                status = f"Speedup: {speedup:.2f}x"
            elif "├─" in stage_name or "└─" in stage_name:
                s4_orig = stats_orig["4. Patch Embedding & Pos/Score Proj"][0]
                pct = (m_o / max(s4_orig, 1e-4)) * 100.0
                status = f"{pct:5.1f}% of Stage 4"
            else:
                status = f"Diff: {abs(delta):.2f} ms"

            print(f"{stage_name:<40} | {orig_str:<23} | {pool_str:<23} | {delta:+6.2f} ms  | {status}")
            print(f"{stage_name:<44} | {orig_str:<23} | {pool_str:<23} | {delta:+6.2f} ms  | {status}")

    print("\n" + "=" * 115)
    print(">> Multi-image statistical benchmark completed.")
    print("=" * 115)


if __name__ == "__main__":
    main()
