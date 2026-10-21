#!/usr/bin/env python3
"""
Comprehensive Optimization Benchmark:
Compares:
1. Normal Dense ViT (Baseline) (1024 tokens)
2. Original Sparse-ViT (K=128) (Original 9.44 ms distance-matrix loop)
3. Optimized Sparse-ViT (Method 1: Direct 2D Grid Slicing) (User's concept)
4. Optimized Sparse-ViT (Method 2: Max-Pool Morphological NMS) (Zero-loop parallel kernel)

Does NOT modify any original codebase files.
"""

import os
import sys
import time
import argparse
import json
import numpy as np

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_DIR, WORKSPACE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

try:
    from models.sparse_vit_det import SparseViTDetector
    from models.baseline_vit_det import BaselineViTDetector
    try:
        from optimized_samplers import GridSlicingFixationSampler, MaxPoolFixationSampler
    except ImportError:
        from hardware_analysis.optimized_samplers import GridSlicingFixationSampler, MaxPoolFixationSampler
except ImportError as e:
    print(f"[ERROR] Could not import modules: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Original vs. Optimized Samplers and End-to-End ViT")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--image-size", type=int, default=512, help="Input resolution (default: 512)")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224", help="ViT backbone architecture")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of timing iterations")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR, help="Directory to save report and json")
    return parser.parse_args()


@torch.no_grad()
def measure_latency_and_fps(model: nn.Module, dummy_input: torch.Tensor, warmup: int = 25, runs: int = 100, device: str = "cuda"):
    model.eval()
    for _ in range(warmup):
        _ = model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        latencies = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
    else:
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    batch_size = dummy_input.shape[0]
    fps = float(batch_size / (mean_ms / 1000.0))
    return mean_ms, std_ms, fps


@torch.no_grad()
def benchmark_sampler_standalone(sampler_fn, S: torch.Tensor, O: torch.Tensor, warmup: int = 25, runs: int = 100, device: str = "cuda"):
    """Isolates and benchmarks purely the fixation sampler execution time."""
    for _ in range(warmup):
        _ = sampler_fn(S, O, image_size=(512, 512))

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        latencies = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = sampler_fn(S, O, image_size=(512, 512))
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
    else:
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = sampler_fn(S, O, image_size=(512, 512))
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return float(np.mean(latencies)), float(np.std(latencies))


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 105)
    print("      BENCHMARK SUITE: NORMAL ViT vs. ORIGINAL SPARSE-ViT vs. OPTIMIZED SAMPLERS")
    print("=" * 105)
    print(f">> Device       : {device} ({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'})")
    print(f">> Resolution   : {args.image_size} x {args.image_size}")
    print(f">> ViT Backbone : {args.backbone}")
    print(f">> Iterations   : {args.warmup} Warmup runs | {args.runs} Timed test runs")
    print("=" * 105)

    # 1. Instantiate the 4 Models
    print("\n>> Initializing Models...")
    # Baseline Normal ViT
    dense_model = BaselineViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, image_size=(args.image_size, args.image_size)
    ).to(device)

    # Original Sparse-ViT (K=128)
    sparse_orig = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)

    # Sparse-ViT with Method 1: Direct 2D Grid Slicing
    sparse_grid_slicing = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)
    sparse_grid_slicing.fixation_sampler = GridSlicingFixationSampler(
        K=128, patch_size=16, image_size=(args.image_size, args.image_size)
    ).to(device)

    # Sparse-ViT with Method 2: Max-Pool NMS (Zero-Loop)
    sparse_maxpool = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size)
    ).to(device)
    sparse_maxpool.fixation_sampler = MaxPoolFixationSampler(
        K=128, patch_size=16, image_size=(args.image_size, args.image_size)
    ).to(device)

    # -------------------------------------------------------------------------
    # PART 1: STANDALONE SAMPLER BENCHMARK
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print(">> [PART 1/3] STANDALONE FIXATION SAMPLER BENCHMARK (ISOLATED EXECUTION)")
    print("=" * 105)
    dummy_S = torch.rand(1, 1, 32, 32, device=device)
    dummy_O = torch.rand(1, 1, 32, 32, device=device)

    t_orig_ms, t_orig_std = benchmark_sampler_standalone(sparse_orig.fixation_sampler, dummy_S, dummy_O, args.warmup, args.runs, args.device)
    t_grid_ms, t_grid_std = benchmark_sampler_standalone(sparse_grid_slicing.fixation_sampler, dummy_S, dummy_O, args.warmup, args.runs, args.device)
    t_pool_ms, t_pool_std = benchmark_sampler_standalone(sparse_maxpool.fixation_sampler, dummy_S, dummy_O, args.warmup, args.runs, args.device)

    sampler_results = [
        {"name": "Original Sampler (1024x1024 dist_matrix)", "latency_ms": t_orig_ms, "std_ms": t_orig_std, "speedup": 1.00},
        {"name": "Method 1: Direct 2D Grid Slicing", "latency_ms": t_grid_ms, "std_ms": t_grid_std, "speedup": t_orig_ms / max(t_grid_ms, 1e-4)},
        {"name": "Method 2: Max-Pool NMS (Zero-Loop)", "latency_ms": t_pool_ms, "std_ms": t_pool_std, "speedup": t_orig_ms / max(t_pool_ms, 1e-4)}
    ]

    print(f"{'Sampler Algorithm':<46} | {'Latency (ms)':<15} | {'Speedup vs Original':<20}")
    print("-" * 90)
    for sr in sampler_results:
        print(f"{sr['name']:<46} | {sr['latency_ms']:5.2f} ± {sr['std_ms']:4.2f} ms | {sr['speedup']:5.2f}x faster")

    # -------------------------------------------------------------------------
    # PART 2: END-TO-END MODEL BENCHMARK (BATCH SIZE = 1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print(">> [PART 2/3] END-TO-END MODEL COMPARISON (BATCH SIZE = 1)")
    print("=" * 105)
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    all_models = [
        ("Normal Dense ViT (Baseline)", dense_model, 1024),
        ("Original Sparse-ViT (K=128)", sparse_orig, 144),
        ("Sparse-ViT (Method 1: 2D Grid Slicing)", sparse_grid_slicing, 144),
        ("Sparse-ViT (Method 2: Max-Pool NMS)", sparse_maxpool, 144)
    ]

    model_results = []
    dense_mean = None
    for name, m, seq_len in all_models:
        mean_ms, std_ms, fps = measure_latency_and_fps(m, dummy_input, args.warmup, args.runs, args.device)
        if dense_mean is None:
            dense_mean = mean_ms
            speedup_vs_dense = 1.00
        else:
            speedup_vs_dense = dense_mean / mean_ms

        model_results.append({
            "name": name,
            "tokens": seq_len,
            "latency_ms": mean_ms,
            "std_ms": std_ms,
            "fps": fps,
            "speedup_vs_dense": speedup_vs_dense
        })
        print(f"   {name:<40} : {mean_ms:5.2f} ± {std_ms:3.2f} ms ({fps:5.1f} FPS) -> Speedup: {speedup_vs_dense:4.2f}x vs Normal ViT")

    # -------------------------------------------------------------------------
    # PART 3: BATCH SIZE SCALING COMPARISON (B = 1, 4, 8, 16)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print(">> [PART 3/3] BATCH SIZE SCALING COMPARISON")
    print("=" * 105)
    batch_sizes = [1, 4, 8, 16]
    batch_data = {
        "Dense": [],
        "Original_Sparse": [],
        "Optimized_Method1": [],
        "Optimized_Method2": []
    }

    for b in batch_sizes:
        b_input = torch.randn(b, 3, args.image_size, args.image_size, device=device)
        d_ms, _, d_fps = measure_latency_and_fps(dense_model, b_input, 10, 25, args.device)
        so_ms, _, so_fps = measure_latency_and_fps(sparse_orig, b_input, 10, 25, args.device)
        sm1_ms, _, sm1_fps = measure_latency_and_fps(sparse_grid_slicing, b_input, 10, 25, args.device)
        sm2_ms, _, sm2_fps = measure_latency_and_fps(sparse_maxpool, b_input, 10, 25, args.device)

        batch_data["Dense"].append({"batch": b, "ms": d_ms, "fps": d_fps})
        batch_data["Original_Sparse"].append({"batch": b, "ms": so_ms, "fps": so_fps})
        batch_data["Optimized_Method1"].append({"batch": b, "ms": sm1_ms, "fps": sm1_fps})
        batch_data["Optimized_Method2"].append({"batch": b, "ms": sm2_ms, "fps": sm2_fps})

        print(f"   Batch {b:2d} | Normal: {d_ms:6.2f} ms ({d_fps:5.1f} FPS) | Orig Sparse: {so_ms:6.2f} ms | Method 1: {sm1_ms:6.2f} ms | Method 2: {sm2_ms:6.2f} ms ({sm2_fps:5.1f} FPS)")

    # -------------------------------------------------------------------------
    # FORMATTED REPORT GENERATION
    # -------------------------------------------------------------------------
    md_path = os.path.join(args.output_dir, "optimized_report.md")
    json_path = os.path.join(args.output_dir, "optimized_results.json")

    with open(md_path, "w") as f:
        f.write("# Optimization Benchmark Report: Original vs. Optimized Samplers\n\n")
        f.write(f"- **Device**: {device} ({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'})\n")
        f.write(f"- **Resolution**: {args.image_size}x{args.image_size}\n")
        f.write(f"- **ViT Backbone**: {args.backbone}\n\n")

        f.write("## 1. Standalone Fixation Sampler Latency Comparison\n\n")
        f.write("| Fixation Sampler Algorithm | Latency (ms) | Speedup vs Original | Description |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        f.write(f"| **Original Brute-Force** | {t_orig_ms:.2f} ± {t_orig_std:.2f} ms | 1.00x | 1024x1024 distance matrix + 128 loops |\n")
        f.write(f"| **Method 1: Direct 2D Grid Slicing** | {t_grid_ms:.2f} ± {t_grid_std:.2f} ms | **{t_orig_ms/max(t_grid_ms,1e-4):.2f}x** | Local 3x3 window zeroing on 2D grid |\n")
        f.write(f"| **Method 2: Max-Pool NMS (Zero-Loop)** | **{t_pool_ms:.2f} ± {t_pool_std:.2f} ms** | **{t_orig_ms/max(t_pool_ms,1e-4):.2f}x** | Single parallel max-pool kernel (No loops) |\n\n")

        f.write("## 2. End-to-End Model Latency & Speedup (Batch Size = 1)\n\n")
        f.write("| Model Architecture | Active Tokens | End-to-End Latency (ms) | Throughput (FPS) | Speedup vs. Normal ViT |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for mr in model_results:
            f.write(f"| **{mr['name']}** | {mr['tokens']} | {mr['latency_ms']:.2f} ± {mr['std_ms']:.2f} ms | {mr['fps']:.1f} FPS | **{mr['speedup_vs_dense']:.2f}x** |\n")

        f.write("\n## 3. Batch Size Scaling (Throughput & Latency)\n\n")
        f.write("| Batch Size | Normal ViT (ms / FPS) | Orig Sparse (ms / FPS) | Method 1 Grid Slicing (ms / FPS) | Method 2 Max-Pool (ms / FPS) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for i, b in enumerate(batch_sizes):
            d = batch_data["Dense"][i]
            so = batch_data["Original_Sparse"][i]
            sm1 = batch_data["Optimized_Method1"][i]
            sm2 = batch_data["Optimized_Method2"][i]
            f.write(f"| {b} | {d['ms']:.1f} ms ({d['fps']:.1f}) | {so['ms']:.1f} ms ({so['fps']:.1f}) | {sm1['ms']:.1f} ms ({sm1['fps']:.1f}) | **{sm2['ms']:.1f} ms ({sm2['fps']:.1f} FPS)** |\n")

    full_output = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
        "standalone_sampler": sampler_results,
        "end_to_end_models": model_results,
        "batch_scaling": batch_data
    }
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=4)

    print("\n" + "=" * 105)
    print(f">> [COMPLETE] Optimization report saved to: {md_path}")
    print(f">> [COMPLETE] Raw JSON data saved to      : {json_path}")
    print("=" * 105)


if __name__ == "__main__":
    main()
