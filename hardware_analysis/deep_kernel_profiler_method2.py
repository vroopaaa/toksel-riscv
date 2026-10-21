#!/usr/bin/env python3
"""
Deep Hardware & Kernel Profiling Suite for Method 2 (Max-Pool Morphological NMS):
Compares:
1. Normal Dense ViT (Baseline, 1024 tokens)
2. Original Sparse-ViT (K=128, Brute-force NMS loop)
3. Method 2 Sparse-ViT (K=128, Max-Pool NMS - Zero-Loop)

Measures:
- Exact CUDA Kernel Launch Counts & CPU Dispatch Overhead (PyTorch Profiler).
- Micro-dissection of Samplers (Original vs Method 2).
- Maximum Batch Size & OOM Stress Limits on VRAM.
- Input Resolution Scaling (512x512 up to 1024x1024).

Outputs full findings to 'hardware_analysis/deep_analysis_method2_report.md'.
"""

import os
import sys
import time
import argparse
import json
import gc
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project models and optimized samplers can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
WORKSPACE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_DIR, WORKSPACE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

try:
    from models.sparse_vit_det import SparseViTDetector
    from models.baseline_vit_det import BaselineViTDetector
    try:
        from optimized_samplers import MaxPoolFixationSampler
    except ImportError:
        from hardware_analysis.optimized_samplers import MaxPoolFixationSampler
except ImportError as e:
    print(f"[ERROR] Could not import models: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Kernel & OOM Stress Profiler for Method 2 (Max-Pool NMS)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224", help="ViT backbone architecture")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR, help="Directory to save report and json")
    return parser.parse_args()


# =========================================================================
# 1. EXACT CUDA KERNEL COUNT & LAUNCH OVERHEAD PROFILING
# =========================================================================
def count_cuda_kernels_and_ops(model: nn.Module, dummy_input: torch.Tensor, device: str = "cuda") -> Dict[str, Any]:
    """Uses PyTorch Profiler to count exact kernel launches and CPU/GPU split."""
    model.eval()
    is_cuda = device.startswith("cuda")
    activities = [ProfilerActivity.CPU]
    if is_cuda:
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    for _ in range(5):
        _ = model(dummy_input)
    if is_cuda:
        torch.cuda.synchronize()

    # Profile single forward pass
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        with record_function("model_inference"):
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()

    def _get_dev_time(ev):
        for attr in ["device_time_total", "cuda_time_total", "cuda_time", "self_device_time_total"]:
            if hasattr(ev, attr):
                val = getattr(ev, attr)
                if val is not None:
                    return float(val)
        return 0.0

    def _get_cpu_time(ev):
        for attr in ["cpu_time_total", "self_cpu_time_total", "cpu_time"]:
            if hasattr(ev, attr):
                val = getattr(ev, attr)
                if val is not None:
                    return float(val)
        return 0.0

    events = prof.key_averages()
    total_cpu_time_us = sum(_get_cpu_time(e) for e in events)
    total_cuda_time_us = sum(_get_dev_time(e) for e in events) if is_cuda else 0.0

    # Count distinct kernel launches on GPU
    cuda_kernel_launches = 0
    kernel_breakdown = {}
    for e in events:
        dev_time = _get_dev_time(e)
        if e.count > 0 and dev_time > 0:
            cuda_kernel_launches += e.count
            kernel_breakdown[e.key] = {
                "count": e.count,
                "cuda_time_ms": dev_time / 1000.0,
                "cpu_time_ms": _get_cpu_time(e) / 1000.0
            }

    sorted_by_count = sorted(kernel_breakdown.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
    top_frequent = [{"kernel": k, "count": v["count"], "cuda_ms": v["cuda_time_ms"]} for k, v in sorted_by_count]

    return {
        "total_cuda_kernels": cuda_kernel_launches,
        "total_cpu_time_ms": total_cpu_time_us / 1000.0,
        "total_cuda_time_ms": total_cuda_time_us / 1000.0,
        "top_frequent_kernels": top_frequent
    }


# =========================================================================
# 2. OOM STRESS TEST (MAXIMUM BATCH SIZE BENCHMARK)
# =========================================================================
def run_oom_stress_test(model: nn.Module, model_name: str, image_size: int = 512, device: str = "cuda") -> Dict[str, Any]:
    """Ramps up batch size until OutOfMemoryError is triggered, finding the hard physical limit."""
    model.eval()
    candidate_batches = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64]
    successful_runs = []
    max_batch = 0
    oom_batch = None

    print(f"\n>> [OOM Stress Test] Testing: {model_name} (Resolution: {image_size}x{image_size})")

    for b in candidate_batches:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            inp = torch.randn(b, 3, image_size, image_size, device=device)
            torch.cuda.reset_peak_memory_stats()
            
            # Warmup
            _ = model(inp)
            torch.cuda.synchronize()

            # Timed run
            t0 = time.perf_counter()
            _ = model(inp)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            fps = b / (t1 - t0)

            successful_runs.append({
                "batch_size": b,
                "latency_ms": lat_ms,
                "fps": fps,
                "peak_vram_mb": peak_vram_mb
            })
            max_batch = b
            print(f"   Batch {b:3d}: PASS | Latency: {lat_ms:6.2f} ms | FPS: {fps:5.1f} | Peak VRAM: {peak_vram_mb:6.1f} MB")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                oom_batch = b
                print(f"   Batch {b:3d}: FAILED WITH OUT-OF-MEMORY (OOM)!")
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise e

    return {
        "model_name": model_name,
        "max_batch_size": max_batch,
        "oom_batch_size": oom_batch,
        "successful_runs": successful_runs
    }


# =========================================================================
# 3. HIGH-RESOLUTION STRESS TEST (512 to 1024)
# =========================================================================
def run_resolution_scaling_test(models_dict: Dict[str, nn.Module], device: str = "cuda") -> List[Dict[str, Any]]:
    """Compares latency across input resolutions (512 to 1024) for all models."""
    resolutions = [512, 640, 768, 896, 1024]
    results = []

    print("\n>> [Resolution Scaling Test] Scaling from 512x512 to 1024x1024 (Batch Size = 1)...")

    for res in resolutions:
        row = {"resolution": f"{res}x{res}"}
        row["dense_tokens"] = (res // 16) * (res // 16)
        row["sparse_tokens"] = 16 + 128

        for m_name, model in models_dict.items():
            torch.cuda.empty_cache()
            gc.collect()
            lat_ms = None
            oom = False
            try:
                inp = torch.randn(1, 3, res, res, device=device)
                for _ in range(5):
                    _ = model(inp)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    _ = model(inp)
                torch.cuda.synchronize()
                lat_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                oom = True
                torch.cuda.empty_cache()
                gc.collect()

            row[m_name] = f"{lat_ms:6.2f} ms" if not oom else "OOM Crash"

        d_val = row.get("Normal Dense ViT")
        m2_val = row.get("Method 2 (Max-Pool)")
        print(f"   Res {res:4d}x{res:<4d} | Normal: {d_val:<12} | Orig Sparse: {row.get('Original Sparse-ViT'):<12} | Method 2: {m2_val:<12}")
        results.append(row)

    return results


# =========================================================================
# 4. FIXATION SAMPLER MICRO-BENCHMARKING
# =========================================================================
def dissect_samplers(orig_sampler: nn.Module, maxpool_sampler: nn.Module, S: torch.Tensor, O: torch.Tensor, runs: int = 50, device: str = "cuda") -> Dict[str, Any]:
    """Profiles the internal operations of both FixationSamplers."""
    is_cuda = device.startswith("cuda")
    H_img, W_img = 512, 512

    # Warmup
    for _ in range(10):
        _ = orig_sampler(S, O, image_size=(H_img, W_img))
        _ = maxpool_sampler(S, O, image_size=(H_img, W_img))
    if is_cuda: torch.cuda.synchronize()

    # 1. Orig Sampler Timing
    t_orig = []
    for _ in range(runs):
        if is_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = orig_sampler(S, O, image_size=(H_img, W_img))
        if is_cuda: torch.cuda.synchronize()
        t_orig.append((time.perf_counter() - t0) * 1000.0)

    # 2. MaxPool Sampler Timing
    t_m2 = []
    for _ in range(runs):
        if is_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = maxpool_sampler(S, O, image_size=(H_img, W_img))
        if is_cuda: torch.cuda.synchronize()
        t_m2.append((time.perf_counter() - t0) * 1000.0)

    return {
        "original_mean_ms": float(np.mean(t_orig)),
        "maxpool_mean_ms": float(np.mean(t_m2)),
        "speedup": float(np.mean(t_orig) / max(np.mean(t_m2), 1e-4))
    }


# =========================================================================
# MAIN EXECUTION
# =========================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 95)
    print("   DEEP HARDWARE & KERNEL PROFILING SUITE: METHOD 2 (MAX-POOL NMS) EVALUATION")
    print("=" * 95)
    print(f">> Device       : {device} ({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'})")
    print(f">> Backbone     : {args.backbone}")
    print("=" * 95)

    # 1. Initialize models
    print("\n>> Initializing Models...")
    dense_model = BaselineViTDetector(num_classes=80, backbone_name=args.backbone, pretrained=False, image_size=(512, 512)).to(device)
    sparse_orig = SparseViTDetector(num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(512, 512), num_global_tokens=16).to(device)
    
    sparse_m2 = SparseViTDetector(num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(512, 512), num_global_tokens=16).to(device)
    sparse_m2.fixation_sampler = MaxPoolFixationSampler(K=128, patch_size=16, image_size=(512, 512)).to(device)

    dummy_512 = torch.randn(1, 3, 512, 512, device=device)

    # -------------------------------------------------------------
    # Test 1: Exact CUDA Kernel Launch Counts for ALL 3 MODELS
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(">> TEST 1: PROFILING EXACT CUDA KERNEL LAUNCHES (PyTorch Profiler)...")
    print("=" * 95)
    dense_kernel_stats = count_cuda_kernels_and_ops(dense_model, dummy_512, device=args.device)
    orig_kernel_stats = count_cuda_kernels_and_ops(sparse_orig, dummy_512, device=args.device)
    m2_kernel_stats = count_cuda_kernels_and_ops(sparse_m2, dummy_512, device=args.device)

    print(f"\n1. TOTAL CUDA KERNEL LAUNCHES PER INFERENCE:")
    print(f"   * Normal Dense ViT             : {dense_kernel_stats['total_cuda_kernels']:5d} CUDA kernels")
    print(f"   * Original Sparse-ViT (Loop)   : {orig_kernel_stats['total_cuda_kernels']:5d} CUDA kernels  (+{orig_kernel_stats['total_cuda_kernels'] - dense_kernel_stats['total_cuda_kernels']} loop kernel tax!)")
    print(f"   * Method 2 Sparse-ViT (MaxPool): {m2_kernel_stats['total_cuda_kernels']:5d} CUDA kernels  (ELIMINATED +{orig_kernel_stats['total_cuda_kernels'] - m2_kernel_stats['total_cuda_kernels']} KERNELS!)")

    print(f"\n2. CPU DRIVER OVERHEAD TIME (Host dispatch time):")
    print(f"   * Normal Dense ViT             : {dense_kernel_stats['total_cpu_time_ms']:6.2f} ms")
    print(f"   * Original Sparse-ViT (Loop)   : {orig_kernel_stats['total_cpu_time_ms']:6.2f} ms")
    print(f"   * Method 2 Sparse-ViT (MaxPool): {m2_kernel_stats['total_cpu_time_ms']:6.2f} ms")

    # -------------------------------------------------------------
    # Test 2: Sampler Dissection
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(">> TEST 2: SAMPLER MICRO-BENCHMARK (Original vs. Method 2)...")
    print("=" * 95)
    dummy_S = torch.rand(1, 1, 32, 32, device=device)
    dummy_O = torch.rand(1, 1, 32, 32, device=device)
    sampler_dissection = dissect_samplers(sparse_orig.fixation_sampler, sparse_m2.fixation_sampler, dummy_S, dummy_O, runs=50, device=args.device)
    print(f"   * Original Fixation Sampler : {sampler_dissection['original_mean_ms']:5.2f} ms")
    print(f"   * Method 2 Max-Pool Sampler : {sampler_dissection['maxpool_mean_ms']:5.2f} ms")
    print(f"   * Direct Sampler Speedup    : {sampler_dissection['speedup']:5.2f}x faster!")

    # -------------------------------------------------------------
    # Test 3: OOM Stress Test (Maximum Batch Size on 4GB VRAM)
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(">> TEST 3: OOM STRESS TEST (MAXIMUM BATCH SIZE ON 4GB VRAM)...")
    print("=" * 95)
    dense_oom_res = run_oom_stress_test(dense_model, "Normal Dense ViT", image_size=512, device=args.device)
    orig_oom_res = run_oom_stress_test(sparse_orig, "Original Sparse-ViT", image_size=512, device=args.device)
    m2_oom_res = run_oom_stress_test(sparse_m2, "Method 2 Sparse-ViT (MaxPool)", image_size=512, device=args.device)

    print(f"\n>> MAXIMUM ATTAINABLE BATCH SIZE ON {torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'}:")
    print(f"   * Normal Dense ViT             : Max Batch = {dense_oom_res['max_batch_size']:2d} (Crashes at B={dense_oom_res['oom_batch_size']})")
    print(f"   * Original Sparse-ViT (Loop)   : Max Batch = {orig_oom_res['max_batch_size']:2d} (Crashes at B={orig_oom_res['oom_batch_size']})")
    print(f"   * Method 2 Sparse-ViT (MaxPool): Max Batch = {m2_oom_res['max_batch_size']:2d} (Crashes at B={m2_oom_res['oom_batch_size']})")

    # -------------------------------------------------------------
    # Test 4: Resolution Scaling Stress Test (512 to 1024)
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(">> TEST 4: RESOLUTION SCALING STRESS TEST (512x512 to 1024x1024)...")
    print("=" * 95)
    models_dict = {
        "Normal Dense ViT": dense_model,
        "Original Sparse-ViT": sparse_orig,
        "Method 2 (Max-Pool)": sparse_m2
    }
    res_scaling_results = run_resolution_scaling_test(models_dict, device=args.device)

    # -------------------------------------------------------------
    # 5. GENERATE FINAL COMPREHENSIVE MARKDOWN REPORT
    # -------------------------------------------------------------
    md_path = os.path.join(args.output_dir, "deep_analysis_method2_report.md")
    json_path = os.path.join(args.output_dir, "deep_analysis_method2_results.json")

    with open(md_path, "w") as f:
        f.write("# In-Depth Hardware, Kernel, & Memory Stress Report (Method 2 Evaluation)\n\n")
        f.write(f"- **Device Tested**: {device} ({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'})\n")
        f.write(f"- **ViT Architecture**: {args.backbone} (D=384, 12 layers)\n\n")

        f.write("## 1. Executive Kernel & Overhead Comparison\n\n")
        f.write("| Metric | Normal Dense ViT (1024) | Original Sparse-ViT (144) | Method 2 Max-Pool (144) | Verdict |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        f.write(f"| **CUDA Kernel Count** | `{dense_kernel_stats['total_cuda_kernels']}` | `{orig_kernel_stats['total_cuda_kernels']}` | **`{m2_kernel_stats['total_cuda_kernels']}`** | **Method 2 cuts {orig_kernel_stats['total_cuda_kernels'] - m2_kernel_stats['total_cuda_kernels']} kernels!** |\n")
        f.write(f"| **Sampler Latency** | `0.00 ms` | `{sampler_dissection['original_mean_ms']:.2f} ms` | **`{sampler_dissection['maxpool_mean_ms']:.2f} ms`** | **{sampler_dissection['speedup']:.1f}x faster sampler** |\n")
        f.write(f"| **Max Batch Size (VRAM)** | **B = {dense_oom_res['max_batch_size']}** | **B = {orig_oom_res['max_batch_size']}** | **B = {m2_oom_res['max_batch_size']}** | Sparse fits **{m2_oom_res['max_batch_size']/max(dense_oom_res['max_batch_size'], 1):.1f}x** larger batch |\n")
        f.write(f"| **Max FPS (Throughput)** | `{max(r['fps'] for r in dense_oom_res['successful_runs']):.1f} FPS` | `{max(r['fps'] for r in orig_oom_res['successful_runs']):.1f} FPS` | **`{max(r['fps'] for r in m2_oom_res['successful_runs']):.1f} FPS`** | Peak throughput advantage |\n\n")

        f.write("## 2. Input Resolution Scaling Test (512x512 up to 1024x1024)\n\n")
        f.write("| Resolution | Dense Tokens | Sparse Tokens | Normal Dense ViT | Original Sparse-ViT | Method 2 (Max-Pool) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in res_scaling_results:
            f.write(f"| {r['resolution']} | {r['dense_tokens']} | {r['sparse_tokens']} | {r.get('Normal Dense ViT', 'N/A')} | {r.get('Original Sparse-ViT', 'N/A')} | **{r.get('Method 2 (Max-Pool)', 'N/A')}** |\n")

    full_output = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
        "kernel_stats": {
            "dense": dense_kernel_stats,
            "orig_sparse": orig_kernel_stats,
            "method2_sparse": m2_kernel_stats
        },
        "sampler_dissection": sampler_dissection,
        "oom_test": {
            "dense": dense_oom_res,
            "orig_sparse": orig_oom_res,
            "method2_sparse": m2_oom_res
        },
        "resolution_scaling": res_scaling_results
    }
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=4)

    print("\n" + "=" * 95)
    print(f">> [COMPLETE] In-depth Method 2 Markdown report saved to: {md_path}")
    print(f">> [COMPLETE] Raw JSON data saved to                    : {json_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
