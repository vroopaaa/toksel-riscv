#!/usr/bin/env python3
"""
Hardware & Latency Profiling Suite: Normal Dense ViT vs. Proposed Sparse-ViT.

Measures:
1. End-to-End Latency (ms) and Throughput (FPS) using microsecond CUDA Events.
2. Side-by-Side Stage-by-Stage Latency Breakdown (Normal ViT vs. Sparse-ViT).
3. Self-Attention FLOPs vs. Total ViT FLOPs.
4. Peak GPU VRAM Memory Allocation (MB).
5. Batch Size Scaling (B=1, 4, 8, 16).
6. Multi-Token Comparison: Normal Dense ViT (1024) vs Sparse-ViT (K=64, 128, 256).

Outputs formatted Markdown report and JSON results to 'hardware_analysis/'.
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure models module from project directory can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import torch
import torch.nn as nn

try:
    from models.sparse_vit_det import SparseViTDetector
    from models.baseline_vit_det import BaselineViTDetector
except ImportError as e:
    print(f"[ERROR] Could not import models from {PROJECT_DIR}: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Hardware Latency, FLOPs, and VRAM: Normal ViT vs. Sparse-ViT")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--image-size", type=int, default=512, help="Input resolution (H=W, default: 512)")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224", help="ViT backbone architecture")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of timing iterations")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR, help="Directory to save report and json")
    return parser.parse_args()


def calculate_vit_flops(sequence_length: int, embed_dim: int = 384, depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0):
    """Calculates theoretical FLOPs for ViT transformer blocks."""
    N = sequence_length
    D = embed_dim
    # Per layer:
    qkv_flops = 3 * 2 * N * (D ** 2)
    attn_matrix_flops = 2 * (N ** 2) * D
    attn_value_flops = 2 * (N ** 2) * D
    proj_flops = 2 * N * (D ** 2)
    mlp_flops = 2 * 2 * mlp_ratio * N * (D ** 2)

    layer_flops = qkv_flops + attn_matrix_flops + attn_value_flops + proj_flops + mlp_flops
    total_vit_flops = depth * layer_flops
    attn_only_flops = depth * (attn_matrix_flops + attn_value_flops)
    return total_vit_flops, attn_only_flops


@torch.no_grad()
def measure_end_to_end_latency(model: nn.Module, dummy_input: torch.Tensor, warmup: int = 20, runs: int = 100, device: str = "cuda") -> Tuple[float, float, float]:
    """Measures end-to-end latency (mean_ms, std_ms, fps) using CUDA Events."""
    model.eval()
    
    # Warmup runs
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
def profile_dense_vit_stages(model: BaselineViTDetector, dummy_input: torch.Tensor, warmup: int = 10, runs: int = 50, device: str = "cuda") -> Dict[str, float]:
    """Measures granular stage-by-stage latency inside Normal Dense ViT."""
    model.eval()
    B, _, H, W = dummy_input.shape

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings = {
        "1. Dense Patch Projection Conv2d": [],
        "2. Positional Embedding Interpolation": [],
        "3. Transformer Blocks (12 Dense Layers, 1024 tokens)": [],
        "4. Detection Head (Class & BBox for 1024 tokens)": []
    }

    is_cuda = device.startswith("cuda")

    for _ in range(runs):
        if is_cuda:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            torch.cuda.synchronize()

            # Stage 1: Patch Embed
            events[0].record()
            tokens = model.vit.patch_embed.proj(dummy_input)
            tokens = tokens.flatten(2).transpose(1, 2)
            if hasattr(model.vit.patch_embed, 'norm') and model.vit.patch_embed.norm is not None:
                tokens = model.vit.patch_embed.norm(tokens)
            events[1].record()

            # Stage 2: Pos Embed
            pos_embed = model.interpolate_pos_embed(tokens)
            tokens = tokens + pos_embed
            tokens = model.vit.pos_drop(tokens)
            events[2].record()

            # Stage 3: Dense Transformer Blocks
            for block in model.vit.blocks:
                tokens = block(tokens)
            tokens = model.vit.norm(tokens)
            events[3].record()

            # Stage 4: Dense Head
            norm_coords = model.get_dense_grid_coords(B, dummy_input.device)
            pred_logits, pred_boxes = model.detection_head(tokens, norm_coords)
            events[4].record()

            torch.cuda.synchronize()

            timings["1. Dense Patch Projection Conv2d"].append(events[0].elapsed_time(events[1]))
            timings["2. Positional Embedding Interpolation"].append(events[1].elapsed_time(events[2]))
            timings["3. Transformer Blocks (12 Dense Layers, 1024 tokens)"].append(events[2].elapsed_time(events[3]))
            timings["4. Detection Head (Class & BBox for 1024 tokens)"].append(events[3].elapsed_time(events[4]))
        else:
            t0 = time.perf_counter()
            tokens = model.vit.patch_embed.proj(dummy_input).flatten(2).transpose(1, 2)
            if hasattr(model.vit.patch_embed, 'norm') and model.vit.patch_embed.norm is not None:
                tokens = model.vit.patch_embed.norm(tokens)
            t1 = time.perf_counter()
            pos_embed = model.interpolate_pos_embed(tokens)
            tokens = model.vit.pos_drop(tokens + pos_embed)
            t2 = time.perf_counter()
            for block in model.vit.blocks:
                tokens = block(tokens)
            tokens = model.vit.norm(tokens)
            t3 = time.perf_counter()
            norm_coords = model.get_dense_grid_coords(B, dummy_input.device)
            pred_logits, pred_boxes = model.detection_head(tokens, norm_coords)
            t4 = time.perf_counter()

            timings["1. Dense Patch Projection Conv2d"].append((t1 - t0) * 1000)
            timings["2. Positional Embedding Interpolation"].append((t2 - t1) * 1000)
            timings["3. Transformer Blocks (12 Dense Layers, 1024 tokens)"].append((t3 - t2) * 1000)
            timings["4. Detection Head (Class & BBox for 1024 tokens)"].append((t4 - t3) * 1000)

    return {k: float(np.mean(v)) for k, v in timings.items()}


@torch.no_grad()
def profile_sparse_vit_stages(model: SparseViTDetector, dummy_input: torch.Tensor, warmup: int = 10, runs: int = 50, device: str = "cuda") -> Dict[str, float]:
    """Measures granular stage-by-stage latency inside SparseViTDetector."""
    model.eval()
    B, _, H, W = dummy_input.shape

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings = {
        "1. CNN Feature Extractor (MobileNetV3)": [],
        "2. Fixation Sampler (Greedy NMS Loop)": [],
        "3. Local Patch Extractor (grid_sample)": [],
        "4. Patch Embedding & Pos/Score Injections": [],
        "5. Global Token Projection": [],
        "6. Transformer Blocks (12 Sparse Layers, 144 tokens)": [],
        "7. Detection Head (Class & BBox for 128 tokens)": []
    }

    is_cuda = device.startswith("cuda")

    for _ in range(runs):
        if is_cuda:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
            torch.cuda.synchronize()

            # Stage 1: CNN Feature Extractor
            events[0].record()
            X, S, O = model.feature_extractor(dummy_input)
            events[1].record()

            # Stage 2: Fixation Sampler
            coords, norm_coords, fixation_scores, F_map = model.fixation_sampler(S, O, image_size=(H, W))
            events[2].record()

            # Stage 3: Local Patch Extractor
            patches = model.patch_extractor(dummy_input, coords)
            events[3].record()

            # Stage 4: Patch Embedding + Pos + Score
            foveal_tokens = model.extract_patch_embeddings(patches)
            grid_pos_embed = model.vit.pos_embed[:, 1:, :]
            r_pos = model.pos_interpolator(norm_coords, grid_pos_embed)
            score_tokens = model.score_embed(fixation_scores)
            foveal_tokens = foveal_tokens + r_pos + score_tokens
            events[4].record()

            # Stage 5: Global Tokens & Sequence Formation
            g_feat = model.global_token_proj(X)
            g_tokens = g_feat.flatten(2).transpose(1, 2) + model.global_pos_embed
            tokens = torch.cat([g_tokens, foveal_tokens], dim=1)
            tokens = model.vit.pos_drop(tokens)
            events[5].record()

            # Stage 6: ViT Transformer Blocks
            for block in model.vit.blocks:
                tokens = block(tokens)
            tokens = model.vit.norm(tokens)
            events[6].record()

            # Stage 7: Detection Head
            out_foveal_tokens = tokens[:, model.num_global_tokens:]
            pred_logits, pred_boxes = model.detection_head(out_foveal_tokens, norm_coords)
            events[7].record()

            torch.cuda.synchronize()

            timings["1. CNN Feature Extractor (MobileNetV3)"].append(events[0].elapsed_time(events[1]))
            timings["2. Fixation Sampler (Greedy NMS Loop)"].append(events[1].elapsed_time(events[2]))
            timings["3. Local Patch Extractor (grid_sample)"].append(events[2].elapsed_time(events[3]))
            timings["4. Patch Embedding & Pos/Score Injections"].append(events[3].elapsed_time(events[4]))
            timings["5. Global Token Projection"].append(events[4].elapsed_time(events[5]))
            timings["6. Transformer Blocks (12 Sparse Layers, 144 tokens)"].append(events[5].elapsed_time(events[6]))
            timings["7. Detection Head (Class & BBox for 128 tokens)"].append(events[6].elapsed_time(events[7]))
        else:
            t0 = time.perf_counter()
            X, S, O = model.feature_extractor(dummy_input)
            t1 = time.perf_counter()
            coords, norm_coords, fixation_scores, F_map = model.fixation_sampler(S, O, image_size=(H, W))
            t2 = time.perf_counter()
            patches = model.patch_extractor(dummy_input, coords)
            t3 = time.perf_counter()
            foveal_tokens = model.extract_patch_embeddings(patches)
            grid_pos_embed = model.vit.pos_embed[:, 1:, :]
            r_pos = model.pos_interpolator(norm_coords, grid_pos_embed)
            score_tokens = model.score_embed(fixation_scores)
            foveal_tokens = foveal_tokens + r_pos + score_tokens
            t4 = time.perf_counter()
            g_feat = model.global_token_proj(X)
            g_tokens = g_feat.flatten(2).transpose(1, 2) + model.global_pos_embed
            tokens = torch.cat([g_tokens, foveal_tokens], dim=1)
            tokens = model.vit.pos_drop(tokens)
            t5 = time.perf_counter()
            for block in model.vit.blocks:
                tokens = block(tokens)
            tokens = model.vit.norm(tokens)
            t6 = time.perf_counter()
            out_foveal_tokens = tokens[:, model.num_global_tokens:]
            pred_logits, pred_boxes = model.detection_head(out_foveal_tokens, norm_coords)
            t7 = time.perf_counter()

            timings["1. CNN Feature Extractor (MobileNetV3)"].append((t1 - t0) * 1000)
            timings["2. Fixation Sampler (Greedy NMS Loop)"].append((t2 - t1) * 1000)
            timings["3. Local Patch Extractor (grid_sample)"].append((t3 - t2) * 1000)
            timings["4. Patch Embedding & Pos/Score Injections"].append((t4 - t3) * 1000)
            timings["5. Global Token Projection"].append((t5 - t4) * 1000)
            timings["6. Transformer Blocks (12 Sparse Layers, 144 tokens)"].append((t6 - t5) * 1000)
            timings["7. Detection Head (Class & BBox for 128 tokens)"].append((t7 - t6) * 1000)

    return {k: float(np.mean(v)) for k, v in timings.items()}


@torch.no_grad()
def measure_peak_vram(model: nn.Module, dummy_input: torch.Tensor, device: str = "cuda") -> float:
    """Measures maximum VRAM allocated during forward pass in MB."""
    if not device.startswith("cuda"):
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = model(dummy_input)
    torch.cuda.synchronize()
    max_bytes = torch.cuda.max_memory_allocated()
    return float(max_bytes / (1024 ** 2))


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 95)
    print("   HARDWARE & LATENCY BENCHMARK: NORMAL DENSE ViT vs. PROPOSED SPARSE-ViT")
    print("=" * 95)
    print(f">> Device       : {device} ({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'})")
    print(f">> Image Size   : {args.image_size} x {args.image_size}")
    print(f">> ViT Backbone : {args.backbone}")
    print(f">> Iterations   : {args.warmup} Warmup runs | {args.runs} Timed test runs")
    print("=" * 95)

    # 1. Instantiate Models
    print("\n>> Initializing Models...")
    dense_model = BaselineViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, image_size=(args.image_size, args.image_size)
    ).to(device)

    sparse_k256 = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=256, image_size=(args.image_size, args.image_size), num_global_tokens=16
    ).to(device)

    sparse_k128 = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=128, image_size=(args.image_size, args.image_size), num_global_tokens=16
    ).to(device)

    sparse_k64 = SparseViTDetector(
        num_classes=80, backbone_name=args.backbone, pretrained=False, K=64, image_size=(args.image_size, args.image_size), num_global_tokens=16
    ).to(device)

    # Optional checkpoint loading
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f">> Loading checkpoint weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sparse_k128.load_state_dict(sd, strict=False)

    # 2. Main Comparison (Batch Size = 1)
    models_to_test = [
        ("Normal Dense ViT (Baseline)", dense_model, 1024, 0),
        ("Sparse-ViT (K=256)", sparse_k256, 272, 256),
        ("Sparse-ViT (K=128)", sparse_k128, 144, 128),
        ("Sparse-ViT (K=64)",  sparse_k64,  80, 64)
    ]

    dummy_single = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    results_summary = []
    print("\n>> [1/4] Running End-to-End Benchmarks (Batch Size = 1)...")

    dense_mean_ms = None
    for name, model, seq_len, k_val in models_to_test:
        print(f"   Profiling: {name:<28} ...", end="", flush=True)
        mean_ms, std_ms, fps = measure_end_to_end_latency(model, dummy_single, warmup=args.warmup, runs=args.runs, device=args.device)
        vram_mb = measure_peak_vram(model, dummy_single, device=args.device)
        total_flops, attn_flops = calculate_vit_flops(seq_len)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        if dense_mean_ms is None:
            dense_mean_ms = mean_ms
            speedup = 1.00
        else:
            speedup = dense_mean_ms / mean_ms

        results_summary.append({
            "name": name,
            "seq_len": seq_len,
            "k_val": k_val,
            "params_M": params_m,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "fps": fps,
            "speedup": speedup,
            "total_gflops": total_flops / 1e9,
            "attn_gflops": attn_flops / 1e9,
            "vram_mb": vram_mb
        })
        print(f" Done! {mean_ms:5.2f} ms ({fps:5.1f} FPS) | VRAM: {vram_mb:5.1f} MB")

    # 3. Side-by-Side Stage Latency Profiling
    print("\n>> [2/4] Profiling Stage-by-Stage Breakdown for Normal ViT...")
    dense_stage_breakdown = profile_dense_vit_stages(dense_model, dummy_single, warmup=args.warmup, runs=args.runs, device=args.device)

    print(">> [3/4] Profiling Stage-by-Stage Breakdown for Sparse-ViT (K=128)...")
    sparse_stage_breakdown = profile_sparse_vit_stages(sparse_k128, dummy_single, warmup=args.warmup, runs=args.runs, device=args.device)

    dense_staged_total = sum(dense_stage_breakdown.values())
    sparse_staged_total = sum(sparse_stage_breakdown.values())

    # 4. Batch Size Scaling
    print("\n>> [4/4] Running Batch Size Scaling Benchmarks (Normal ViT vs. Sparse-ViT K=128)...")
    batch_sizes = [1, 4, 8, 16]
    batch_results = {"batch_sizes": batch_sizes, "Dense": [], "Sparse_K128": []}

    for b in batch_sizes:
        try:
            b_input = torch.randn(b, 3, args.image_size, args.image_size, device=device)
            d_ms, _, d_fps = measure_end_to_end_latency(dense_model, b_input, warmup=10, runs=30, device=args.device)
            s_ms, _, s_fps = measure_end_to_end_latency(sparse_k128, b_input, warmup=10, runs=30, device=args.device)
            batch_results["Dense"].append({"batch": b, "ms": d_ms, "fps": d_fps})
            batch_results["Sparse_K128"].append({"batch": b, "ms": s_ms, "fps": s_fps})
            print(f"   Batch {b:2d} | Normal ViT: {d_ms:6.2f} ms ({d_fps:5.1f} FPS) | Sparse-K128: {s_ms:6.2f} ms ({s_fps:5.1f} FPS) -> Speedup: {d_ms/s_ms:.2f}x")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   Batch {b:2d} | OUT OF MEMORY (OOM) on {args.device}!")
                torch.cuda.empty_cache()
            else:
                raise e

    # 5. Display Formatted Tables
    print("\n" + "=" * 115)
    print(f"{'OVERALL COMPARISON: NORMAL DENSE ViT vs. SPARSE-ViT':^115}")
    print("=" * 115)

    headers = ["Model Architecture", "Tokens", "Params (M)", "Total FLOPs", "Attn FLOPs", "Latency (ms)", "FPS", "Speedup", "Peak VRAM"]
    print(f"{headers[0]:<28} | {headers[1]:<6} | {headers[2]:<10} | {headers[3]:<11} | {headers[4]:<11} | {headers[5]:<13} | {headers[6]:<7} | {headers[7]:<7} | {headers[8]:<9}")
    print("-" * 115)
    for r in results_summary:
        print(f"{r['name']:<28} | {r['seq_len']:<6} | {r['params_M']:5.2f} M   | {r['total_gflops']:5.2f} GFLOP | {r['attn_gflops']:5.2f} GFLOP | {r['mean_ms']:5.2f}±{r['std_ms']:3.1f} ms  | {r['fps']:5.1f} | {r['speedup']:4.2f}x  | {r['vram_mb']:5.1f} MB")

    # Side-by-side Stage breakdown table
    print("\n" + "=" * 95)
    print(f"{'SIDE-BY-SIDE STAGE BREAKDOWN: NORMAL ViT vs. SPARSE-ViT (K=128)':^95}")
    print("=" * 95)

    dense_trans_time = dense_stage_breakdown["3. Transformer Blocks (12 Dense Layers, 1024 tokens)"]
    sparse_trans_time = sparse_stage_breakdown["6. Transformer Blocks (12 Sparse Layers, 144 tokens)"]
    trans_speedup = dense_trans_time / sparse_trans_time if sparse_trans_time > 0 else 1.0

    print(f"1. NORMAL DENSE ViT (Total: {dense_staged_total:.2f} ms):")
    for k, v in dense_stage_breakdown.items():
        print(f"   - {k:<55}: {v:6.2f} ms ({(v/dense_staged_total)*100:4.1f}%)")

    print(f"\n2. PROPOSED SPARSE-ViT (Total: {sparse_staged_total:.2f} ms):")
    for k, v in sparse_stage_breakdown.items():
        print(f"   - {k:<55}: {v:6.2f} ms ({(v/sparse_staged_total)*100:4.1f}%)")

    print("\n3. DIRECT COMPONENT COMPARISON:")
    print(f"   * Transformer Execution Alone : Normal = {dense_trans_time:5.2f} ms vs Sparse = {sparse_trans_time:5.2f} ms ({trans_speedup:.2f}x Speedup!)")
    sampling_tax = sparse_stage_breakdown["1. CNN Feature Extractor (MobileNetV3)"] + sparse_stage_breakdown["2. Fixation Sampler (Greedy NMS Loop)"] + sparse_stage_breakdown["3. Local Patch Extractor (grid_sample)"]
    print(f"   * Fixation Preprocessing Tax  : {sampling_tax:5.2f} ms (CNN + Greedy Sampler + dynamic crop)")
    print(f"   * Net End-to-End Speedup      : {dense_staged_total / sparse_staged_total:.2f}x Faster overall")

    # 6. Save Markdown Report & JSON
    md_report_path = os.path.join(args.output_dir, "hardware_report.md")
    json_path = os.path.join(args.output_dir, "hardware_results.json")

    with open(md_report_path, "w") as f:
        f.write("# Hardware & Latency Profiling Report: Normal ViT vs. Sparse-ViT\n\n")
        f.write(f"- **Device**: {device} ({torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'})\n")
        f.write(f"- **Input Image Size**: {args.image_size} x {args.image_size}\n")
        f.write(f"- **ViT Backbone**: {args.backbone}\n\n")
        
        f.write("## 1. End-to-End Latency & FLOPs Comparison (Batch Size = 1)\n\n")
        f.write("| Model Architecture | Active Tokens | Params (M) | Total ViT FLOPs | Attention FLOPs | Latency (ms) | Throughput (FPS) | Speedup | Peak VRAM |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results_summary:
            f.write(f"| **{r['name']}** | {r['seq_len']} | {r['params_M']:.2f} M | {r['total_gflops']:.2f} GFLOPs | {r['attn_gflops']:.2f} GFLOPs | {r['mean_ms']:.2f} ± {r['std_ms']:.2f} ms | {r['fps']:.1f} FPS | **{r['speedup']:.2f}x** | {r['vram_mb']:.1f} MB |\n")

        f.write("\n## 2. Granular Stage-by-Stage Latency Breakdown Comparison\n\n")
        f.write("### Normal Dense ViT Stages (1024 Tokens)\n\n")
        f.write("| Stage | Latency (ms) | Percentage (%) |\n")
        f.write("| :--- | :---: | :---: |\n")
        for stage, t_ms in dense_stage_breakdown.items():
            f.write(f"| {stage} | {t_ms:.2f} ms | {(t_ms/dense_staged_total)*100:.1f}% |\n")
        f.write(f"| **Total Normal ViT Pipeline** | **{dense_staged_total:.2f} ms** | **100.0%** |\n\n")

        f.write("### Proposed Sparse-ViT Stages (144 Tokens)\n\n")
        f.write("| Stage | Latency (ms) | Percentage (%) |\n")
        f.write("| :--- | :---: | :---: |\n")
        for stage, t_ms in sparse_stage_breakdown.items():
            f.write(f"| {stage} | {t_ms:.2f} ms | {(t_ms/sparse_staged_total)*100:.1f}% |\n")
        f.write(f"| **Total Sparse-ViT Pipeline** | **{sparse_staged_total:.2f} ms** | **100.0%** |\n\n")

        f.write("### Direct Stage Comparison Summary\n\n")
        f.write(f"- **Transformer Self-Attention Time**: Normal ViT = `{dense_trans_time:.2f} ms` vs Sparse-ViT = `{sparse_trans_time:.2f} ms` (**{trans_speedup:.2f}x faster**)\n")
        f.write(f"- **Sparse Fixation Preprocessing Tax**: `{sampling_tax:.2f} ms` (CNN + Sampler + grid_sample)\n")
        f.write(f"- **Net End-to-End Speedup**: **{dense_staged_total / sparse_staged_total:.2f}x faster**\n\n")

        f.write("## 3. Batch Size Scaling (Throughput & Latency)\n\n")
        f.write("| Batch Size | Normal ViT Latency (ms) | Normal ViT FPS | Sparse-K128 Latency (ms) | Sparse-K128 FPS | Latency Speedup |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for i, b in enumerate(batch_sizes):
            if i < len(batch_results["Dense"]):
                d = batch_results["Dense"][i]
                s = batch_results["Sparse_K128"][i]
                f.write(f"| {b} | {d['ms']:.2f} ms | {d['fps']:.1f} | {s['ms']:.2f} ms | {s['fps']:.1f} | **{d['ms']/s['ms']:.2f}x** |\n")

    full_data = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
        "image_size": args.image_size,
        "backbone": args.backbone,
        "results_summary": results_summary,
        "dense_stage_breakdown": dense_stage_breakdown,
        "sparse_stage_breakdown": sparse_stage_breakdown,
        "batch_scaling": batch_results
    }
    with open(json_path, "w") as f:
        json.dump(full_data, f, indent=4)

    print(f"\n>> [COMPLETE] Markdown report written to: {md_report_path}")
    print(f">> [COMPLETE] Raw JSON results written to  : {json_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
