#!/usr/bin/env python3
"""
Positional-Interpolation Mode Latency Comparison (Bicubic vs Bilinear).

2x3 grid:
    {Original Sparse-ViT (greedy sampler), Method 2 (MaxPool NMS sampler)}
  x {bicubic grid_sample, bilinear grid_sample, fast bicubic (GEMM, torch.compile)}
of the pretrained ViT pos-embed table. Fast bicubic is numerically identical to
bicubic grid_sample (see fast_bicubic.py).

Stage timing mirrors profile_multi_image_analysis.py (CUDA events, fresh image
batch per run). Only the interpolation in stage 4b changes between configs.

Does NOT modify any original files in models/.
"""

import os
import sys
import argparse
import json
import gc
from typing import Dict, Optional, Tuple

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
from hardware_analysis.optimized_samplers import MaxPoolFixationSampler
from hardware_analysis.profile_multi_image_analysis import generate_diverse_batch
from hardware_analysis.fast_bicubic import fast_bicubic_pos_interp

_compiled_fast_bicubic = None


def _fast_bicubic(norm_coords, grid_pos_embed):
    global _compiled_fast_bicubic
    if _compiled_fast_bicubic is None:
        _compiled_fast_bicubic = torch.compile(fast_bicubic_pos_interp, dynamic=False)
    return _compiled_fast_bicubic(norm_coords, grid_pos_embed)


STAGES = [
    "1. CNN Feature Extractor",
    "2. Fixation Sampler",
    "3. Local Patch Extractor",
    "4. Patch Embedding & Pos/Score Proj",
    "   ├─ 4a. Pure Patch Embed (Conv2d + LN)",
    "   ├─ 4b. Pos Grid Interpolation",
    "   ├─ 4c. 2D Fourier Positional MLP",
    "   ├─ 4d. Fixation Score MLP",
    "   └─ 4e. Global Token Proj & Assembly",
    "5. Transformer Blocks (12 Layers)",
    "6. Detection Head",
    "Total Pipeline (End-to-End)",
]


def interpolate_pos(model: SparseViTDetector, norm_coords: torch.Tensor, mode: str) -> torch.Tensor:
    """Same as PositionalInterpolator's grid_sample step, with selectable mode ('fast_bicubic' = GEMM form)."""
    if mode == "fast_bicubic":
        return _fast_bicubic(norm_coords, model.vit.pos_embed[:, 1:, :])
    B, K, _ = norm_coords.shape
    G, D = model.grid_size, model.embed_dim
    grid_embed = model.vit.pos_embed[:, 1:, :].view(1, G, G, D).permute(0, 3, 1, 2)
    sample_grid = torch.stack(
        [2.0 * norm_coords[..., 0] - 1.0, 2.0 * norm_coords[..., 1] - 1.0], dim=-1
    ).view(B, K, 1, 2)
    return F.grid_sample(
        grid_embed.expand(B, D, G, G),
        sample_grid,
        mode=mode,
        padding_mode='border',
        align_corners=False
    ).squeeze(-1).permute(0, 2, 1)


@torch.no_grad()
def forward_staged(model, images, mode, events=None):
    """Full sparse forward pass; records CUDA events between stages if given."""
    rec = (lambda i: events[i].record()) if events is not None else (lambda i: None)
    B, _, H, W = images.shape

    rec(0)
    X, S, O = model.feature_extractor(images)
    rec(1)
    coords, norm_coords, fixation_scores, F_map = model.fixation_sampler(S, O, image_size=(H, W))
    rec(2)
    patches = model.patch_extractor(images, coords)
    rec(3)
    foveal_tokens = model.extract_patch_embeddings(patches)
    rec(4)
    interpolated = interpolate_pos(model, norm_coords, mode)
    rec(5)
    fourier_pos = model.pos_interpolator.fourier_encoder(norm_coords)
    r_pos = interpolated + model.pos_interpolator.alpha * fourier_pos
    rec(6)
    score_tokens = model.score_embed(fixation_scores)
    foveal_tokens = foveal_tokens + r_pos + score_tokens
    rec(7)
    g_tokens = model.global_token_proj(X).flatten(2).transpose(1, 2) + model.global_pos_embed
    tokens = model.vit.pos_drop(torch.cat([g_tokens, foveal_tokens], dim=1))
    rec(8)
    for block in model.vit.blocks:
        tokens = block(tokens)
    tokens = model.vit.norm(tokens)
    rec(9)
    pred_logits, pred_boxes = model.detection_head(tokens[:, model.num_global_tokens:], norm_coords)
    rec(10)
    return pred_logits, pred_boxes


@torch.no_grad()
def profile(model, mode, batch_size, image_size, device, warmup, num_images) -> Optional[Dict[str, Tuple[float, float]]]:
    model.eval()
    try:
        torch.cuda.empty_cache()
        gc.collect()
        for w in range(warmup):
            forward_staged(model, generate_diverse_batch(batch_size, image_size, device, seed=1000 + w), mode)
        torch.cuda.synchronize()

        ev = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
        times = {k: [] for k in STAGES}
        # (stage, start_event, end_event)
        spans = [(STAGES[0], 0, 1), (STAGES[1], 1, 2), (STAGES[2], 2, 3), (STAGES[3], 3, 8),
                 (STAGES[4], 3, 4), (STAGES[5], 4, 5), (STAGES[6], 5, 6), (STAGES[7], 6, 7),
                 (STAGES[8], 7, 8), (STAGES[9], 8, 9), (STAGES[10], 9, 10), (STAGES[11], 0, 10)]

        for run_idx in range(num_images):
            images = generate_diverse_batch(batch_size, image_size, device, seed=2000 + run_idx)
            torch.cuda.synchronize()
            forward_staged(model, images, mode, ev)
            torch.cuda.synchronize()
            for name, a, b in spans:
                times[name].append(ev[a].elapsed_time(ev[b]))

        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in times.items()}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return None
        raise


@torch.no_grad()
def isolated_grid_sample(model, batch_size, K, device, iters=200) -> Dict[str, float]:
    """Micro-benchmark of only the interpolation call, plus numerical deviation vs bicubic grid_sample."""
    norm_coords = torch.rand(batch_size, K, 2, device=device)
    out = {}
    for mode in ("bicubic", "bilinear", "fast_bicubic"):
        for _ in range(20):
            interpolate_pos(model, norm_coords, mode)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            interpolate_pos(model, norm_coords, mode)
        e.record()
        torch.cuda.synchronize()
        out[mode] = s.elapsed_time(e) / iters
    bic = interpolate_pos(model, norm_coords, "bicubic")
    bil = interpolate_pos(model, norm_coords, "bilinear")
    out["max_abs_diff"] = float((bic - bil).abs().max())
    out["rel_l2_diff"] = float((bic - bil).norm() / bic.norm())
    out["fast_bicubic_max_abs_diff"] = float((bic - interpolate_pos(model, norm_coords, "fast_bicubic")).abs().max())
    return out


def _bicubic_ref_name(name: str) -> str:
    return name.split(" + ")[0] + " + Bicubic"


def print_granular_breakdown(stats: Dict[str, Optional[Dict[str, Tuple[float, float]]]], B: int):
    """Side-by-side per-stage table: mean +/- std and % of each config's end-to-end, for every config."""
    total_key = "Total Pipeline (End-to-End)"
    names = list(stats.keys())
    w = 21
    line = "-" * (44 + (w + 3) * len(names))
    print(f"\n>> [Granular Pipeline Breakdown, B={B}]  mean +/- std ms  (% of that config's end-to-end)")
    print(f"{'Pipeline Stage':<42} | " + " | ".join(f"{n:^{w}}" for n in names))
    print(line)
    for stage in STAGES:
        if stage == total_key:
            print(line)
        cells, pcts = [], []
        for name in names:
            st = stats.get(name)
            if st is None:
                cells.append(f"{'OOM':^{w}}")
                pcts.append(f"{'':^{w}}")
                continue
            m, s = st[stage]
            cells.append(f"{m:7.3f} +/- {s:5.3f} ms".center(w))
            pcts.append(f"({m / st[total_key][0] * 100:5.1f}%)".center(w))
        print(f"{stage:<42} | " + " | ".join(cells))
        if stage != total_key:
            print(f"{'':<42} | " + " | ".join(pcts))

    def row(label, fn):
        print(f"{label:<42} | " + " | ".join(
            (fn(n).center(w) if stats.get(n) and stats.get(_bicubic_ref_name(n)) and stats.get(names[0])
             else f"{'-':^{w}}") for n in names))

    tot = lambda n: stats[n][total_key][0]
    row("Saved vs same-sampler Bicubic (ms)", lambda n: f"{tot(_bicubic_ref_name(n)) - tot(n):+7.3f} ms")
    row("Throughput (FPS)", lambda n: f"{B / (tot(n) / 1000):7.1f} FPS")
    row(f"Speedup vs {names[0]}", lambda n: f"{tot(names[0]) / tot(n):5.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Bicubic vs Bilinear vs Fast Bicubic pos-interp latency: Original vs Method 2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="vit_small_patch16_224")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--pretrained", action="store_true",
                        help="Load real timm pos_embed (affects only the numerical diff, not latency)")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "interp_mode_results.json"))
    parser.add_argument("--from-json", type=str, default=None,
                        help="Only re-print the granular tables from a saved results JSON (no profiling)")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            saved = json.load(f)
        print(f">> {saved['device']} | image {saved['image_size']} | K={saved['K']} | {saved['num_images']} runs")
        for B, d in saved["batches"].items():
            print("\n" + "#" * 180 + f"\n{'':80}BATCH SIZE = {B}\n" + "#" * 180)
            print_granular_breakdown(d["stages"], int(B))
        return

    device = torch.device(args.device)
    img = (args.image_size, args.image_size)
    print("=" * 120)
    print("   POSITIONAL INTERPOLATION: BICUBIC vs BILINEAR vs FAST BICUBIC  |  ORIGINAL SPARSE-ViT vs METHOD 2 (MaxPool NMS)")
    print("=" * 120)
    print(f">> Device: {device} ({torch.cuda.get_device_name(0)}) | torch {torch.__version__}")
    print(f">> Image {args.image_size}x{args.image_size} | K={args.K} | backbone {args.backbone} | "
          f"{args.num_images} runs, {args.warmup} warmup")

    model_orig = SparseViTDetector(num_classes=80, backbone_name=args.backbone, pretrained=args.pretrained,
                                   K=args.K, image_size=img).to(device)
    model_m2 = SparseViTDetector(num_classes=80, backbone_name=args.backbone, pretrained=args.pretrained,
                                 K=args.K, image_size=img).to(device)
    model_m2.fixation_sampler = MaxPoolFixationSampler(K=args.K, patch_size=16, image_size=img).to(device)

    configs = [
        ("Original + Bicubic", model_orig, "bicubic"),
        ("Original + Bilinear", model_orig, "bilinear"),
        ("Method 2 + Bicubic", model_m2, "bicubic"),
        ("Method 2 + Bilinear", model_m2, "bilinear"),
        ("Original + Fast Bicubic", model_orig, "fast_bicubic"),
        ("Method 2 + Fast Bicubic", model_m2, "fast_bicubic"),
    ]

    results = {"device": torch.cuda.get_device_name(0), "image_size": args.image_size, "K": args.K,
               "num_images": args.num_images, "batches": {}}

    for B in args.batch_sizes:
        print("\n" + "#" * 120)
        print(f"{'':40}BATCH SIZE = {B}")
        print("#" * 120)
        stats = {name: profile(m, mode, B, args.image_size, args.device, args.warmup, args.num_images)
                 for name, m, mode in configs}
        micro = isolated_grid_sample(model_orig, B, args.K, device)

        # Executive summary
        print(f"\n>> [End-to-End]  (speedup column is vs Original + Bicubic)")
        print(f"{'Configuration':<26} | {'Total latency':<20} | {'Pos interp (4b)':<20} | {'4b % of total':<13} | {'FPS':<8} | Speedup")
        print("-" * 120)
        ref = stats["Original + Bicubic"]["Total Pipeline (End-to-End)"][0] if stats["Original + Bicubic"] else None
        for name, _, _ in configs:
            st = stats[name]
            if st is None:
                print(f"{name:<26} | OOM")
                continue
            m, s = st["Total Pipeline (End-to-End)"]
            im, is_ = st["   ├─ 4b. Pos Grid Interpolation"]
            sp = f"{ref / m:.2f}x" if ref else "-"
            print(f"{name:<26} | {m:7.2f} +/- {s:5.2f} ms | {im:7.3f} +/- {is_:5.3f} ms | {im / m * 100:11.1f}% | "
                  f"{B / (m / 1000):6.1f} | {sp}")

        print_granular_breakdown(stats, B)

        print(f"\n>> [Isolated interpolation micro-benchmark, B={B}, K={args.K}]")
        print(f"   bicubic grid_sample: {micro['bicubic']:.4f} ms | bilinear grid_sample: {micro['bilinear']:.4f} ms | "
              f"fast bicubic (compiled GEMM): {micro['fast_bicubic']:.4f} ms")
        print(f"   numerical diff (bilinear vs bicubic): max|Δ| = {micro['max_abs_diff']:.4e}, "
              f"rel L2 = {micro['rel_l2_diff']:.4e}")
        print(f"   numerical diff (fast bicubic vs bicubic): max|Δ| = {micro['fast_bicubic_max_abs_diff']:.4e}")

        results["batches"][str(B)] = {"stages": stats, "isolated_grid_sample": micro}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>> Saved results to {args.output}")


if __name__ == "__main__":
    main()
