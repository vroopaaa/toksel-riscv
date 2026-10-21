#!/usr/bin/env python3
"""
Full-pipeline latency of exact greedy sampler variants.

Configs (same model weights, only model.fixation_sampler changes):
  torch greedy (orig)  repo FixationSampler on the GPU (float F, cdist + 128-step loop)
  max-pool (Method 2)  reference only: parallel but NOT the same picks as greedy
  host: <variant>      F computed on the GPU, quantised to uint8, copied to the host, picked
                       by a greedy_variants.py algorithm (naive / rowmax / tilemax / sortscan /
                       threshold / bucketscan / threshold+), indices sent back. threshold+ also
                       gets its histogram from the GPU quantisation step. Mirrors a host core picking fixations
                       while an accelerator runs the network.

Stage timing reuses profile_interp_mode_comparison.profile (CUDA events, fresh image per run,
bicubic positional interpolation as in the original model). The host sampler blocks the stream,
so its CPU time and both transfers land in stage "2. Fixation Sampler".
Does NOT modify any original files in models/.
"""

import os
import sys
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
for p in [PROJECT_DIR, os.path.abspath(os.path.join(SCRIPT_DIR, "..")), SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.sparse_vit_det import SparseViTDetector
from models.fixation_sampler import FixationSampler
from hardware_analysis.optimized_samplers import MaxPoolFixationSampler
from hardware_analysis.greedy_variants import ALGOS, TAKES_HIST
from hardware_analysis.profile_interp_mode_comparison import profile


class HostGreedySampler(nn.Module):
    """Drop-in fixation sampler that runs an exact greedy variant on the host on a map quantised to
    `bits` bits (8 -> uint8; more bits only for bucketscan, which then uses 2^bits buckets)."""

    def __init__(self, algo: str, K: int = 128, alpha: float = 0.60, beta: float = 0.40,
                 patch_size: int = 16, image_size: Tuple[int, int] = (512, 512), bits: int = 8):
        super().__init__()
        assert bits == 8 or algo == "bucketscan", "only bucketscan supports more than 8 bits"
        self.algo, self.fn, self.bits = algo, ALGOS[algo], bits
        self.K, self.alpha, self.beta = K, alpha, beta
        self.patch_size, self.image_size = patch_size, image_size

    @staticmethod
    def normalize_map(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B = M.shape[0]
        flat = M.view(B, -1)
        mn = flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
        mx = flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
        return (M - mn) / (mx - mn + eps)

    def forward(self, S: torch.Tensor, O: torch.Tensor, image_size: Optional[Tuple[int, int]] = None):
        B, _, Hf, Wf = S.shape
        H, W = image_size if image_size is not None else self.image_size
        F = self.alpha * self.normalize_map(S) + self.beta * self.normalize_map(O)
        u8 = (F[:, 0] * (2 ** self.bits - 1)).round().to(torch.uint8 if self.bits == 8 else torch.int16)
        if self.algo in TAKES_HIST:
            # histogram fused into the quantisation step: one bincount over all images (offset per image)
            off = (torch.arange(B, device=u8.device) * 256).view(B, 1)
            hists = torch.bincount((u8.view(B, -1).long() + off).flatten(), minlength=256 * B).view(B, 256).cpu().numpy()
        maps = u8.cpu().numpy()                                            # device -> host
        idx = np.zeros((B, self.K), np.int64)                              # pad = cell 0, like the original
        for b in range(B):
            kw = {"hist": hists[b]} if self.algo in TAKES_HIST else ({"levels": 2 ** self.bits} if self.bits != 8 else {})
            picks, _ = self.fn(maps[b], self.K, **kw)
            idx[b, :len(picks)] = picks
        idx = torch.from_numpy(idx).to(S.device)                           # host -> device
        r, c = idx // Wf, idx % Wf
        half = self.patch_size / 2.0
        px = ((c.float() + 0.5) * (W / Wf)).clamp(half, W - half)
        py = ((r.float() + 0.5) * (H / Hf)).clamp(half, H - half)
        coords = torch.stack([px, py], -1)
        norm_coords = torch.stack([px / W, py / H], -1)
        grid = (2.0 * norm_coords - 1.0).view(B, self.K, 1, 2)
        scores = Fn.grid_sample(F, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return coords, norm_coords, scores.squeeze(-1).permute(0, 2, 1), F


def main():
    ap = argparse.ArgumentParser(description="Full-pipeline latency of exact greedy sampler variants")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "greedy_variants_pipeline.json"))
    args = ap.parse_args()

    print("=" * 112)
    print("   FULL-PIPELINE LATENCY: EXACT GREEDY SAMPLER VARIANTS")
    print("=" * 112)
    print(f">> {torch.cuda.get_device_name(0)} | K={args.K} | {args.num_images} fresh images per test, "
          f"{args.warmup} warmup | stage 2 includes GPU->host->GPU transfers for host samplers")
    results = {}
    for size in args.image_sizes:
        img = (size, size)
        model = SparseViTDetector(num_classes=80, pretrained=False, K=args.K, image_size=img).to(args.device)
        configs = [("torch greedy (orig)", FixationSampler(K=args.K, image_size=img)),
                   ("max-pool (Method 2)", MaxPoolFixationSampler(K=args.K, image_size=img))]
        configs += [(f"host: {a}", HostGreedySampler(a, K=args.K, image_size=img)) for a in ALGOS]
        for B in args.batch_sizes:
            print(f"\n{'#' * 112}\n   IMAGE {size}x{size} (grid {size // 16}x{size // 16}), BATCH {B}\n{'#' * 112}")
            print(f"{'Sampler':<22} | {'Sampler stage (ms)':>19} | {'CNN (ms)':>9} | {'ViT blocks (ms)':>15} | "
                  f"{'End-to-end (ms)':>17} | {'FPS':>6} | {'vs orig':>7}")
            print("-" * 112)
            ref = None
            for name, smp in configs:
                model.fixation_sampler = smp.to(args.device)
                st = profile(model, "bicubic", B, size, args.device, args.warmup, args.num_images)
                if st is None:
                    print(f"{name:<22} | OOM")
                    continue
                tot = st["Total Pipeline (End-to-End)"]
                ref = ref or tot[0]
                s2 = st["2. Fixation Sampler"]
                print(f"{name:<22} | {s2[0]:8.3f} +/- {s2[1]:6.3f} | {st['1. CNN Feature Extractor'][0]:9.3f} | "
                      f"{st['5. Transformer Blocks (12 Layers)'][0]:15.3f} | {tot[0]:8.2f} +/- {tot[1]:5.2f} | "
                      f"{B / (tot[0] / 1000):6.1f} | {ref / tot[0]:6.2f}x")
                results[f"{size}/B{B}/{name}"] = st
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>> saved {args.output}")


if __name__ == "__main__":
    main()
