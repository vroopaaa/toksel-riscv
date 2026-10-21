#!/usr/bin/env python3
"""
Greedy vs Max-Pool NMS sampler: how different are the chosen cells?

Metrics per K (averaged over maps):
  peaks      : number of 3x3 local maxima in F (Max-Pool keeps these first)
  filler %   : share of Max-Pool tokens that are NOT peaks (K - peaks) / K
  overlap %  : |greedy ∩ maxpool| / K, exact same cells
  greedy uniq: distinct cells greedy returns (it can run out of unsuppressed cells)
  adjacent % : share of tokens with another token in their 3x3 neighbourhood (clustering)
  coverage % : share of the 32x32 grid within 2 cells (Chebyshev) of some token

Maps:
  --maps file.pt   tensor (N, 1, 32, 32) of fused F maps dumped from a trained model
                   (e.g. out["F"] from the model forward on COCO val images)
  default          synthetic maps: i.i.d. noise (what verify_sampler_fidelity.py used)
                   and smooth saliency-like maps (Gaussian blobs + blurred noise)

Does NOT modify any original files in models/.
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as Fn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
for p in [PROJECT_DIR, os.path.abspath(os.path.join(SCRIPT_DIR, ".."))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.fixation_sampler import FixationSampler
from hardware_analysis.optimized_samplers import MaxPoolFixationSampler

G = 32


def cell_idx(coords: torch.Tensor) -> torch.Tensor:
    """Pixel coords (B, K, 2) at cell centres -> flat cell index r * 32 + c."""
    c = ((coords[..., 0] - 8) / 16).round().long()
    r = ((coords[..., 1] - 8) / 16).round().long()
    return r * G + c


def smooth_maps(n: int, seed: int) -> torch.Tensor:
    """Saliency-like maps: a few Gaussian blobs of random size plus weak blurred noise."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(G).float(), torch.arange(G).float(), indexing="ij")
    out = torch.zeros(n, 1, G, G)
    for i in range(n):
        m = torch.zeros(G, G)
        for _ in range(int(torch.randint(2, 7, (1,), generator=g))):
            cx, cy = torch.rand(2, generator=g) * (G - 1)
            s = 1.5 + torch.rand(1, generator=g).item() * 5
            m += torch.rand(1, generator=g).item() * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        noise = Fn.avg_pool2d(torch.rand(1, 1, G, G, generator=g), 5, 1, 2)[0, 0]
        out[i, 0] = m + 0.15 * noise
    return out


@torch.no_grad()
def compare(F_maps: torch.Tensor, K: int) -> dict:
    """F_maps (N, 1, 32, 32). Passing F as both S and O with alpha + beta = 1 reproduces F exactly."""
    greedy = FixationSampler(K=K, alpha=0.5, beta=0.5)
    pool = MaxPoolFixationSampler(K=K, alpha=0.5, beta=0.5)
    cg, *_ = greedy(F_maps, F_maps, image_size=(512, 512))
    cp, *_ = pool(F_maps, F_maps, image_size=(512, 512))
    ig, ip = cell_idx(cg), cell_idx(cp)

    Fn_ = (F_maps - F_maps.amin((2, 3), keepdim=True)) / (F_maps.amax((2, 3), keepdim=True) - F_maps.amin((2, 3), keepdim=True) + 1e-6)
    peaks = (Fn_ == Fn.max_pool2d(Fn_, 3, 1, 1)).flatten(1).sum(1).float()

    def token_mask(idx):
        m = torch.zeros(idx.shape[0], G * G)
        m.scatter_(1, idx, 1.0)
        return m.view(-1, 1, G, G)

    def adjacent(idx):
        m = token_mask(idx)
        cnt = Fn.conv2d(m, torch.ones(1, 1, 3, 3), padding=1) - m
        return ((cnt > 0) & (m > 0)).flatten(1).sum(1).float() / m.flatten(1).sum(1)

    def coverage(idx):
        return Fn.max_pool2d(token_mask(idx), 5, 1, 2).flatten(1).mean(1)

    overlap = torch.stack([torch.isin(ig[i], ip[i]).float().mean() for i in range(ig.shape[0])])
    uniq_g = torch.tensor([ig[i].unique().numel() for i in range(ig.shape[0])]).float()
    return {
        "peaks": peaks.mean().item(),
        "filler": ((K - peaks.clamp(max=K)) / K).mean().item() * 100,
        "overlap": overlap.mean().item() * 100,
        "greedy_uniq": uniq_g.mean().item(),
        "greedy_uniq_min": uniq_g.min().item(),
        "adj_greedy": adjacent(ig).mean().item() * 100,
        "adj_pool": adjacent(ip).mean().item() * 100,
        "cov_greedy": coverage(ig).mean().item() * 100,
        "cov_pool": coverage(ip).mean().item() * 100,
    }


def report(name: str, F_maps: torch.Tensor, Ks):
    print(f"\n>> {name}  ({F_maps.shape[0]} maps)")
    print(f"{'K':>4} | {'peaks':>6} | {'filler %':>8} | {'overlap %':>9} | {'greedy uniq (min)':>17} | "
          f"{'adjacent % G / MP':>17} | {'coverage % G / MP':>17}")
    print("-" * 100)
    for K in Ks:
        r = compare(F_maps, K)
        print(f"{K:>4} | {r['peaks']:6.1f} | {r['filler']:8.1f} | {r['overlap']:9.1f} | "
              f"{r['greedy_uniq']:8.1f} ({r['greedy_uniq_min']:4.0f})   | "
              f"{r['adj_greedy']:7.1f} / {r['adj_pool']:5.1f}   | {r['cov_greedy']:7.1f} / {r['cov_pool']:5.1f}")


def main():
    ap = argparse.ArgumentParser(description="Compare cells chosen by the greedy and Max-Pool NMS samplers")
    ap.add_argument("--maps", type=str, default=None, help="torch .pt file with F maps of shape (N, 1, 32, 32)")
    ap.add_argument("--K", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--n", type=int, default=200, help="number of synthetic maps")
    args = ap.parse_args()

    if args.maps:
        report(f"Real F maps from {args.maps}", torch.load(args.maps, map_location="cpu").float(), args.K)
        return
    torch.manual_seed(0)
    report("i.i.d. noise maps (what verify_sampler_fidelity.py tested)", torch.rand(args.n, 1, G, G), args.K)
    report("smooth saliency-like maps (2-6 Gaussian blobs + weak blurred noise)", smooth_maps(args.n, 0), args.K)


if __name__ == "__main__":
    main()
