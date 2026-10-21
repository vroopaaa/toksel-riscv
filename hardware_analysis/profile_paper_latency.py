#!/usr/bin/env python3
"""
One consistent stage-wise latency benchmark for the paper figure:

  dense                     BaselineViTDetector (1024 tokens)
  sparse original           SparseViTDetector, repo FixationSampler (iterative GPU greedy loop),
                            bicubic grid_sample positional interpolation
  sparse optimized (<algo>) same model, exact greedy on the host on a uint8 map
                            (greedy_variants.py: bucketscan / sortscan / ...), GEMM fast bicubic

All configs see the same images in the same order, round-robin per image (image i runs on every
config before image i+1), so thermal / clock drift hits every config equally. Batch size 1,
CUDA events between stages, mean / std / median over --num-images.

Images:
  --image-dir DIR   real images (e.g. COCO val2017): resized to SIZE x SIZE, ImageNet-normalised
  (default)         synthetic images from profile_multi_image_analysis.generate_diverse_batch
Weights:
  --sparse-ckpt / --dense-ckpt   trained checkpoints (model_state_dict). Without a sparse checkpoint
                                 the S/O maps come from untrained heads, which only matters for the
                                 host sampler (its scan length depends on the map).

Does NOT modify any original files in models/.
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
for p in [PROJECT_DIR, os.path.abspath(os.path.join(SCRIPT_DIR, "..")), SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.baseline_vit_det import BaselineViTDetector
from models.sparse_vit_det import SparseViTDetector
from models.fixation_sampler import FixationSampler
from hardware_analysis.profile_interp_mode_comparison import forward_staged
from hardware_analysis.profile_multi_image_analysis import generate_diverse_batch
from hardware_analysis.profile_greedy_variants import HostGreedySampler

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DENSE_STAGES = ["patch_embed", "pos_embed", "transformer", "head", "total"]
SPARSE_STAGES = ["cnn", "token_selection", "patch_extract", "patch_embed", "pos_interp",
                 "fourier_pos", "score_embed", "global_tokens", "transformer", "head", "total"]
# forward_staged records events 0..10; (stage, start, end)
SPARSE_SPANS = list(zip(SPARSE_STAGES, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]))


def load_images(args):
    if args.image_dir is None:
        return [generate_diverse_batch(1, args.size, "cpu", seed=2000 + i) for i in range(args.num_images)], "synthetic"
    files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    files = files[:args.num_images]
    if not files:
        sys.exit(f"no images in {args.image_dir}")
    out = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None] / 255.0
        out.append((x - MEAN) / STD)
    return out, os.path.abspath(args.image_dir)


def load_ckpt(model, path, name):
    if path is None:
        return "none (random init)"
    sd = torch.load(path, map_location="cpu", weights_only=False)
    res = model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    print(f">> {name}: loaded {path} (missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)})")
    return os.path.abspath(path)


@torch.no_grad()
def dense_staged(model, x, ev=None):
    rec = (lambda i: ev[i].record()) if ev is not None else (lambda i: None)
    B = x.shape[0]
    rec(0)
    t = model.vit.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(model.vit.patch_embed, "norm", None) is not None:
        t = model.vit.patch_embed.norm(t)
    rec(1)
    t = model.vit.pos_drop(t + model.interpolate_pos_embed(t))
    rec(2)
    for blk in model.vit.blocks:
        t = blk(t)
    t = model.vit.norm(t)
    rec(3)
    model.detection_head(t, model.get_dense_grid_coords(B, x.device))
    rec(4)


def main():
    ap = argparse.ArgumentParser(description="Consistent dense / original / optimized stage latency for the paper")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--algos", nargs="+", default=["bucketscan", "sortscan"],
                    help="host samplers; append @bits for bucketscan, e.g. bucketscan@10")
    ap.add_argument("--sparse-ckpt", default=None)
    ap.add_argument("--dense-ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "paper_latency_results.json"))
    args = ap.parse_args()
    dev, img = args.device, (args.size, args.size)

    images, source = load_images(args)
    print(f">> {torch.cuda.get_device_name(0)} | {args.size}x{args.size} | K={args.K} | batch 1 | "
          f"{len(images)} images ({source}) | {args.warmup} warm-up")

    dense = BaselineViTDetector(num_classes=80, pretrained=False, image_size=img).to(dev).eval()
    dense_ck = load_ckpt(dense, args.dense_ckpt, "dense")
    sparse = SparseViTDetector(num_classes=80, pretrained=False, K=args.K, image_size=img).to(dev).eval()
    sparse_ck = load_ckpt(sparse, args.sparse_ckpt, "sparse")

    torch_greedy = FixationSampler(K=args.K, image_size=img).to(dev)
    configs = [("dense", None, None), ("sparse_original", torch_greedy, "bicubic")]
    for spec in args.algos:                                        # "bucketscan" or "bucketscan@10" (bits)
        a, bits = (spec.split("@")[0], int(spec.split("@")[1])) if "@" in spec else (spec, 8)
        name = f"sparse_optimized_{a}" + (f"_{bits}bit" if bits != 8 else "")
        configs.append((name, HostGreedySampler(a, K=args.K, image_size=img, bits=bits).to(dev), "fast_bicubic"))

    def run(name, sampler, mode, x, ev=None):
        if name == "dense":
            dense_staged(dense, x, ev)
        else:
            sparse.fixation_sampler = sampler
            forward_staged(sparse, x, mode, ev)

    for w in range(args.warmup):                                   # also triggers torch.compile
        x = images[w % len(images)].to(dev)
        for cfg in configs:
            run(*cfg, x)
    torch.cuda.synchronize()

    times = {n: {s: [] for s in (DENSE_STAGES if n == "dense" else SPARSE_STAGES)} for n, _, _ in configs}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
    for i, x_cpu in enumerate(images):
        x = x_cpu.to(dev)
        for name, sampler, mode in configs:
            torch.cuda.synchronize()
            run(name, sampler, mode, x, ev)
            torch.cuda.synchronize()
            if name == "dense":
                for j, s in enumerate(DENSE_STAGES[:-1]):
                    times[name][s].append(ev[j].elapsed_time(ev[j + 1]))
                times[name]["total"].append(ev[0].elapsed_time(ev[4]))
            else:
                for s, a, b in SPARSE_SPANS:
                    times[name][s].append(ev[a].elapsed_time(ev[b]))
        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(images)} images")

    stats = {n: {s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v))}
                 for s, v in st.items()} for n, st in times.items()}
    out = {"metadata": {"device": torch.cuda.get_device_name(0), "image_size": args.size, "K": args.K,
                        "batch_size": 1, "num_images": len(images), "warmup": args.warmup,
                        "image_source": source, "sparse_checkpoint": sparse_ck, "dense_checkpoint": dense_ck,
                        "timing": "CUDA events between stages, round-robin over configs per image"},
           "results": stats}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'stage (mean ms)':<18}" + "".join(f"{n:>30}" for n, _, _ in configs))
    for s in SPARSE_STAGES:
        row = [stats[n].get(s, {}).get("mean") for n, _, _ in configs]
        print(f"{s:<18}" + "".join(f"{'-' if v is None else f'{v:.3f}':>30}" for v in row))
    print(f"{'pos_embed (dense)':<18}{stats['dense']['pos_embed']['mean']:>30.3f}")
    print(f"\n>> saved {args.output}")


if __name__ == "__main__":
    main()
