#!/usr/bin/env python3
"""
How many fixation tokens change when greedy token selection runs on quantised scores?

For every image: F = alpha * norm(S) + beta * norm(O) from the (trained) guidance CNN, then
  float greedy  : the original selection on float32 F
  q-bit greedy  : the same selection on round(F * (2^q - 1)), q in --bits (8 = what bucketscan uses)
Both pick the highest unblocked cell, ties -> lower index, and block its 3x3 neighbourhood
(d_min = 28 px at 16 px cells). Agreement = |float picks ∩ q-bit picks| / (number of picks, <= K); token order is
ignored because each token carries its own coordinates.

The numpy float greedy is checked against the repo FixationSampler on the first batch.

Usage (cluster):
  python measure_quantization_agreement.py --image-dir <coco/val2017> --checkpoint <det_K128 best_detector.pt>
Without --image-dir it uses synthetic images (smoke test only).

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

from models.sparse_vit_det import SparseViTDetector
from hardware_analysis.profile_multi_image_analysis import generate_diverse_batch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def greedy(scores: np.ndarray, K: int, D: int) -> list:
    """Exact greedy on a flat score array: highest first, ties -> lower index, block 3x3."""
    blocked = np.zeros((D, D), bool)
    picks = []
    for i in np.argsort(-scores, kind="stable"):
        r, c = divmod(int(i), D)
        if blocked[r, c]:
            continue
        picks.append(int(i))
        blocked[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
        if len(picks) == K:
            break
    return picks


def image_batches(args):
    if args.image_dir is None:
        for s in range(0, args.num_images, args.batch_size):
            yield generate_diverse_batch(min(args.batch_size, args.num_images - s), args.size, "cpu", seed=3000 + s)
        return
    files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(args.image_dir, f"*.{ext}")))
    files = files[:args.num_images]
    if not files:
        sys.exit(f"no images in {args.image_dir}")
    for s in range(0, len(files), args.batch_size):
        xs = []
        for f in files[s:s + args.batch_size]:
            im = Image.open(f).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
            xs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0)
        yield (torch.stack(xs) - MEAN) / STD


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Token agreement: float greedy vs quantised greedy")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--checkpoint", default=None, help="trained SparseViTDetector checkpoint")
    ap.add_argument("--num-images", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--K", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 10, 12, 16])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-maps", default=None, help="optional .pt file to save all F maps (N, 1, G, G)")
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "quantization_agreement.json"))
    args = ap.parse_args()

    model = SparseViTDetector(num_classes=80, pretrained=False, K=128, image_size=(args.size, args.size))
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        res = model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        print(f">> loaded {args.checkpoint} (missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)})")
    else:
        print(">> WARNING: no checkpoint, S/O come from untrained heads (smoke test only)")
    model = model.to(args.device).eval()
    smp = model.fixation_sampler

    agree = {(K, b): [] for K in args.K for b in args.bits}
    all_maps, checked, n = [], False, 0
    for x in image_batches(args):
        x = x.to(args.device)
        _, S, O = model.feature_extractor(x)
        F = smp.alpha * smp.normalize_map(S) + smp.beta * smp.normalize_map(O)        # (B, 1, G, G)
        G = F.shape[-1]
        Fn = F[:, 0].flatten(1).float().cpu().numpy()
        if args.dump_maps:
            all_maps.append(F.cpu())

        if not checked:                                    # numpy greedy == repo FixationSampler?
            coords, *_ = smp(S, O, image_size=(args.size, args.size))
            stride = args.size / G
            ref = (((coords[..., 1] / stride) - 0.5).round() * G + ((coords[..., 0] / stride) - 0.5).round()).long().cpu()
            ok = all(greedy(Fn[b], smp.K, G) == ref[b].tolist() for b in range(Fn.shape[0]))
            print(f">> numpy float greedy identical to FixationSampler on first batch: {ok}")
            checked = True

        for b in range(Fn.shape[0]):
            for K in args.K:
                ref = set(greedy(Fn[b], K, G))         # can hold < K cells: the grid runs out at large K
                for bits in args.bits:
                    q = np.round(Fn[b] * (2 ** bits - 1)).astype(np.int64)
                    got = set(greedy(q, K, G))
                    agree[(K, bits)].append(len(ref & got) / max(len(ref), len(got)))
        n += Fn.shape[0]
        print(f"   {n} images")

    out = {"metadata": {"image_source": os.path.abspath(args.image_dir) if args.image_dir else "synthetic",
                        "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
                        "num_images": n, "image_size": args.size, "grid": G},
           "results": {}}
    print(f"\n{'K':>4} {'bits':>5} | {'mean agreement %':>16} | {'min %':>6} | {'images identical %':>18}")
    print("-" * 60)
    for (K, bits), v in agree.items():
        v = np.array(v)
        r = {"mean": float(v.mean() * 100), "min": float(v.min() * 100), "identical": float((v == 1).mean() * 100)}
        out["results"][f"K{K}_{bits}bit"] = r
        print(f"{K:>4} {bits:>5} | {r['mean']:16.2f} | {r['min']:6.1f} | {r['identical']:18.1f}")
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n>> saved {args.output}")
    if args.dump_maps:
        torch.save(torch.cat(all_maps), args.dump_maps)
        print(f">> saved F maps to {args.dump_maps}")


if __name__ == "__main__":
    main()
