#!/usr/bin/env python3
"""
Export fixation-map test vectors for the RISC-V sampler.

For each image (resized to SIZE x SIZE -> G x G grid at stride 16; default 1024 -> 64x64,
--size 512 -> 32x32) writes:
  <name>_F_float.txt     64 rows x 64 cols, fused map in [0, 1]
  <name>_F_u8.txt        same map quantised to uint8: round(255 * F)            (--bits 8, default)
  <name>_F_q<B>.txt      same map quantised to B bits: round((2^B - 1) * F)     (--bits B > 8, stored
                         as uint16_t in C: values 0 .. 2^B - 1, upper bits zero)
  <name>_greedy_K<K>.txt golden greedy picks on the quantised map, in pick order
  sampler_testdata.h     all quantised maps and golden picks as C arrays (bare-metal friendly)
  preview.png            image | map | K=128 picks, for a visual check

Map source:
  default       proxy saliency: channel-wise L2 energy of ImageNet-pretrained MobileNetV3
                block-3 features (the trained S/O heads are not available locally)
  --checkpoint  real F = 0.6*norm(S) + 0.4*norm(O) from a trained SparseViTDetector
                checkpoint built from this repo's models/ (MobileNetV3 guidance CNN)

Greedy definition (golden): visit cells in descending quantised score, ties -> lower index
(i = G*r + c); accept a cell if no accepted cell is in its 3x3 neighbourhood; stop at K.
Does NOT modify any original files in models/.
"""

import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Saliency-Objectness-Guided-Sparse-ViT"))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

IMG = 1024
G = IMG // 16  # 64


def load_image(path: str) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None] / 255.0


@torch.no_grad()
def proxy_map(x: torch.Tensor) -> np.ndarray:
    import timm
    bb = timm.create_model("mobilenetv3_small_050", pretrained=True, features_only=True, out_indices=(3,)).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feat = bb((x - mean) / std)[0]  # (1, 24, 64, 64)
    return feat.pow(2).sum(1).sqrt()[0].numpy()


@torch.no_grad()
def model_map(x: torch.Tensor, ckpt: str) -> np.ndarray:
    from models.sparse_vit_det import SparseViTDetector
    m = SparseViTDetector(num_classes=80, pretrained=False, K=128, image_size=(IMG, IMG)).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    _, S, O = m.feature_extractor(x)
    norm = lambda t: (t - t.min()) / (t.max() - t.min() + 1e-6)
    return (0.6 * norm(S) + 0.4 * norm(O))[0, 0].numpy()


def greedy(F_u8: np.ndarray, K: int) -> list:
    order = np.argsort(-F_u8.ravel().astype(np.int32), kind="stable")
    blocked = np.zeros((G + 2, G + 2), bool)
    picks = []
    for i in order:
        r, c = divmod(int(i), G)
        if not blocked[r:r + 3, c:c + 3].any():
            blocked[r + 1, c + 1] = True
            picks.append(int(i))
            if len(picks) == K:
                break
    return picks


def check_against_pytorch(F_u8: np.ndarray, K: int, picks: list) -> bool:
    """Same picks as the repo's FixationSampler (d_min = 28 px) run on this map?"""
    from models.fixation_sampler import FixationSampler
    t = torch.from_numpy(F_u8.astype(np.float32))[None, None]
    coords, *_ = FixationSampler(K=K, alpha=1.0, beta=0.0, image_size=(IMG, IMG))(t, t, image_size=(IMG, IMG))
    ref = (((coords[0, :, 1] - 8) / 16).long() * G + ((coords[0, :, 0] - 8) / 16).long()).tolist()
    return ref[:len(picks)] == picks


def write_matrix(path: str, M: np.ndarray, header: list, fmt: str):
    with open(path, "w") as f:
        for h in header:
            f.write(f"# {h}\n")
        for row in M:
            f.write(" ".join(fmt.format(v) for v in row) + "\n")


def main():
    global IMG, G
    ap = argparse.ArgumentParser(description="Export sampler test vectors from real images")
    ap.add_argument("images", nargs="+", help="image files")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "sampler_testdata"))
    ap.add_argument("--checkpoint", default=None, help="trained SparseViTDetector checkpoint (real S/O maps)")
    ap.add_argument("--K", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--bits", type=int, default=8, help="score bit width (8 -> uint8_t, 9..16 -> uint16_t)")
    ap.add_argument("--size", type=int, default=1024, help="input image size; grid = size / 16")
    args = ap.parse_args()
    IMG, G = args.size, args.size // 16
    L = 2 ** args.bits - 1
    tag, ctype = ("u8", "uint8_t") if args.bits == 8 else (f"q{args.bits}", "uint16_t")
    os.makedirs(args.out, exist_ok=True)
    source = f"trained model {os.path.basename(args.checkpoint)}" if args.checkpoint else \
        "proxy: L2 energy of ImageNet-pretrained MobileNetV3-Small-0.5 block-3 features"

    h_lines = ["/* Generated by hardware_analysis/export_sampler_testdata.py */", "#include <stdint.h>",
               f"#define FX_GRID {G}", f"#define FX_BITS {args.bits}",
               f"#define FX_LEVELS {L + 1}   /* number of score values = buckets */", ""]
    previews = []
    for path in args.images:
        name = "img_" + os.path.splitext(os.path.basename(path))[0]
        x = load_image(path)
        raw = model_map(x, args.checkpoint) if args.checkpoint else proxy_map(x)
        F = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        F_u8 = np.round(F * L).astype(np.uint8 if args.bits == 8 else np.uint16)   # quantised map
        hdr = [f"source image: {os.path.basename(path)} resized to {IMG}x{IMG}", f"map: {source}",
               f"grid {G} x {G}, row r = 0..{G-1} top to bottom, col c = 0..{G-1} left to right, index i = {G}*r + c",
               f"cell (r, c) covers pixels [16r, 16r+16) x [16c, 16c+16), centre (16c+8, 16r+8)"]
        write_matrix(os.path.join(args.out, f"{name}_F_float.txt"), F, hdr + ["values: min-max normalised to [0, 1]"], "{:.6f}")
        write_matrix(os.path.join(args.out, f"{name}_F_{tag}.txt"), F_u8,
                     hdr + [f"values: {args.bits}-bit = round({L} * F_float), 0 .. {L}"], "{:%dd}" % len(str(L)))
        h_lines.append(f"static const {ctype} {name}_F_{tag}[{G * G}] = {{")
        h_lines += ["  " + ", ".join(str(v) for v in F_u8[r]) + "," for r in range(G)]
        h_lines.append("};")
        print(f"{name}: unique {args.bits}-bit values {len(np.unique(F_u8))}, cells tied with another cell "
              f"{(np.bincount(F_u8.ravel()) > 1)[F_u8.ravel()].mean() * 100:.1f}%")
        for K in args.K:
            picks = greedy(F_u8, K)
            ok = check_against_pytorch(F_u8, K, picks)
            with open(os.path.join(args.out, f"{name}_greedy_K{K}.txt"), "w") as f:
                f.write(f"# golden greedy picks on {name}_F_{tag}.txt, K = {K}, accepted = {len(picks)}\n")
                f.write("# order of acceptance; ties broken by lower index; 3x3 neighbourhood blocked after each pick\n")
                f.write(f"# matches repo FixationSampler (d_min = 28 px) on the same map: {ok}\n")
                f.write(f"# k  index  row  col  score_{tag}  pixel_x  pixel_y\n")
                for k, i in enumerate(picks):
                    r, c = divmod(i, G)
                    f.write(f"{k:3d} {i:5d} {r:4d} {c:4d} {F_u8[r, c]:9d} {16*c+8:8d} {16*r+8:8d}\n")
            h_lines.append(f"static const uint16_t {name}_greedy_K{K}[{len(picks)}] = {{ " + ", ".join(map(str, picks)) + " };")
            print(f"   K={K:3d}: accepted {len(picks)}, matches PyTorch FixationSampler: {ok}")
        h_lines.append("")
        previews.append((x[0].permute(1, 2, 0).numpy(), F, greedy(F_u8, 128), name))

    with open(os.path.join(args.out, "sampler_testdata.h"), "w") as f:
        f.write("\n".join(h_lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(previews), 3, figsize=(12, 4 * len(previews)), squeeze=False)
    for row, (img, F, picks, name) in zip(ax, previews):
        row[0].imshow(img); row[0].set_title(name)
        row[1].imshow(F, cmap="magma"); row[1].set_title(f"{G}x{G} map")
        row[2].imshow(img)
        ys, xs = zip(*[divmod(i, G) for i in picks])
        row[2].scatter([16 * c + 8 for c in xs], [16 * r + 8 for r in ys], s=10, c="cyan")
        row[2].set_title("greedy K=128 picks")
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "preview.png"), dpi=70)
    print(f">> wrote {args.out}")


if __name__ == "__main__":
    main()
