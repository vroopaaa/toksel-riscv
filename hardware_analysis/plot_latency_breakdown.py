#!/usr/bin/env python3
"""
Stacked horizontal bar chart: per-stage latency of the dense ViT, SparseViT-Det with the
original token selection, and SparseViT-Det after the software optimizations.

Data: paper_latency_results.json (profile_paper_latency.py; one run, same images, round-robin
over configs, batch 1, CUDA events). Stages are grouped into five categories shared by all
pipelines so one legend covers every bar.

Writes figures/latency_breakdown.pdf (for the paper) and .png (for a quick look).
"""

import os
import json
import argparse

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

BARS = [  # (label, config key in the results file), top -> bottom
    ("Dense ViT", "dense"),
    ("SparseViT-Det", "sparse_original"),
    ("SparseViT-Det (opt.)", "sparse_optimized_{algo}"),
]

# Pipeline order = stacking order. Colours are the validated categorical slots
# in this exact adjacency order (scripts/validate_palette.js, light mode: PASS).
STAGES = [
    ("Guidance CNN",                 "#2a78d6", ["cnn"]),
    ("Token selection",              "#eb6834", ["token_selection"]),
    ("Patch extraction & embedding", "#1baf7a", ["patch_extract", "patch_embed", "pos_embed", "pos_interp",
                                                 "fourier_pos", "score_embed", "global_tokens"]),
    ("Transformer blocks",           "#eda100", ["transformer"]),
    ("Detection head",               "#e87ba4", ["head"]),
]

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def grouped(stats):
    return {name: sum(stats[k]["mean"] for k in keys if k in stats) for name, _, keys in STAGES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(SCRIPT_DIR, "paper_latency_results.json"))
    ap.add_argument("--algo", default="bucketscan", help="which optimized config to plot")
    args = ap.parse_args()

    with open(args.results) as f:
        res = json.load(f)["results"]
    rows = [(label, grouped(res[key.format(algo=args.algo)])) for label, key in BARS][::-1]  # bottom -> top

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 1.85))              # IEEE single column

    bar_h = 0.55
    for y, (_, stages) in enumerate(rows):
        left = 0.0
        for name, color, _ in STAGES:
            w = stages[name]
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=bar_h, color=color,
                    edgecolor="white", linewidth=0.8)
            # value inside a segment only where the text fits
            if w >= 4.0:
                ax.text(left + w / 2, y, f"{w:.2f}", ha="center", va="center",
                        fontsize=7, color=INK)
            left += w
        ax.text(left + 0.8, y, f"{left:.2f} ms", ha="left", va="center",
                fontsize=8, color=INK, fontweight="bold")

    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Latency (ms)", color=INK_2)
    ax.set_xlim(0, max(sum(r[1].values()) for r in rows) * 1.18)
    ax.set_ylim(-0.55, len(rows) - 0.45)

    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=INK_2, width=0.6)

    # matplotlib fills legend columns first; this order makes the rows read in pipeline order
    legend_order = [STAGES[i] for i in (0, 3, 1, 4, 2)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in legend_order]
    ax.legend(handles, [n for n, _, _ in legend_order], loc="lower center",
              bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, fontsize=7,
              handlelength=1.0, handleheight=0.8, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.2)

    fig.tight_layout(pad=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"latency_breakdown.{ext}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print(f">> saved {OUT_DIR}/latency_breakdown.pdf and .png")
    for label, stages in rows[::-1]:
        print(f"   {label:<22}", "  ".join(f"{k}={v:.2f}" for k, v in stages.items()),
              f"  total={sum(stages.values()):.2f} ms")


if __name__ == "__main__":
    main()
