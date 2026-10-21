#!/usr/bin/env python3
"""
Grouped horizontal bar chart: detailed per-stage latency of SparseViT-Det before and after the
software optimizations (exact host-side greedy token selection + GEMM fast bicubic).

Data: paper_latency_results.json (profile_paper_latency.py; same run as plot_latency_breakdown.py).
Writes figures/sparse_stage_comparison.pdf (for the paper) and .png (for a quick look).
"""

import os
import json
import argparse

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

SERIES = [  # (legend label, config key, colour)
    ("SparseViT-Det", "sparse_original", "#a3a29c"),          # muted neutral: the baseline
    ("SparseViT-Det (opt.)", "sparse_optimized_{algo}", "#2a78d6"),
]

STAGES = [  # (row label, result key, optimized here?) in pipeline order, top -> bottom
    ("Guidance CNN",             "cnn",             False),
    ("Token selection",          "token_selection", True),
    ("Patch extraction",         "patch_extract",   False),
    ("Patch embedding",          "patch_embed",     False),
    ("Positional interpolation", "pos_interp",      True),
    ("Fourier positional MLP",   "fourier_pos",     False),
    ("Fixation-score MLP",       "score_embed",     False),
    ("Global tokens",            "global_tokens",   False),
    ("Transformer blocks",       "transformer",     False),
    ("Detection head",           "head",            False),
]

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(SCRIPT_DIR, "paper_latency_results.json"))
    ap.add_argument("--algo", default="bucketscan", help="which optimized config to plot")
    args = ap.parse_args()

    with open(args.results) as f:
        res = json.load(f)["results"]
    data = [res[key.format(algo=args.algo)] for _, key, _ in SERIES]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 3.0))               # IEEE single column

    n = len(STAGES)
    bar_h, gap = 0.36, 0.04
    xmax = max(d[k]["mean"] for d in data for _, k, _ in STAGES)
    for i, (label, key, optimized) in enumerate(STAGES):
        y0 = n - 1 - i                                        # first stage on top
        vals = [d[key]["mean"] for d in data]
        for j, ((_, _, color), v) in enumerate(zip(SERIES, vals)):
            y = y0 + (bar_h + gap) / 2 * (1 if j == 0 else -1)
            ax.barh(y, v, height=bar_h, color=color, edgecolor="white", linewidth=0.5)
            ax.text(v + xmax * 0.012, y, f"{v:.2f}", ha="left", va="center", fontsize=6.5, color=INK_2)
        if optimized:
            ax.text(xmax * 1.30, y0, f"{round(vals[0], 2) / round(vals[1], 2):.1f}$\\times$",
                    ha="right", va="center", fontsize=8, color=INK, fontweight="bold")

    ax.set_yticks(range(n), [s[0] for s in STAGES][::-1])
    ax.set_xlabel("Latency (ms)", color=INK_2)
    ax.set_xlim(0, xmax * 1.32)
    ax.set_ylim(-0.6, n - 0.4)

    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=INK_2, width=0.6)

    totals = [d["total"]["mean"] for d in data]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SERIES]
    labels = [f"{name}: {t:.2f} ms total" for (name, _, _), t in zip(SERIES, totals)]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1,
              frameon=False, fontsize=7, handlelength=1.0, handleheight=0.8,
              handletextpad=0.4, borderaxespad=0.2)

    fig.tight_layout(pad=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"sparse_stage_comparison.{ext}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print(f">> saved {OUT_DIR}/sparse_stage_comparison.pdf and .png")
    print(f"   {'stage':<26}{'original':>10}{'optimized':>11}")
    for label, key, _ in STAGES:
        print(f"   {label:<26}{data[0][key]['mean']:>10.2f}{data[1][key]['mean']:>11.2f}")
    print(f"   {'Total':<26}{totals[0]:>10.2f}{totals[1]:>11.2f}")


if __name__ == "__main__":
    main()
