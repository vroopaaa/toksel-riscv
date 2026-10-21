#!/usr/bin/env python3
"""
Grouped bar chart of token-selection cycles: software bucket-scan vs + BLK, + BKT,
+ BLK + BKT, for each map size x K. Mean over the images in the CSV, reduction vs
software printed above each accelerated bar. Sized for one IEEE column (3.5 in).

Usage:
  python3 plot_cycles.py [results/cycles_sampler_testdata_10bit_coco300.csv] [--opt O2]
Writes results/toksel_cycles.pdf and .png next to the CSV.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "results", "cycles_sampler_testdata_10bit_coco300.csv")

# (algo, legend label, colour, hatch). Software is the neutral baseline; the three
# accelerated versions take the first three categorical slots. BKT gets a hatch so it
# stays distinct from BLK + BKT in grayscale print.
SERIES = [("greedy_bucketscan", "Software", "#9a9a96", ""),
          ("greedy_bucketscan_blk", "+ BLK", "#2a78d6", ""),
          ("greedy_bucketscan_bktonly", "+ BKT", "#eb6834", "////"),
          ("greedy_bucketscan_bkt", "+ BLK + BKT", "#1baf7a", "")]
CONFIGS = [(32, 128), (32, 256), (64, 128), (64, 256)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("--opt", default="O2")
    args = ap.parse_args()

    cycles = defaultdict(list)
    for r in csv.DictReader(open(args.csv)):
        if r["opt"] == args.opt:
            cycles[(r["algo"], int(r["grid"]), int(r["K"]))].append(int(r["cycles"]))
    mean = {key: sum(v) / len(v) for key, v in cycles.items()}

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.linewidth": 0.6, "hatch.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    width = 0.2
    for s, (algo, label, colour, hatch) in enumerate(SERIES):
        xs = [c + (s - 1.5) * width for c in range(len(CONFIGS))]
        ys = [mean[(algo, g, k)] / 1000 for g, k in CONFIGS]
        bars = ax.bar(xs, ys, width * 0.9, color=colour, hatch=hatch, edgecolor="white",
                      linewidth=0, label=label, zorder=2)
        for bar in bars:   # hatch drawn in a darker tone of the fill
            bar.set_edgecolor("#00000055" if hatch else colour)
        if s:
            for x, y, (g, k) in zip(xs, ys, CONFIGS):
                red = 100 * (1 - mean[(algo, g, k)] / mean[(SERIES[0][0], g, k)])
                ax.text(x, y + 1.5, f"{red:.0f}%", ha="center", va="bottom", rotation=90, fontsize=6, color="#333333")

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([f"{g}×{g}\nK = {k}" for g, k in CONFIGS])
    ax.set_ylabel("Cycles (thousands)")
    ax.set_ylim(0, max(mean.values()) / 1000 * 1.12)
    ax.yaxis.grid(True, linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, width=0.6)
    ax.legend(ncol=4, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(pad=0.3)

    out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "toksel_cycles")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
