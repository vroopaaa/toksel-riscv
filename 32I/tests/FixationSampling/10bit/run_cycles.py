#!/usr/bin/env python3
"""
Cycle sweep for the 10-bit greedy token selectors (naive, sort-scan, bucket-scan, and
bucket-scan with the BLK block-map instructions).

For every algorithm x grid (32, 64) x K (128, 256) x image x -O level it:
  1. writes a tiny wrapper .c that includes data/sampler_testdata_<D>x<D>.h, picks the
     image's map (FX_MAP) and K (NUM_FIXATIONS), then #includes the algorithm file,
  2. cross-compiles it bare-metal for rv32im with tests/python/start.s + link.ld
     (same flags as scripts/emul),
  3. runs it on build/run_test (scalar pipeline, or superscalar with --width N),
     dumping picks[] (address from nm),
  4. checks n (a0) and every pick against the golden arrays in the same data header.

Writes results/cycles.csv (one row per run) and results/cycles.md (mean cycles over the
three images, one table per grid/K); results/cycles_superscalar_w<N>.* with --width N.
Exit code 1 if any run mismatches the golden picks.

--data-dir DIR uses one header per image instead, DIR/<D>x<D>/img_<id>.h (from
hardware_analysis/export_sampler_testdata_per_image.py, e.g. the 300 COCO images), and
writes results/cycles_<basename DIR>.* (…_superscalar_w<N>_<basename DIR>.* with --width).

Usage:
  python3 run_cycles.py                       # everything
  python3 run_cycles.py --opts O2 --grids 32  # a subset
  python3 run_cycles.py --width 4             # superscalar, issue width 4
  python3 run_cycles.py --opts O2 --algos greedy_bucketscan greedy_bucketscan_blk greedy_bucketscan_bkt \
      --data-dir <hardware_analysis>/sampler_testdata_10bit_coco300
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))  # -> 32I
START_S = os.path.join(PROJECT_ROOT, "tests", "python", "start.s")
LINK_LD = os.path.join(PROJECT_ROOT, "tests", "python", "link.ld")
HARNESS = os.path.join(PROJECT_ROOT, "build", "run_test")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

RISCV_FLAGS = ["-march=rv32im", "-mabi=ilp32", "-nostdlib", "-nostartfiles", "-ffreestanding"]
MAX_CYCLES = 400_000_000

ALGOS = ["greedy_naive", "greedy_sortscan", "greedy_bucketscan", "greedy_bucketscan_blk",
         "greedy_bucketscan_bktonly", "greedy_bucketscan_bkt"]
GRIDS = [32, 64]
KS = [128, 256]
OPTS = ["O0", "O1", "O2", "O3", "Os"]

DATA_DIR = None  # --data-dir: one header per image instead of data/sampler_testdata_<D>x<D>.h


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result


def data_header(grid, image):
    if DATA_DIR:
        return os.path.join(DATA_DIR, f"{grid}x{grid}", f"img_{image}.h")
    return os.path.join(SCRIPT_DIR, "data", f"sampler_testdata_{grid}x{grid}.h")


def load_golden(grid):
    """{(image_id, K): [picks]} parsed from the data header(s)' greedy arrays."""
    if DATA_DIR:
        grid_dir = os.path.join(DATA_DIR, f"{grid}x{grid}")
        paths = [os.path.join(grid_dir, f) for f in sorted(os.listdir(grid_dir)) if f.endswith(".h")]
    else:
        paths = [data_header(grid, None)]
    golden = {}
    for path in paths:
        with open(path) as f:
            text = f.read()
        for m in re.finditer(r"img_(\d+)_greedy_K(\d+)\[\d+\]\s*=\s*\{([^}]*)\}", text):
            golden[(m.group(1), int(m.group(2)))] = [int(x) for x in m.group(3).split(",") if x.strip()]
    return golden


def symbol_address(elf_path, symbol):
    for line in run(["riscv64-unknown-elf-nm", elf_path]).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == symbol:
            return int(parts[0], 16)
    raise RuntimeError(f"symbol {symbol} not found in {elf_path}")


def run_one(work_dir, algo, grid, k, image, opt, golden, width=0):
    """width 0 = scalar pipeline, N > 0 = superscalar with issue width N."""
    name = f"{algo}_{grid}_K{k}_{image}_{opt}_w{width}"
    c_path = os.path.join(work_dir, name + ".c")
    elf_path = os.path.join(work_dir, name + ".elf")
    bin_path = os.path.join(work_dir, name + ".bin")
    dump_path = os.path.join(work_dir, name + ".dump")
    with open(c_path, "w") as f:
        f.write(f'#include "{data_header(grid, image)}"\n'
                f"#define FX_MAP img_{image}_F_q10\n"
                f"#define NUM_FIXATIONS {k}\n"
                f'#include "{os.path.join(SCRIPT_DIR, algo + ".c")}"\n')
    run(["riscv64-unknown-elf-gcc", *RISCV_FLAGS, f"-{opt}", "-T", LINK_LD,
         "-o", elf_path, START_S, c_path])
    run(["riscv64-unknown-elf-objcopy", "-O", "binary", elf_path, bin_path])

    picks_addr = symbol_address(elf_path, "picks")
    cmd = [HARNESS, bin_path, str(MAX_CYCLES), f"{picks_addr:x}", f"{4 * k:x}", dump_path]
    if width:
        cmd += ["superscalar", str(width)]
    out = run(cmd).stdout
    final = out[out.rindex("--- Final State ---"):]
    cycles = int(re.search(r"Cycles:\s*(\d+)", final).group(1))
    n = int(re.search(r"x10:\s*0x([0-9a-fA-F]+)", final).group(1), 16)
    with open(dump_path) as f:
        picks = [int(line.split(":")[1], 16) for line in f if ":" in line][:n]

    expected = golden[(image, k)]
    ok = cycles < MAX_CYCLES and n == len(expected) and picks == expected
    return {"algo": algo, "grid": grid, "K": k, "image": image, "opt": opt,
            "n": n, "cycles": cycles, "ok": ok}


def write_markdown(rows, path, width, csv_name):
    pipeline = f"Superscalar pipeline, issue width {width}" if width else "Scalar pipeline"
    num_images = len({r["image"] for r in rows})
    source = f"`{os.path.basename(DATA_DIR)}`" if DATA_DIR else "the test"
    lines = ["# Cycles: 10-bit greedy token selection",
             "",
             f"Generated by `run_cycles.py`. {pipeline}, mean over {source} images (up to {num_images} per grid)",
             f"(per-image numbers in `{csv_name}`). Every run matched the golden picks"
             if all(r["ok"] for r in rows) else f"**Some runs did NOT match the golden picks, see {csv_name}**",
             ""]
    opts = [o for o in OPTS if any(r["opt"] == o for r in rows)]
    for grid in GRIDS:
        for k in KS:
            sel = [r for r in rows if r["grid"] == grid and r["K"] == k]
            if not sel:
                continue
            ns = sorted({r["n"] for r in sel})
            lines += [f"## {grid}x{grid}, K = {k} (n = {', '.join(map(str, ns))})", "",
                      "| algorithm | " + " | ".join(f"-{o}" for o in opts) + " |",
                      "|---|" + "---:|" * len(opts)]
            for algo in ALGOS:
                cells = []
                for o in opts:
                    c = [r["cycles"] for r in sel if r["algo"] == algo and r["opt"] == o]
                    cells.append(f"{round(sum(c) / len(c)):,}" if c else "")
                if any(cells):
                    lines.append(f"| {algo.replace('greedy_', '')} | " + " | ".join(cells) + " |")
            lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--grids", nargs="+", type=int, default=GRIDS, choices=GRIDS)
    ap.add_argument("--ks", nargs="+", type=int, default=KS, choices=KS)
    ap.add_argument("--opts", nargs="+", default=OPTS, choices=OPTS)
    ap.add_argument("--width", type=int, default=0, help="superscalar issue width (default: scalar)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--data-dir", default=None, help="per-image headers, DIR/<D>x<D>/img_<id>.h")
    args = ap.parse_args()
    global DATA_DIR
    DATA_DIR = os.path.abspath(args.data_dir.rstrip("/")) if args.data_dir else None

    run(["make", "build/run_test"], cwd=PROJECT_ROOT)
    golden = {g: load_golden(g) for g in args.grids}

    jobs = []
    for grid in args.grids:
        images = sorted({img for (img, _) in golden[grid]})
        for algo in args.algos:
            for k in args.ks:
                for image in images:
                    for opt in args.opts:
                        jobs.append((algo, grid, k, image, opt))

    with tempfile.TemporaryDirectory() as work_dir, ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda j: run_one(work_dir, *j, golden[j[1]], args.width), jobs))

    for r in rows:
        if not r["ok"]:
            print(f"MISMATCH {r['algo']} {r['grid']}x{r['grid']} K={r['K']} img {r['image']} -{r['opt']}: "
                  f"n={r['n']} cycles={r['cycles']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = f"cycles_superscalar_w{args.width}" if args.width else "cycles"
    if DATA_DIR:
        stem += "_" + os.path.basename(DATA_DIR)
    with open(os.path.join(RESULTS_DIR, stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows, os.path.join(RESULTS_DIR, stem + ".md"), args.width, stem + ".csv")

    passed = sum(r["ok"] for r in rows)
    print(f"{passed}/{len(rows)} runs match the golden picks. Results in {RESULTS_DIR}/")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
