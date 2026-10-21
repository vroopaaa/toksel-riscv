#!/usr/bin/env python3
"""
Tests for the custom-3 token-selection instructions (opcode 0x7B): BLK block map
(../doc/blk_design.md) and BKT bucket heads (../doc/bkt_design.md).

Every program runs at -O0 and -O2 on the scalar CPU
and on the superscalar CPU at widths 1, 2, 4, 10:
  blk_model.c      BLK corner cases + 6000 random ops against a plain-C model   a0 == 0
  bkt_model.c      BKT named cases + 6000 random ops (BLK mixed in) vs a model  a0 == 0
  toksel_spec.c    blk / bkt.ins / bkt.clr on a mispredicted path change nothing a0 == 0
  toksel_errors.c  blk.clr 48 and reserved funct3 2, 6, 7 are ignored           a0 == 0 + stderr
Then greedy_bucketscan_blk.c and greedy_bucketscan_bkt.c must give the golden picks (both
grids, both K, all images) on the superscalar CPU at the same widths (the scalar sweep is
../10bit/run_cycles.py).

Exit code 1 if anything fails.
"""

import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))  # -> 32I
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "10bit"))
import run_cycles  # noqa: E402  (same build flags, harness and golden check)

WIDTHS = [0, 1, 2, 4, 10]           # 0 = scalar pipeline

TESTS = [
    # file, opts, check(a0) , stderr substrings that must appear
    ("blk_model.c",     ["O0", "O2"], lambda a0: a0 == 0, []),
    ("bkt_model.c",     ["O0", "O2"], lambda a0: a0 == 0, []),
    ("toksel_spec.c",   ["O0", "O2"], lambda a0: a0 == 0, []),
    ("toksel_errors.c", ["O0", "O2"], lambda a0: a0 == 0,
     ["blk.clr with grid width 48", "reserved BLK funct3 2", "reserved BKT funct3 6",
      "reserved BKT funct3 7"]),
]


def run_program(work_dir, c_file, opt, width):
    name = f"{os.path.splitext(c_file)[0]}_{opt}_w{width}"
    elf_path = os.path.join(work_dir, name + ".elf")
    bin_path = os.path.join(work_dir, name + ".bin")
    run_cycles.run(["riscv64-unknown-elf-gcc", *run_cycles.RISCV_FLAGS, f"-{opt}",
                    "-T", run_cycles.LINK_LD,
                    "-o", elf_path, run_cycles.START_S, os.path.join(SCRIPT_DIR, c_file)])
    run_cycles.run(["riscv64-unknown-elf-objcopy", "-O", "binary", elf_path, bin_path])
    cmd = [run_cycles.HARNESS, bin_path, "50000000"]
    if width:
        cmd += ["0", "0", "", "superscalar", str(width)]
    result = run_cycles.run(cmd)
    final = result.stdout[result.stdout.rindex("--- Final State ---"):]
    a0 = int(re.search(r"x10:\s*0x([0-9a-fA-F]+)", final).group(1), 16)
    return a0, result.stderr


def main():
    run_cycles.run(["make", "build/run_test"], cwd=PROJECT_ROOT)
    failures = 0
    total = 0
    with tempfile.TemporaryDirectory() as work_dir:
        for c_file, opts, check, needles in TESTS:
            for opt in opts:
                for width in WIDTHS:
                    total += 1
                    a0, stderr = run_program(work_dir, c_file, opt, width)
                    missing = [s for s in needles if s not in stderr]
                    if not check(a0) or missing or (not needles and stderr):
                        failures += 1
                        print(f"FAIL {c_file} -{opt} width {width or 'scalar'}: a0={a0:#x} "
                              f"missing stderr {missing}\n{stderr}")

        for grid in run_cycles.GRIDS:
            golden = run_cycles.load_golden(grid)
            for (image, k) in sorted(golden):
                for algo in ["greedy_bucketscan_blk", "greedy_bucketscan_bkt"]:
                    for width in WIDTHS[1:]:
                        total += 1
                        r = run_cycles.run_one(work_dir, algo, grid, k, image, "O2", golden, width)
                        if not r["ok"]:
                            failures += 1
                            print(f"FAIL {algo} {grid}x{grid} K={k} img {image} width {width}: n={r['n']}")

    print(f"{total - failures}/{total} token-selection tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
