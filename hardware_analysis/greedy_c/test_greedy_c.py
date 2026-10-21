#!/usr/bin/env python3
"""
Checks the C greedy samplers (naive, sort-scan, bucket-scan) against the golden picks in
../sampler_testdata for every image and K = 64, 128, 256.

  native  gcc build with a small harness that prints n and picks[]
  rv32im  riscv64-unknown-elf-gcc -march=rv32im bare-metal build run under qemu-riscv32,
          using a minimal _start that runs the file's own main() and prints its checksum
          through the Linux write/exit syscalls (no libc needed)

The C files are compiled unchanged; main is renamed with -Dmain=sampler_main.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "sampler_testdata")
SAMPLERS = {
    "greedy_naive.c": "greedy_fixation",
    "greedy_sortscan.c": "greedy_sortscan",
    "greedy_bucketscan.c": "greedy_bucketscan",
}
IMAGES = ["img_000000039769", "img_000000000139", "img_000000000785"]
KS = [64, 128, 256]

NATIVE_HARNESS = r"""
#include <stdio.h>
extern int picks[];
int ALGO(int k);
int main(void) {
    int n = ALGO(NUM_FIXATIONS);
    printf("%d", n);
    for (int i = 0; i < n; i++) printf(" %d", picks[i]);
    printf("\n");
    return 0;
}
"""

RV32_CRT = r"""
int sampler_main(void);
static void sys_write(const char *buf, int len) {
    register long a0 asm("a0") = 1, a1 asm("a1") = (long)buf, a2 asm("a2") = len, a7 asm("a7") = 64;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
}
static void sys_exit(int code) {
    register long a0 asm("a0") = code, a7 asm("a7") = 93;
    asm volatile("ecall" : : "r"(a0), "r"(a7));
    for (;;) {}
}
void *memset(void *d, int v, unsigned long n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)v; return d; }
void _start(void) {
    unsigned int x = (unsigned int)sampler_main();
    char buf[12]; int i = 11; buf[i] = '\n';
    do { buf[--i] = (char)('0' + x % 10u); x /= 10u; } while (x);
    sys_write(buf + i, 12 - i);
    sys_exit(0);
}
"""


def golden(img, K):
    with open(os.path.join(TESTDATA, f"{img}_greedy_K{K}.txt")) as f:
        return [int(line.split()[1]) for line in f if not line.startswith("#")]


def checksum(picks):
    cs = len(picks)
    for p in picks:
        cs = (cs * 31 + p) & 0xFFFFFFFF
    return cs


def write_header(path, img):
    m = np.loadtxt(os.path.join(TESTDATA, f"{img}_F_u8.txt"), dtype=int)
    with open(path, "w") as f:
        f.write(f"#define FIX_DIM {m.shape[0]}\n#define FIX_CELLS {m.size}\n")
        f.write("static const unsigned char FIX_MAP[FIX_CELLS] = {" + ",".join(map(str, m.ravel())) + "};\n")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main():
    have_rv = shutil.which("riscv64-unknown-elf-gcc") and shutil.which("qemu-riscv32")
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "harness.c"), "w") as f:
            f.write(NATIVE_HARNESS)
        with open(os.path.join(tmp, "crt.c"), "w") as f:
            f.write(RV32_CRT)
        print(f"{'file':<22} {'image':<18} {'K':>4} | {'native == golden':>16} | {'rv32im checksum == golden':>25}")
        print("-" * 96)
        for src, fn in SAMPLERS.items():
            for img in IMAGES:
                write_header(os.path.join(tmp, "fixation_matrix.h"), img)
                for K in KS:
                    gold = golden(img, K)
                    common = ["-O2", "-w", f"-DNUM_FIXATIONS={K}", f"-I{tmp}"]
                    obj, exe = os.path.join(tmp, "sampler.o"), os.path.join(tmp, "nat")
                    run(["gcc", *common, "-Dmain=sampler_main", "-c", os.path.join(HERE, src), "-o", obj])
                    run(["gcc", *common, f"-DALGO={fn}", obj, os.path.join(tmp, "harness.c"), "-o", exe])
                    out = list(map(int, run([exe]).stdout.split()))
                    ok_nat = out[0] == len(gold) and out[1:] == gold
                    ok_rv = "skipped (no toolchain)"
                    if have_rv:
                        rv = os.path.join(tmp, "rv")
                        run(["riscv64-unknown-elf-gcc", *common, "-Dmain=sampler_main", "-march=rv32im", "-mabi=ilp32",
                             "-nostdlib", "-static", os.path.join(HERE, src), os.path.join(tmp, "crt.c"), "-o", rv])
                        ok_rv = int(run(["qemu-riscv32", rv]).stdout.strip()) == checksum(gold)
                    all_ok &= ok_nat and ok_rv is not False
                    print(f"{src:<22} {img:<18} {K:>4} | {str(ok_nat):>16} | {str(ok_rv):>25}")
    print(f"\n>> all builds match the golden picks: {all_ok}")


if __name__ == "__main__":
    main()
