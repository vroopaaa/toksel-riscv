# toksel-riscv

Hardware acceleration of greedy token selection (fixation sampling) for a sparse ViT
detector, on a RISC-V emulator. Two custom units sit in an RV32IM core under the
`custom-3` opcode:

- **BLK**: a 64×64-bit free map. `blk` tests a cell and, if it is free, blocks its 3×3
  neighbourhood in one cycle; `blk.clr` resets the map.
- **BKT**: on-core bucket heads (1024 × 12-bit SRAM plus one occupancy bit per bucket) for
  the bucket-sort linking pass. `bkt.ins`, `bkt.head` and `bkt.clr`.

On 300 COCO validation images (`-O2`), BLK alone cuts the cycle count of token selection
by 27–52 %, BKT alone by 10–21 %, and both together by 48–62 %, with the selected tokens
identical to the 10-bit software selection.

## Layout

| Path | What it is |
|---|---|
| `32I/` | The emulator: RV32I + M, an in-order superscalar mode, the BLK/BKT units (`src/CPU.cpp`, `run_blk`/`run_bkt`). Also contains an NPU and a SIMT GPU model, which this work does not use. |
| `32I/tests/FixationSampling/10bit/` | The selection algorithms in C (naive, sort-scan, bucket-scan, + BLK, + BKT, + both), `run_cycles.py` (cycle sweep + golden check), `plot_cycles.py`, results. Start with its `README.md`. |
| `32I/tests/FixationSampling/toksel_tests/` | Instruction-level tests for BLK/BKT against C models, superscalar speculation and error cases. |
| `32I/tests/FixationSampling/doc/` | Design notes for BLK and BKT, and the paper figure (`figures/`). |
| `32I/tests/FixationSampling/toksel_isa.h` | C wrappers for the custom instructions. |
| `scripts/` | `emul` (cross-compile and run one C file) and `assemblyinstruction` (disassemble or trace it). See `scripts/README.md`. |
| `hardware_analysis/` | The GPU-side analysis: stage-wise latency profiling, quantisation accuracy, bicubic interpolation rewrite, test-vector export. |

## Requirements

- `g++` (C++11) and `make`
- `riscv64-unknown-elf-gcc` / `objcopy` (bare-metal RV32IM cross-compiler)
- Python 3; `matplotlib` for the plot. `hardware_analysis/` also needs PyTorch, timm and numpy.

## Running

```
make -C 32I                                                # build the emulator
python3 32I/tests/FixationSampling/toksel_tests/run_toksel_tests.py   # BLK/BKT tests
cd 32I/tests/FixationSampling/10bit
python3 run_cycles.py --opts O2                            # cycle sweep on the 3 bundled images
python3 plot_cycles.py                                     # bar chart from the 300-image CSV
```

### The 300-image results

`results/cycles_sampler_testdata_10bit_coco300.csv` was produced from 300 COCO val2017
images whose IDs are listed in `hardware_analysis/coco_val2017_300_keys.txt`. The images
and the generated per-image headers are not in the repo. To regenerate them:

```
cd hardware_analysis
python export_sampler_testdata_per_image.py <images>/*.jpg --size 512  --bits 10 --K 128 256 --out sampler_testdata_10bit_coco300
python export_sampler_testdata_per_image.py <images>/*.jpg --size 1024 --bits 10 --K 128 256 --out sampler_testdata_10bit_coco300
cd ../32I/tests/FixationSampling/10bit
python3 run_cycles.py --opts O2 --data-dir ../../../../hardware_analysis/sampler_testdata_10bit_coco300 \
    --algos greedy_bucketscan greedy_bucketscan_blk greedy_bucketscan_bktonly greedy_bucketscan_bkt
```

## Notes

- The emulator is functional, not cycle-accurate: every instruction, including each load
  and store to a flat, cache-less memory, takes one cycle.
- The Python scripts in `hardware_analysis/` import the detector code from
  `../Saliency-Objectness-Guided-Sparse-ViT`, which is a separate repository. Its
  `run_*.sh` wrappers still `cd` to the original absolute path on the author's machine.
- `make -C 32I run-test` does not work: `tests/basic/` has the harness but no `start.s`,
  `main.c` or `link.ld`. The working correctness harness is `32I/tests/python/run_tests.py`.
- This repository was split out of the `cuda_prototype` branch of the RSVP emulator.
