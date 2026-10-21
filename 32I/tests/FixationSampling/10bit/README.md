# 10-bit greedy token selection on RISC-V

Picks up to K cells from a D×D score map: the highest-scoring free cell (ties: lower index),
then blocks its 3×3 neighbourhood, until K picks or no free cell. Returns n ≤ K.
The spec and test data come from the handoff (`hardware_analysis/greedy_c/RISCV_HANDOFF.md`).

| File | |
|---|---|
| `greedy_naive.c` | full scan for every pick |
| `greedy_sortscan.c` | counting sort over 1024 scores, then walk the sorted list |
| `greedy_bucketscan.c` | one linked list per score value, walk 1023 → 0 |
| `greedy_bucketscan_blk.c` | bucket-scan with the BLK instructions (`../toksel_isa.h`) |
| `greedy_bucketscan_bkt.c` | bucket-scan with BLK + BKT bucket-head instructions |
| `greedy_config.h` | map/K selection; defaults so `emul 32I greedy_x.c -O2` runs standalone |
| `data/sampler_testdata_{32x32,64x64}.h` | copied from `hardware_analysis/sampler_testdata_10bit_*`: 3 COCO maps + golden picks |
| `run_cycles.py` | sweep + golden check → `results/cycles.{csv,md}` (`--width N`: superscalar) |

## Score storage

One score per `uint16_t` (upper 6 bits zero), as in the test data. Reading a score is a
single `lhu`.

Packing would save memory (3 per word: 1,368 B instead of 2,048 B on 32×32), but this
emulator counts **one cycle per instruction** and does not model memory size or latency.
Packing can only add instructions here (the shift and mask). Revisit it if the map has to fit in a
fixed-size buffer.

## Running

```
python3 run_cycles.py                        # all algos × {32,64} × K {128,256} × 3 images × -O0/1/2/3/s
python3 run_cycles.py --opts O2 --grids 32   # subset
python3 run_cycles.py --width 4 --opts O2    # superscalar, issue width 4
python3 ../toksel_tests/run_toksel_tests.py  # BLK/BKT instruction tests
```

Each run checks n (a0) and every pick against the golden arrays; the script exits 1 on any mismatch.

## Results (-O2, mean over 3 images)

Full tables for every -O level: [results/cycles.md](results/cycles.md) (scalar) and
[results/cycles_superscalar_w4.md](results/cycles_superscalar_w4.md). Every run matches the golden picks.

| scalar | 32×32 K=128 | 32×32 K=256 | 64×64 K=128 | 64×64 K=256 |
|---|---:|---:|---:|---:|
| naive | 956,447 | 1,468,793 | 4,084,202 | 7,952,790 |
| sortscan | 44,177 | 54,645 | 106,836 | 121,310 |
| bucketscan | 40,225 | 61,191 | 74,028 | 103,619 |
| bucketscan + BLK | 22,523 | 29,116 | 53,614 | 59,535 |
| bucketscan + BLK + BKT | **16,371** | **22,965** | **38,246** | **44,167** |
| superscalar width 4: bucketscan | 18,940 | 31,480 | 31,626 | 48,926 |
| superscalar width 4: bucketscan + BLK | 8,460 | 10,989 | 19,660 | 21,941 |
| superscalar width 4: bucketscan + BLK + BKT | **6,408** | **8,938** | **14,536** | **16,817** |

### 2026-10-10: baseline

- Bucket-scan is best on 64×64. On 32×32 with K=256, sort-scan wins (54.6k vs 61.2k). The
  greedy runs out of cells at n ≈ 205, so the walk goes all the way to the end. With 1024
  buckets for 1024 cells, bucket-scan then pays for every bucket head and every list hop.
- -O0 costs 2.5–4× more than -O2. -O3 is 5–19 % below -O2 for sort/bucket-scan; naive barely moves.
- Bucket-scan phases (one image, K=128): zeroing the blocked bitmap is only 99 / 259 cycles
  (32×32 / 64×64); the link pass is 14.4k / 48.2k; the walk with its tests and stamps is 25.5k / 26.1k.

### 2026-10-10: BLK (custom-3 `0x7B`, 1 cycle, `../doc/blk_design.md`)

- `blk(idx)` tests a cell and claims its 3×3 neighbourhood in one instruction; `blk_clr(D)` clears the map.
  The walk drops from 25.5k / 26.1k to 7.8k / 5.8k cycles (same image, K=128).
- Scalar: −44 % / −52 % on 32×32, −28 % / −43 % on 64×64 (K=128 / K=256).
- The link pass is now 65 % (32×32) and 89 % (64×64) of the total. It is the next target.

### 2026-10-10: link-pass analysis (one image, K=128, -O2)

- Cost = 3 cycles per bucket head (the `head[] = -1` init, 3,072 for 1024 buckets) + **11 per cell**.
  Per cell: 4 memory ops (load score, load head, store next, store head), 2 for the
  `head + 2*v` address, 5 loop bookkeeping (three induction variables, `mv`, `bne`).
- `-funroll-loops`: link 14.4k → 9.0k (32×32), 48.2k → 32.0k (64×64). -O3 alone: −3 to −11 %.
- Superscalar: ~4.25 cycles/cell at width 4 *and* 10. The limit is the dependency chain
  score → address → head load → store (no forwarding inside a window), not the width.
- The walk only needs part of the sorted order: 64×64 uses 6–8 % of cells at K=128
  and 16–27 % at K=256, yet the link pass touches all 4096. 32×32 uses 39–48 % at K=128
  and 98–100 % at K=256.

### 2026-10-10: BKT bucket heads (custom-3 funct3 3–5, 1 cycle, `../doc/bkt_design.md`)

- `head[]` moves into dedicated storage: 1024 × 12-bit heads plus 1024 valid bits. `bkt_clr()`
  replaces the −1 init, `bkt_ins(i, score)` returns the old head and pushes i, `bkt_head(v)`
  replaces the `head[v]` load in the walk.
- Link pass per cell: 11 → 8 instructions (`lhu`, `bkt.ins`, `sh`, plus 5 loop bookkeeping).
  Link 14.4k → 8.2k (32×32), 48.2k → 32.8k (64×64); walk 7.8k / 5.8k (one image, K=128).
- Scalar vs BLK: −21 to −29 %. Vs plain bucket-scan: −48 to −62 %.
- Next: the link pass is still 51 % (32×32) and 85 % (64×64) of the total, and 5 of its 8
  instructions per cell are loop bookkeeping, so unrolling is the obvious next step. On
  32×32 the walk reads 700–1024 bucket heads, many of them empty (356–481 distinct scores);
  `bkt.next` (funct3 6, reserved) would skip the empty ones.
- At -O0 the BKT version is slightly slower than BLK on 64×64 (179k vs 167k): at -O0 the
  `toksel_isa.h` wrappers are real function calls, and the link pass now makes one per cell.
