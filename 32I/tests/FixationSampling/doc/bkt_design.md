# BKT — bucket-head instructions for the bucket-scan link pass

Status: **implemented (2026-10-10)** in `32I/` (`CPU::run_bkt`), 1-cycle latency.
Funct3 3–7 of the custom-3 opcode `0x7B`; BLK (`blk_design.md`) uses funct3 0–2.
Tests: `../toksel_tests/run_toksel_tests.py`. Measured results: `../10bit/README.md`.

## 1. Why

The bucket-scan link pass keeps one linked-list head per score value in `head[1024]`.
In memory that costs 3 cycles per head to set everything to −1 (3.1k cycles), and 11
instructions per cell, of which 4 only exist to read and write `head[v]` (the head load,
the head store, and `slli`/`add` for `head + 2*v`).

## 2. State

| Part | Size | Hardware | Holds |
|---|---|---|---|
| `bktHead[v]` | 1024 × 12 bits | SRAM | first cell (0…4095) of bucket v |
| `bktValid[v]` | 1024 × 1 bit | flops | 1 = bucket v is non-empty |

- 1024 = one bucket per 10-bit score value; 12 bits = a cell index on the 64×64 grid.
- "Empty" (−1) is never stored. A read returns −1 while the valid bit is 0, using a mux
  on the read path. `bkt.clr` resets only the valid bits (one cycle, flops); the SRAM is
  never cleared.
- Reset value: every bucket empty. Independent of the BLK map: neither clear touches the other.

## 3. Instructions

| asm | encoding | effect |
|---|---|---|
| `bkt.clr` | `.insn r 0x7B, 3, 0, x0, x0, x0` | every bucket empty |
| `bkt.ins rd, rs1, rs2` | `.insn r 0x7B, 4, 0, rd, rs1, rs2` | `rd` = old head of bucket `rs2[9:0]` (−1 if empty); then head = `rs1[11:0]`, valid = 1 |
| `bkt.head rd, rs1` | `.insn r 0x7B, 5, 0, rd, rs1, x0` | `rd` = head of bucket `rs1[9:0]`, −1 if empty |

- Upper bits of the cell index and the score are ignored.
- Funct3 6 is reserved for `bkt.next` (highest non-empty bucket ≤ v: a priority encoder
  over the valid bits, to skip empty buckets in the walk). Funct3 7 is reserved.
  Both print an error and write nothing.

The link pass becomes one score load, one `bkt.ins` and one `next_cell` store per cell:

```c
bkt_clr();
for (int i = FX_CELLS - 1; i >= 0; i--)
    next_cell[i] = (short)bkt_ins(i, FX_MAP[i]);
...
for (int best = bkt_head(v); best != END_OF_LIST && n < k; best = next_cell[best])
```

## 4. Emulator integration

Same pattern as BLK (`blk_design.md` §6):
- **State:** `CPU::bktHead[1024]` and `CPU::bktValid[1024]`.
- **Dispatch:** `CPU::run_toksel()` sends funct3 3–7 to `run_bkt()`, from `read()` / `read_m()`;
  the result replaces the ALU result and is committed to `rd`.
- **Superscalar:** `reg_usage` is rs1 for all of them, plus rs2 for `bkt.ins`; rs2 is read in
  `read_m()`, like NPU/LAUNCH. `mem_class` gives `MEM_BKT`, so one BKT instruction per window. It's
  a separate class from `MEM_BLK`, so a `blk` and a `bkt.*` can share a window. Squashed slots
  never run (`toksel_spec.c`).
- **GPU:** not supported, host-CPU only (see `blk_design.md` §6).
- **C API:** `bkt_clr()`, `bkt_ins(cell, score)` and `bkt_head(score)` in `../toksel_isa.h`.
