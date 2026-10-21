# BLK — block-map instruction for greedy fixation sampling

Status: **implemented (2026-10-10)** in `32I/` (`CPU::run_blk`), 1-cycle latency.
Changes from the original proposal: `blk.clr` takes the grid width D (32 or 64) in `rs1`,
and the state is two `CPU` members, not a separate class. Tests: `../toksel_tests/run_toksel_tests.py`.
Measured results for the 10-bit data: `../10bit/README.md`. Sections 1 and 8 still
describe the older 8-bit, K=64 code.

## 1. Why

Greedy fixation sampling repeats: pick the best cell that is still pickable, then make that
cell and its 3×3 neighbourhood unpickable. After the sort/bucket step, the per-pick work is
this "test the cell, stamp the neighbourhood" operation.

In `greedy_bucketsort.c` it costs ~175 cycles per pick *(measured)*: a 9-iteration loop with
four bounds checks each, plus a div/mod to get row/column. In hardware it is one cell lookup
and nine bit-clears that all happen at once.

## 2. Architectural state

A hidden **valid map** of `64 × 64 = 4096` bits (512 bytes), used as a D × D grid with
D = 32 or 64 (set by `blk.clr`).

- `1` = cell still pickable, `0` = blocked.
- Not an architectural register: software reaches it only through BLK instructions, the way
  the NPU's data window is reachable only through its own instructions.
- Reset value: D = 64, all ones (everything pickable). `blk.clr D` makes all D × D cells pickable again.
- Fixed size, independent of K. Cell index `idx = D·r + c`: for D = 64, `r = idx[11:6]`,
  `c = idx[5:0]`; for D = 32, `r = idx[9:5]`, `c = idx[4:0]`.

## 3. Instructions

New custom-3 opcode `0x7B` (custom-0/1/2 = `0x0B`/`0x2B`/`0x5B` are taken), funct3 0-2. R-type.
funct3 3-7 of the same opcode are the bucket heads (`bkt_design.md`).

| asm | encoding | effect |
|---|---|---|
| `blk rd, rs1` | `.insn r 0x7B, 0, 0, rd, rs1, x0` | **test-and-claim**: `rd = valid[rs1]`; if it was 1, clear the 3×3 block around `rs1` (clipped at the map edges) |
| `blk.clr rs1` | `.insn r 0x7B, 1, 0, x0, rs1, x0` | grid width D = `rs1` (32 or 64), every cell pickable again. Any other width prints an error and is ignored |

Details:
- Only `rs1[11:0]` (D = 64) or `rs1[9:0]` (D = 32) is used; upper bits are ignored (hardware just doesn't wire them).
- If `valid[rs1] == 0`: `rd = 0`, and **nothing changes**. A blocked candidate must not
  extend the blocked area.
- The claimed cell itself is inside its own 3×3 block, so claiming it clears it, and a second
  `blk` of the same index returns 0.
- Funct3 2 (`blk.test`: read without stamping) is reserved. Greedy never needs it; add it only
  if another use appears.

## 4. Hardware micro-architecture

### 4.1 Flip-flop array (the model we emulate)

4096 flops, about the size of four RISC-V register files.

**Test.** Read one bit out of 4096: a 64:1 row mux followed by a 64:1 column mux. This is the
longest path in the block.

**Stamp.** Needs no divider and no bounds checks:
1. Split `idx` into `r` and `c` (just wire slices).
2. Decode each to a 64-wire one-hot.
3. Dilate each one-hot by OR-ing it with itself shifted ±1 (`rowdil = oh(r-1)|oh(r)|oh(r+1)`,
   same for columns). Shifting past the end drops the bit, so clipping at the edges is
   automatic.
4. Flop `[r'][c']` gets clear-enable `rowdil[r'] & colDil[c'] & valid[r][c]`, one AND per flop.

Everything happens in one cycle. `blk.clr` is a synchronous set of all flops, also one cycle.
If the mux-then-AND path limits the clock, the instruction can be split across two pipeline
stages with no change to the architecture.

### 4.2 Smaller alternative (not modelled, noted for RTL)

3-bank SRAM, 64-bit words, banked by `row mod 3`. Rows `r-1, r, r+1` always fall in three
different banks, so one stamp touches all three banks in one cycle. This needs bit-write-enable
SRAM (or a 2-cycle read-modify-write), and `blk.clr` becomes ~22 cycles (or an epoch-tag
trick). The architectural behaviour is identical, so the emulator model below covers both.

## 5. Cost model for the emulator

| op | cycles |
|---|---|
| `blk`, `blk.clr` | 1 issue slot, like an ALU op (flop version) |

The stamp costs nothing extra. An SRAM implementation would be 2 cycles for `blk`, so this is
a parameter to keep in one place.

## 6. Emulator integration (all in `32I/`, mirrors how NPU/LAUNCH were added)

As implemented:
- **State:** `CPU::blkValid[64]` (`uint64_t` rows, bit c = column c) and `CPU::blkDim`
  (not in `Memory`: it is instruction-accessed, not memory-mapped). The stamp is a 3-bit
  column mask cleared in three rows.
- **Decode:** `case OPCODE_TOKSEL` (`0x7B`, funct3 0-2) in both control switches: no scalar memory access,
  `reg_write_enable = true` for `blk`, false for `blk.clr`.
- **Work happens in `read()` / `read_m()`** through the shared `CPU::run_toksel()` → `run_blk()`, like LAUNCH.
  Execute passes `rs1` through; the read stage replaces the slot's ALU result with
  `run_blk()`'s return value, and writeback commits it to `rd` like any ALU result.
- **Superscalar mode:**
  - `reg_usage`: `blk` and `blk.clr` both read `rs1` only.
  - `mem_class`: a new class `MEM_BLK`. Two BLK instructions in one window would depend
    on each other's stamp through hidden state (a hazard the register RAW check cannot see),
    so at most one issues per window.
  - **Speculation is safe by construction.** `read_m()` skips cancelled slots
    (`windowCancelled`), so a `blk` in a mispredicted slot never stamps (`toksel_spec.c` checks
    this). Unlike LAUNCH, it does *not* need to issue alone.
  - `blk`'s result is written back at the end of the cycle, so a branch on `rd` always
    lands in the next window. That is the ordinary RAW rule, with no special case.
- **GPU:** not supported. The token-selection unit is a host-CPU-only experiment; nothing in
  `SIMT.cpp` knows about it (a lane that hits `0x7B` just faults as an unknown opcode).
- **Trace:** `assemblyinstruction -trace` tags `BLK` / `BLK.CLR`.
- **C API:** `../toksel_isa.h`, `blk(idx)` and `blk_clr(dim)`, as `volatile` asm with a
  `"memory"` clobber so the compiler can't reorder or delete calls: the effect is hidden state.

## 7. How the greedy code changes

Bucket walk, per candidate `best` (everything else is unchanged):

```c
blk_clr(FX_GRID);                       // instead of zeroing blocked_bits
...
for (best = head[v]; best != END_OF_LIST && n < k; best = next_cell[best])
    if (blk(best))                      // was valid; neighbourhood now blocked
        picks[n++] = best;
```

The `is_blocked` / `block_cell` helpers, the 9-iteration stamp loop, the bounds checks and the
`best / 64`, `best % 64` all disappear. A candidate costs a few instructions instead of ~175
cycles for a pick.

## 8. Expected impact (original 8-bit estimate)

Measured on the 10-bit data (scalar, -O2, K=128 / 256): bucket-scan 40.2k / 61.2k →
22.5k / 29.1k on 32×32, 74.0k / 103.6k → 53.6k / 59.5k on 64×64. Details: `../10bit/README.md`.
The estimate below holds: with BLK the link pass is what's left (65 % / 89 % of the total).

| | now *(measured)* | with BLK *(est.)* |
|---|---|---|
| bucket, K=64 | 57.3k | ~47k |
| bucket, K=256 | 95.8k | ~50k |
| sortscan, K=64 | 95.1k | ~86k |

BLK removes the per-pick cost but **not** the fixed pass over all 4096 cells (~46k bucket,
~85k sortscan *(measured, K=1)*). At K=64 BLK alone buys ~18%. It pays off properly only
after that pass is also fixed (SWAR threshold prefilter in software, or a streaming
histogram/bucket unit in hardware). BLK is the second half of the optimisation, not the first.

## 9. Test plan (done: `../toksel_tests/run_toksel_tests.py`)

`blk_model.c` covers 1 (plus 6000 random ops against a C model, at D = 32 and 64);
`greedy_bucketscan_blk.c` covers 2 against the 10-bit golden picks; `toksel_spec.c` and the
width sweep cover 3; `toksel_errors.c` covers bad widths and reserved funct3. Item 4
was dropped: the unit is host-CPU only.

1. **Instruction tests** (real assembled programs, per the repo's testing preference):
   corner cells 0, 63, 4032, 4095 (clipping), an edge column and row, double claim of the
   same cell (second returns 0), a blocked candidate leaving the map unchanged, `blk.clr`
   restoring everything, upper bits of `rs1` ignored.
2. **Golden check:** a BLK version of `greedy_bucketsort.c` must produce the K=64/128/256
   checksums from `check_golden.py` (`0x4c513263` at K=64).
3. **Superscalar:** same program at widths 1/2/4/10 gives identical results; a program with
   `blk` placed after a mispredicted branch must leave the map untouched.
4. ~~**SIMT:** a kernel containing `blk` faults cleanly.~~ Dropped: host-CPU only.

## 10. Decisions

- `blk.test` (funct3 2): left reserved (prints an error, writes nothing).
- Latency: 1 cycle, like an ALU op. No cost-model constant yet; add one if 2-cycle SRAM is modelled.
