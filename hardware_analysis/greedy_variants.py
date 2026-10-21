#!/usr/bin/env python3
"""
Exact greedy fixation samplers on a uint8 score map (D x D, D = 32 at 512 px, 64 at 1024 px).

All variants return the same picks as the naive loop:
  repeat K times: take the highest-scoring unblocked cell (ties -> lower index i = D*r + c),
  then block its 3x3 neighbourhood. Stop early if every cell is blocked.

  naive       full masked argmax over D*D cells per pick
  rowmax      best unblocked cell per row; a pick rescans rows r-1, r, r+1
  tilemax     best unblocked cell per 8x8 tile; a pick rescans the 1, 2 or 4 tiles it touched
  sortscan    counting sort of all cells by score once, then walk the list skipping blocked cells
  threshold   sort only the top band of ~4K cells; move to the next band if it runs out
  bucketscan  one pass links every cell into a per-score list (highest index first, so lists
              come out in raster order); walk scores 255 -> 0 with a bit-packed blocked map
  threshold+  threshold with a bit-packed blocked map, the histogram taken from the
              quantisation step (fused, passed in as `hist`), and a fallback to bucketscan
              when one score band holds more than half the grid (e.g. mostly-zero maps)

Work model (same for every variant): every load and store of an array element counts once
(score, blocked flag or bitmap word, histogram / list / order entry, row or tile summary).
This is what the RISC-V cost scales with; the Python time also includes interpreter overhead.

Run directly to verify all variants against naive and micro-benchmark them.
Does NOT modify any original files in models/.
"""

import os
import time
import argparse

import numpy as np

TILE = 8


# ---------------------------------------------------------------- blocked-map helpers
def _block_bytes(B2: np.ndarray, r: int, c: int) -> int:
    """Byte-per-cell blocked map: store 1 into the (clipped) 3x3 window. Returns stores."""
    win = B2[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
    win[:] = True
    return win.size


class BitRows:
    """Bit-packed blocked map, one integer per grid row (D words; 2 x 32-bit words per row on rv32 at D=64).
    claim() ORs a 3-bit mask into rows r-1, r, r+1; is_blocked() is a shift and an AND."""

    def __init__(self, D: int):
        self.D, self.full, self.rows = D, (1 << D) - 1, [0] * D

    def is_blocked(self, r: int, c: int) -> int:
        return (self.rows[r] >> c) & 1

    def claim(self, r: int, c: int) -> int:
        m = ((7 << c) >> 1) & self.full
        n = 0
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < self.D:
                self.rows[rr] |= m
                n += 1
        return n


def _scan_bytes(order, D, K, blocked, B2, picks):
    """Walk cells in score order on a byte map: load order[j] + blocked[i]; claim stores <= 9 bytes."""
    reads = writes = 0
    for i in order:
        reads += 2
        if blocked[i]:
            continue
        picks.append(int(i))
        writes += _block_bytes(B2, *divmod(int(i), D))
        if len(picks) == K:
            break
    return reads, writes


def _scan_bits(order, D, K, bits: BitRows, picks):
    """Walk cells in score order on a bitmap: load order[j] + one bitmap word; claim = 3 load/OR/store."""
    reads = writes = 0
    for i in order:
        r, c = divmod(int(i), D)
        reads += 2
        if bits.is_blocked(r, c):
            continue
        picks.append(int(i))
        n = bits.claim(r, c)
        reads += n
        writes += n
        if len(picks) == K:
            break
    return reads, writes


def _histogram(flat):
    """hist[v] += 1 per cell: load score, load + store hist[v]."""
    return np.bincount(flat, minlength=256), 2 * flat.size, flat.size


def _counting_sort(flat, cells=None):
    """Stable counting sort, descending score. Cost: histogram + 256-entry prefix + scatter
    (load score, load pos[v], store order, store pos[v]). np.argsort(kind='stable') on uint8
    is a radix sort and produces the identical order."""
    n = flat.size if cells is None else cells.size
    sub = flat if cells is None else flat[cells]
    order = np.argsort(255 - sub, kind="stable")
    order = order if cells is None else cells[order]
    return order, 2 * n + 256 + 2 * n, n + 256 + 2 * n


# ---------------------------------------------------------------- the variants
def greedy_naive(s: np.ndarray, K: int):
    D = s.shape[0]
    sc = s.astype(np.int16).ravel()
    blocked = np.zeros(D * D, bool)
    B2 = blocked.reshape(D, D)
    picks, reads, writes = [], 0, D * D
    for _ in range(K):
        masked = np.where(blocked, -1, sc)
        i = int(masked.argmax())
        reads += 2 * D * D
        if masked[i] < 0:
            break
        picks.append(i)
        writes += _block_bytes(B2, *divmod(i, D))
    return picks, {"reads": reads, "writes": writes}


def greedy_rowmax(s: np.ndarray, K: int):
    D = s.shape[0]
    S2 = s.astype(np.int16)
    B2 = np.zeros((D, D), bool)
    row_val = np.empty(D, np.int16)
    row_col = np.empty(D, np.int64)
    reads, writes = 0, D * D

    def rescan(r0, r1):
        m = np.where(B2[r0:r1], -1, S2[r0:r1])
        row_col[r0:r1] = m.argmax(1)            # first max -> lowest column
        row_val[r0:r1] = m.max(1)
        return 2 * (r1 - r0) * D, 2 * (r1 - r0)

    rd, wr = rescan(0, D)
    reads, writes = reads + rd, writes + wr
    picks = []
    for _ in range(K):
        r = int(row_val.argmax())               # first max -> top row on ties
        reads += D + 1
        if row_val[r] < 0:
            break
        c = int(row_col[r])
        picks.append(r * D + c)
        writes += _block_bytes(B2, r, c)
        rd, wr = rescan(max(r - 1, 0), min(r + 2, D))
        reads, writes = reads + rd, writes + wr
    return picks, {"reads": reads, "writes": writes}


def greedy_tilemax(s: np.ndarray, K: int):
    D = s.shape[0]
    nt = D // TILE
    S2 = s.astype(np.int16)
    B2 = np.zeros((D, D), bool)
    tile_val = np.empty(nt * nt, np.int16)
    tile_idx = np.empty(nt * nt, np.int64)
    reads, writes = 0, D * D

    def rescan(t):
        tr, tc = divmod(t, nt)
        rs, cs = tr * TILE, tc * TILE
        blk = np.where(B2[rs:rs + TILE, cs:cs + TILE], -1, S2[rs:rs + TILE, cs:cs + TILE])
        j = int(blk.argmax())                   # row-major inside a tile = increasing global index
        tile_val[t] = blk.flat[j]
        tile_idx[t] = (rs + j // TILE) * D + cs + j % TILE
        return 2 * TILE * TILE, 2

    for t in range(nt * nt):
        rd, wr = rescan(t)
        reads, writes = reads + rd, writes + wr
    picks = []
    for _ in range(K):
        m = tile_val.max()
        reads += 2 * nt * nt                    # max pass + tie pass over tile summaries
        if m < 0:
            break
        i = int(tile_idx[tile_val == m].min())  # tiles are not in raster order: compare cell indices
        picks.append(i)
        r, c = divmod(i, D)
        writes += _block_bytes(B2, r, c)
        trs = {(rr // TILE) for rr in (r - 1, r, r + 1) if 0 <= rr < D}
        tcs = {(cc // TILE) for cc in (c - 1, c, c + 1) if 0 <= cc < D}
        for tr in trs:
            for tc in tcs:
                rd, wr = rescan(tr * nt + tc)
                reads, writes = reads + rd, writes + wr
    return picks, {"reads": reads, "writes": writes}


def greedy_sortscan(s: np.ndarray, K: int):
    D = s.shape[0]
    flat = s.ravel()
    order, reads, writes = _counting_sort(flat)
    blocked = np.zeros(D * D, bool)
    writes += D * D
    picks = []
    rd, wr = _scan_bytes(order.tolist(), D, K, blocked, blocked.reshape(D, D), picks)
    return picks, {"reads": reads + rd, "writes": writes + wr}


def greedy_threshold(s: np.ndarray, K: int, band_factor: int = 4):
    D = s.shape[0]
    flat = s.ravel()
    hist, reads, writes = _histogram(flat)
    blocked = np.zeros(D * D, bool)
    B2 = blocked.reshape(D, D)
    writes += D * D
    picks, hi = [], 256
    while len(picks) < K and hi > 0:
        lo, cnt = hi, 0
        while lo > 0 and cnt < band_factor * K:
            lo -= 1
            cnt += int(hist[lo])
            reads += 1
        cells = np.flatnonzero((flat >= lo) & (flat < hi))    # one pass to collect the band
        reads += flat.size
        writes += cells.size
        order, rd, wr = _counting_sort(flat, cells)
        reads, writes = reads + rd, writes + wr
        rd, wr = _scan_bytes(order.tolist(), D, K, blocked, B2, picks)
        reads, writes = reads + rd, writes + wr
        hi = lo
    return picks, {"reads": reads, "writes": writes}


def greedy_bucketscan(s: np.ndarray, K: int, levels: int = 256):
    D = s.shape[0]
    flat = s.ravel().tolist()
    N = len(flat)
    head = [-1] * levels                        # one bucket per score value (2^bits)
    nxt = [-1] * N
    for i in range(N - 1, -1, -1):              # one pass: prepend, so each list ends up in raster order
        v = flat[i]
        nxt[i] = head[v]
        head[v] = i
    reads, writes = 2 * N, 2 * N + levels       # load score + head[v]; store nxt[i] + head[v]
    bits = BitRows(D)
    writes += D
    picks = []
    for v in range(levels - 1, -1, -1):
        i = head[v]
        reads += 1
        while i != -1:
            r, c = divmod(i, D)
            reads += 2                          # bitmap word + nxt[i]
            if not bits.is_blocked(r, c):
                picks.append(i)
                n = bits.claim(r, c)
                reads, writes = reads + n, writes + n
                if len(picks) == K:
                    return picks, {"reads": reads, "writes": writes}
            i = nxt[i]
    return picks, {"reads": reads, "writes": writes}


def greedy_threshold_plus(s: np.ndarray, K: int, hist: np.ndarray = None,
                          band_factor: int = 4, fallback_frac: float = 0.5):
    D = s.shape[0]
    flat = s.ravel()
    reads = writes = 0
    if hist is None:                            # not fused: pay for the histogram pass here
        hist, reads, writes = _histogram(flat)
    else:                                       # fused into quantisation: the score is already in a
        reads, writes = flat.size, flat.size    # register, but hist[v] += 1 still loads and stores
    bits = BitRows(D)
    writes += D
    picks, hi = [], 256
    while len(picks) < K and hi > 0:
        lo, cnt = hi, 0
        while lo > 0 and cnt < band_factor * K:
            lo -= 1
            cnt += int(hist[lo])
            reads += 1
        if cnt > fallback_frac * flat.size:     # one huge tie group: a band can't be cut out
            p, w = greedy_bucketscan(s, K)
            return p, {"reads": reads + w["reads"], "writes": writes + w["writes"], "fallback": 1}
        cells = np.flatnonzero((flat >= lo) & (flat < hi))
        reads += flat.size
        writes += cells.size
        order, rd, wr = _counting_sort(flat, cells)
        reads, writes = reads + rd, writes + wr
        rd, wr = _scan_bits(order.tolist(), D, K, bits, picks)
        reads, writes = reads + rd, writes + wr
        hi = lo
    return picks, {"reads": reads, "writes": writes}


ALGOS = {
    "naive": greedy_naive,
    "rowmax": greedy_rowmax,
    "tilemax": greedy_tilemax,
    "sortscan": greedy_sortscan,
    "threshold": greedy_threshold,
    "bucketscan": greedy_bucketscan,
    "threshold+": greedy_threshold_plus,
}
TAKES_HIST = {"threshold+"}                     # can use a histogram fused into quantisation


# ---------------------------------------------------------------- verification / benchmark
def _test_maps(rng):
    td = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampler_testdata")
    maps = []
    if os.path.isdir(td):
        for f in sorted(os.listdir(td)):
            if f.endswith("_F_u8.txt"):
                maps.append((f.replace("_F_u8.txt", ""), np.loadtxt(os.path.join(td, f), dtype=np.uint8)))
    for D in (32, 64):
        maps.append((f"random uint8 {D}x{D}", rng.integers(0, 256, (D, D), dtype=np.uint8)))
        maps.append((f"heavy ties 0..7 {D}x{D}", rng.integers(0, 8, (D, D), dtype=np.uint8)))
        maps.append((f"all zeros {D}x{D}", np.zeros((D, D), np.uint8)))
        sparse = np.zeros((D, D), np.uint8)
        sparse.flat[rng.choice(D * D, 20, replace=False)] = rng.integers(1, 256, 20)
        maps.append((f"mostly zeros {D}x{D}", sparse))
    return maps


def verify(rng) -> bool:
    ok_all = True
    for name, m in _test_maps(rng):
        for K in (64, 128, 256):
            ref, _ = greedy_naive(m, K)
            res = {a: f(m, K)[0] == ref for a, f in ALGOS.items() if a != "naive"}
            res["threshold+ (fused hist)"] = greedy_threshold_plus(m, K, hist=np.bincount(m.ravel(), minlength=256))[0] == ref
            ok_all &= all(res.values())
            if not all(res.values()):
                print(f"  MISMATCH {name} K={K}: {res}")
    print(f">> all variants identical to naive on every test map and K: {ok_all}")
    return ok_all


def microbench(rng, reps: int):
    maps = [m for n, m in _test_maps(rng) if n.startswith("img_")]
    print(f"\n>> Standalone sampler cost on the 64x64 real-image maps (mean over {len(maps)} maps, {reps} reps)")
    print("   threshold+ builds its histogram inside the quantisation pass: the hist[v] += 1 load/store is counted,")
    print("   re-reading the score is not")
    print(f"{'K':>4} {'algorithm':<11} | {'Python time (ms)':>16} | {'loads':>10} | {'stores':>8} | {'loads + stores':>14}")
    print("-" * 76)
    for K in (64, 128, 256):
        for a, f in ALGOS.items():
            ts, rd, wr = [], [], []
            for m in maps:
                kw = {"hist": np.bincount(m.ravel(), minlength=256)} if a in TAKES_HIST else {}
                f(m, K, **kw)
                t0 = time.perf_counter()
                for _ in range(reps):
                    _, w = f(m, K, **kw)
                ts.append((time.perf_counter() - t0) / reps * 1e3)
                rd.append(w["reads"]); wr.append(w["writes"])
            print(f"{K:>4} {a:<11} | {np.mean(ts):16.3f} | {np.mean(rd):10,.0f} | {np.mean(wr):8,.0f} | "
                  f"{np.mean(rd) + np.mean(wr):14,.0f}")
        print("-" * 76)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify and micro-benchmark exact greedy sampler variants")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    verify(rng)
    microbench(rng, args.reps)
