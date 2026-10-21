// C API for the custom-3 token-selection unit (opcode 0x7B, see 32I/include/CPU.h):
//   BLK  block map (doc/blk_design.md): one bit per cell of a D x D grid, 1 = still pickable.
//   BKT  bucket heads (doc/bkt_design.md): one linked-list head per 10-bit score value.
// Both are hidden CPU state, reachable only through these instructions.
//
// volatile + the "memory" clobber keep the compiler from reordering, merging or deleting
// calls: the effect lives in state the compiler can't see.

#ifndef TOKSEL_ISA_H
#define TOKSEL_ISA_H

// Test-and-claim cell idx = D*r + c. Returns 1 if it was pickable, and then its 3x3
// neighbourhood (clipped at the grid edges) is made unpickable. Returns 0 otherwise,
// and nothing changes.
static inline int blk(unsigned idx)
{
    int r;
    __asm__ volatile(".insn r 0x7B, 0, 0, %0, %1, x0" : "=r"(r) : "r"(idx) : "memory");
    return r;
}

// Grid width D (32 or 64); every cell pickable again.
static inline void blk_clr(unsigned dim)
{
    __asm__ volatile(".insn r 0x7B, 1, 0, x0, %0, x0" : : "r"(dim) : "memory");
}

// Every bucket empty.
static inline void bkt_clr(void)
{
    __asm__ volatile(".insn r 0x7B, 3, 0, x0, x0, x0" : : : "memory");
}

// Push cell (0..4095) on the bucket for score (0..1023). Returns the bucket's old head,
// -1 if it was empty; store it as next[cell] to keep the list.
static inline int bkt_ins(unsigned cell, unsigned score)
{
    int r;
    __asm__ volatile(".insn r 0x7B, 4, 0, %0, %1, %2" : "=r"(r) : "r"(cell), "r"(score) : "memory");
    return r;
}

// First cell of the bucket for score (0..1023), -1 if empty.
static inline int bkt_head(unsigned score)
{
    int r;
    __asm__ volatile(".insn r 0x7B, 5, 0, %0, %1, x0" : "=r"(r) : "r"(score) : "memory");
    return r;
}

#endif
