// Token-selection instructions on a mispredicted path must not touch the block map or
// the bucket heads (superscalar squash).
//
// Each asm block is an always-taken branch, predicted not-taken by the empty BTB,
// followed by wrong-path instructions. `li t0` -> `bnez t0` is a RAW hazard, so the
// branch always starts a new window and the wrong-path instructions sit right behind it
// (blk and bkt.ins are different classes, so both fit in the window at width >= 3),
// where they must be squashed.
// Returns 0 in a0 if nothing changed, 1 = a wrong-path blk stamped, 2 = a wrong-path
// bkt.ins or bkt.clr changed bucket 3.

#include "../toksel_isa.h"

int main(void)
{
    unsigned idx = 100, score = 3;
    blk_clr(64);
    bkt_clr();
    bkt_ins(42, score);                     // bucket 3 = {42}
    __asm__ volatile(
        "li   t0, 1\n"
        "bnez t0, 1f\n"                     // always taken
        ".insn r 0x7B, 0, 0, t1, %0, x0\n"  // wrong path: blk 100
        ".insn r 0x7B, 4, 0, t2, %0, %1\n"  // wrong path: bkt.ins 100 on bucket 3
        "1:\n"
        : : "r"(idx), "r"(score) : "t0", "t1", "t2", "memory");
    __asm__ volatile(
        "li   t0, 1\n"
        "bnez t0, 1f\n"                     // always taken
        ".insn r 0x7B, 3, 0, x0, x0, x0\n"  // wrong path: bkt.clr
        "1:\n"
        : : : "t0", "memory");
    if (blk(idx) != 1)
        return 1;
    if (bkt_head(score) != 42)
        return 2;
    return 0;
}
