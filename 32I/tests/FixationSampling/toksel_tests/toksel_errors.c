// Token-selection misuse: blk.clr with a width other than 32/64, and the reserved
// funct3 2, 6, 7. Each must print an error (run_toksel_tests.py checks stderr) and leave
// the block map, the bucket heads and rd unchanged.
// Returns 0 in a0 on success, 1..5 = which check failed.

#include "../toksel_isa.h"

int main(void)
{
    blk_clr(32);
    blk(33);                    // blocks the 3x3 around (1, 1) on the 32x32 grid
    blk_clr(48);                // invalid width: ignored, still 32x32 with 33 blocked
    if (blk(33) != 0)
        return 1;

    int r = 7;                  // reserved funct3 2: rd must not be written
    __asm__ volatile(".insn r 0x7B, 2, 0, %0, %1, x0" : "+r"(r) : "r"(5u) : "memory");
    if (r != 7)
        return 2;
    if (blk(5) != 1)            // ... and cell 5 must not have been claimed
        return 3;

    bkt_clr();
    bkt_ins(11, 4);             // bucket 4 = {11}
    int r6 = 7, r7 = 7;         // reserved funct3 6 and 7: rd not written, buckets unchanged
    __asm__ volatile(".insn r 0x7B, 6, 0, %0, %1, %2" : "+r"(r6) : "r"(4u), "r"(4u) : "memory");
    __asm__ volatile(".insn r 0x7B, 7, 0, %0, %1, %2" : "+r"(r7) : "r"(4u), "r"(4u) : "memory");
    if (r6 != 7 || r7 != 7)
        return 4;
    if (bkt_head(4) != 11)
        return 5;
    return 0;
}
