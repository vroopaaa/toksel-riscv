// Bucket-scan greedy token selection using both parts of the token-selection unit
// (../toksel_isa.h): BKT bucket heads for the link pass, BLK for the walk.
// Same picks as greedy_bucketscan.c.
//
// Link pass: bkt_ins(i, score) returns the bucket's old head and makes i the new head,
// so each cell is one score load, one bkt.ins and one next_cell store. bkt_clr()
// replaces setting all 1024 heads to -1, and head[] leaves memory.
// Walk: bkt_head(v) replaces the head[v] load; blk(best) as in greedy_bucketscan_blk.c.
//
// Returns (in a0) n, the number of picks (<= K); picks[] holds them in order.

#include "greedy_config.h"
#include "../toksel_isa.h"

#define NUM_SCORES     1024
#define END_OF_LIST    (-1)

int picks[NUM_FIXATIONS];
short next_cell[FX_CELLS];              // next cell in the same bucket, or END_OF_LIST

int greedy_bucketscan_bkt(int k)
{
    // 1. link pass
    bkt_clr();
    for (int i = FX_CELLS - 1; i >= 0; i--)
        next_cell[i] = (short)bkt_ins(i, FX_MAP[i]);

    blk_clr(FX_GRID);

    // 2. walk the buckets, highest score first
    int n = 0;
    for (int v = NUM_SCORES - 1; v >= 0 && n < k; v--) {
        for (int best = bkt_head(v); best != END_OF_LIST && n < k; best = next_cell[best]) {
            if (blk(best))
                picks[n++] = best;
        }
    }
    return n;
}

int main(void)
{
    return greedy_bucketscan_bkt(NUM_FIXATIONS);
}
