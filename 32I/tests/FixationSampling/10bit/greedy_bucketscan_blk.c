// Bucket-scan greedy token selection using the BLK block-map instructions
// (../toksel_isa.h). Same picks as greedy_bucketscan.c.
//
// Same link pass. In the walk, blk(best) replaces the blocked-bitmap test, the 3x3
// stamp loop, its bounds checks and the best / D, best % D: it returns 1 and blocks
// the neighbourhood if the cell was free, 0 (no change) if it was blocked.
// blk_clr(FX_GRID) replaces zeroing the bitmap.
//
// Returns (in a0) n, the number of picks (<= K); picks[] holds them in order.

#include "greedy_config.h"
#include "../toksel_isa.h"

#define NUM_SCORES     1024
#define END_OF_LIST    (-1)

int picks[NUM_FIXATIONS];
short head[NUM_SCORES];                 // first cell of each bucket, or END_OF_LIST
short next_cell[FX_CELLS];              // next cell in the same bucket, or END_OF_LIST

int greedy_bucketscan_blk(int k)
{
    // 1. link pass
    for (int v = 0; v < NUM_SCORES; v++)
        head[v] = END_OF_LIST;
    for (int i = FX_CELLS - 1; i >= 0; i--) {
        int v = FX_MAP[i];
        next_cell[i] = head[v];
        head[v] = (short)i;
    }

    blk_clr(FX_GRID);

    // 2. walk the buckets, highest score first
    int n = 0;
    for (int v = NUM_SCORES - 1; v >= 0 && n < k; v--) {
        for (int best = head[v]; best != END_OF_LIST && n < k; best = next_cell[best]) {
            if (blk(best))
                picks[n++] = best;
        }
    }
    return n;
}

int main(void)
{
    return greedy_bucketscan_blk(NUM_FIXATIONS);
}
