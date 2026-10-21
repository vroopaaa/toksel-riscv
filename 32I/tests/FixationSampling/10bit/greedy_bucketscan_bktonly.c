// Bucket-scan greedy token selection using only the BKT bucket heads (../toksel_isa.h),
// without BLK: the link pass is greedy_bucketscan_bkt.c's, the walk is
// greedy_bucketscan.c's software blocked bitmap. Same picks as greedy_bucketscan.c.
// Exists to measure BKT's saving on its own (greedy_bucketscan_blk.c measures BLK's).
//
// Returns (in a0) n, the number of picks (<= K); picks[] holds them in order.

#include "greedy_config.h"
#include "../toksel_isa.h"

#define NUM_SCORES     1024
#define END_OF_LIST    (-1)
#define WORDS_PER_ROW  ((FX_GRID + 31) / 32)

int picks[NUM_FIXATIONS];
short next_cell[FX_CELLS];              // next cell in the same bucket, or END_OF_LIST
unsigned int blocked_bits[FX_GRID][WORDS_PER_ROW];

static int is_blocked(int r, int c)
{
    return (blocked_bits[r][c >> 5] >> (c & 31)) & 1u;
}

static void block_cell(int r, int c)
{
    blocked_bits[r][c >> 5] |= 1u << (c & 31);
}

int greedy_bucketscan_bktonly(int k)
{
    // 1. link pass
    bkt_clr();
    for (int i = FX_CELLS - 1; i >= 0; i--)
        next_cell[i] = (short)bkt_ins(i, FX_MAP[i]);

    for (int r = 0; r < FX_GRID; r++)
        for (int w = 0; w < WORDS_PER_ROW; w++)
            blocked_bits[r][w] = 0;

    // 2. walk the buckets, highest score first
    int n = 0;
    for (int v = NUM_SCORES - 1; v >= 0 && n < k; v--) {
        for (int best = bkt_head(v); best != END_OF_LIST && n < k; best = next_cell[best]) {
            int r = best / FX_GRID;
            int c = best % FX_GRID;
            if (is_blocked(r, c))
                continue;

            picks[n++] = best;

            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int rr = r + dr;
                    int cc = c + dc;
                    if (rr < 0 || rr >= FX_GRID || cc < 0 || cc >= FX_GRID)
                        continue;
                    block_cell(rr, cc);
                }
            }
        }
    }
    return n;
}

int main(void)
{
    return greedy_bucketscan_bktonly(NUM_FIXATIONS);
}
