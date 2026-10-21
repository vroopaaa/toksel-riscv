// Bucket-scan greedy token selection on a 10-bit score map (see greedy_config.h).
//
// Same picks as greedy_naive.c. One linked list ("bucket") per score value, built in a
// single pass, then walk the buckets from score 1023 down to 0 and take every cell that
// is not blocked yet.
//   1. link pass   for i = last cell down to 0: put i at the front of bucket score[i]
//                  (going backwards + inserting at the front leaves every bucket in
//                  increasing index order, so the lower index comes first on ties)
//   2. walk        score 1023 -> 0, follow each bucket, skip blocked cells, pick the
//                  others, stop after k picks
// The blocked map is stored as bits: one 32-bit word holds 32 cells of a row.
//
// Returns (in a0) n, the number of picks (<= K); picks[] holds them in order.

#include "greedy_config.h"

#define NUM_SCORES     1024
#define END_OF_LIST    (-1)
#define WORDS_PER_ROW  ((FX_GRID + 31) / 32)

int picks[NUM_FIXATIONS];
short head[NUM_SCORES];                 // first cell of each bucket, or END_OF_LIST
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

int greedy_bucketscan(int k)
{
    // 1. link pass
    for (int v = 0; v < NUM_SCORES; v++)
        head[v] = END_OF_LIST;
    for (int i = FX_CELLS - 1; i >= 0; i--) {
        int v = FX_MAP[i];
        next_cell[i] = head[v];
        head[v] = (short)i;
    }

    for (int r = 0; r < FX_GRID; r++)
        for (int w = 0; w < WORDS_PER_ROW; w++)
            blocked_bits[r][w] = 0;

    // 2. walk the buckets, highest score first
    int n = 0;
    for (int v = NUM_SCORES - 1; v >= 0 && n < k; v--) {
        for (int best = head[v]; best != END_OF_LIST && n < k; best = next_cell[best]) {
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
    return greedy_bucketscan(NUM_FIXATIONS);
}
