// Sort-scan greedy token selection on a 10-bit score map (see greedy_config.h).
//
// Same picks as greedy_naive.c. Sort all cells by score once (counting sort, scores are
// 0..1023), then walk the sorted list from the top and take every cell not blocked yet.
//   1. histogram   count[v] = cells with score v
//   2. prefix      start[v] = cells with a score higher than v
//   3. scatter     order[start[score[i]]++] = i, for i = 0, 1, 2, ...
//                  (raster order keeps the lower index first on ties)
//   4. walk        skip blocked cells, pick the others, stop after k picks
//
// Returns (in a0) n, the number of picks (<= K); picks[] holds them in order.

#include "greedy_config.h"

#define NUM_SCORES 1024

int picks[NUM_FIXATIONS];
unsigned short order[FX_CELLS];         // cell indices sorted by score, high to low
int start[NUM_SCORES];                  // histogram, then start position of each score
unsigned char blocked[FX_CELLS];

int greedy_sortscan(int k)
{
    // 1. histogram
    for (int v = 0; v < NUM_SCORES; v++)
        start[v] = 0;
    for (int i = 0; i < FX_CELLS; i++)
        start[FX_MAP[i]]++;

    // 2. prefix: highest score goes first
    int pos = 0;
    for (int v = NUM_SCORES - 1; v >= 0; v--) {
        int count = start[v];
        start[v] = pos;
        pos += count;
    }

    // 3. scatter in raster order
    for (int i = 0; i < FX_CELLS; i++)
        order[start[FX_MAP[i]]++] = (unsigned short)i;

    // 4. walk the sorted list
    for (int i = 0; i < FX_CELLS; i++)
        blocked[i] = 0;

    int n = 0;
    for (int j = 0; j < FX_CELLS && n < k; j++) {
        int best = order[j];
        if (blocked[best])
            continue;

        picks[n++] = best;

        int r = best / FX_GRID;
        int c = best % FX_GRID;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int rr = r + dr;
                int cc = c + dc;
                if (rr < 0 || rr >= FX_GRID || cc < 0 || cc >= FX_GRID)
                    continue;
                blocked[rr * FX_GRID + cc] = 1;
            }
        }
    }
    return n;
}

int main(void)
{
    return greedy_sortscan(NUM_FIXATIONS);
}
