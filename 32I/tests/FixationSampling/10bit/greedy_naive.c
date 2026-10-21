// Naive greedy token selection on a 10-bit score map (see greedy_config.h).
//
// Repeat until K cells are picked or none is free: scan every cell for the highest-scoring
// free one (ties go to the lower index), pick it, block it and its 3x3 neighbourhood.
//
// Returns (in a0) n, the number of picks (<= K); picks[] holds them in order.

#include "greedy_config.h"

int picks[NUM_FIXATIONS];
unsigned char blocked[FX_CELLS];

int greedy_naive(int k)
{
    int n = 0;
    for (int i = 0; i < FX_CELLS; i++)
        blocked[i] = 0;

    while (n < k) {
        int best = -1;
        int best_score = -1;
        for (int i = 0; i < FX_CELLS; i++) {
            if (blocked[i])
                continue;
            if (FX_MAP[i] > best_score) {    // strict '>' keeps the lower index on ties
                best_score = FX_MAP[i];
                best = i;
            }
        }
        if (best < 0)
            break;                           // everything is blocked

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
    return greedy_naive(NUM_FIXATIONS);
}
