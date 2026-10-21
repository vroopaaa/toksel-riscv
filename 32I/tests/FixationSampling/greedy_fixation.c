// Greedy fixation sampling on the 64x64 saliency map in fixation_matrix.h.
//
// Repeat K times: pick the highest-scoring cell that is not blocked (ties go to
// the lower index), then block that cell and its 3x3 neighbourhood.
// Matches the golden picks in greedy_64.txt / greedy_128.txt / greedy_256.txt.
//
// Returns (in a0) a checksum over the picked indices so tests/python/run_tests.py
// can compare native vs rv32im. picks[] holds the picked cell indices in order.

#include "fixation_matrix.h"

#ifndef NUM_FIXATIONS
#define NUM_FIXATIONS 64
#endif

int picks[NUM_FIXATIONS];
unsigned char blocked[FIX_CELLS];

// Returns how many fixations were actually picked (<= k).
int greedy_fixation(int k)
{
    int n = 0;
    for (int i = 0; i < FIX_CELLS; i++)
        blocked[i] = 0;

    while (n < k) {
        int best = -1;
        int best_score = -1;
        for (int i = 0; i < FIX_CELLS; i++) {
            if (blocked[i])
                continue;
            if (FIX_MAP[i] > best_score) {   // strict '>' keeps the lower index on ties
                best_score = FIX_MAP[i];
                best = i;
            }
        }
        if (best < 0)
            break;                           // everything is blocked

        picks[n++] = best;

        int r = best / FIX_DIM;
        int c = best % FIX_DIM;
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int rr = r + dr;
                int cc = c + dc;
                if (rr < 0 || rr >= FIX_DIM || cc < 0 || cc >= FIX_DIM)
                    continue;
                blocked[rr * FIX_DIM + cc] = 1;
            }
        }
    }
    return n;
}

int main(void)
{
    int n = greedy_fixation(NUM_FIXATIONS);

    int checksum = n;
    for (int i = 0; i < n; i++)
        checksum = checksum * 31 + picks[i];
    return checksum;
}
