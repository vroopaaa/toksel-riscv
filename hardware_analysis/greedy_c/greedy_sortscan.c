// Sort-scan greedy fixation sampling on the 64x64 saliency map in fixation_matrix.h.
//
// Same picks as the naive loop: highest-scoring cell that is not blocked (ties go to
// the lower index), then block that cell and its 3x3 neighbourhood.
//
// How: sort all cells by score once (counting sort, scores are 0..255), then walk the
// sorted list from the top and take every cell that is not blocked yet.
//   1. histogram   count[v] = cells with score v
//   2. prefix      start[v] = cells with a score higher than v
//   3. scatter     order[start[score[i]]++] = i, for i = 0, 1, 2, ...
//                  (raster order keeps the lower index first on ties)
//   4. walk        skip blocked cells, pick the others, stop after k picks
//
// FIX_MAP must be unsigned (uint8_t / unsigned char): plain char is signed on x86.
// Returns (in a0) a checksum over the picked indices, like the naive version.

#include "fixation_matrix.h"

#ifndef NUM_FIXATIONS
#define NUM_FIXATIONS 64
#endif

#define NUM_SCORES 256

int picks[NUM_FIXATIONS];
unsigned short order[FIX_CELLS];        // cell indices sorted by score, high to low
int start[NUM_SCORES];                  // histogram, then start position of each score
unsigned char blocked[FIX_CELLS];

// Returns how many fixations were actually picked (<= k).
int greedy_sortscan(int k)
{
    // 1. histogram
    for (int v = 0; v < NUM_SCORES; v++)
        start[v] = 0;
    for (int i = 0; i < FIX_CELLS; i++)
        start[FIX_MAP[i]]++;

    // 2. prefix: highest score goes first
    int pos = 0;
    for (int v = NUM_SCORES - 1; v >= 0; v--) {
        int count = start[v];
        start[v] = pos;
        pos += count;
    }

    // 3. scatter in raster order
    for (int i = 0; i < FIX_CELLS; i++)
        order[start[FIX_MAP[i]]++] = (unsigned short)i;

    // 4. walk the sorted list
    for (int i = 0; i < FIX_CELLS; i++)
        blocked[i] = 0;

    int n = 0;
    for (int j = 0; j < FIX_CELLS && n < k; j++) {
        int best = order[j];
        if (blocked[best])
            continue;

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
    int n = greedy_sortscan(NUM_FIXATIONS);

    unsigned int checksum = n;          // unsigned: wraps the same way, without signed overflow
    for (int i = 0; i < n; i++)
        checksum = checksum * 31u + (unsigned int)picks[i];
    return (int)checksum;
}
