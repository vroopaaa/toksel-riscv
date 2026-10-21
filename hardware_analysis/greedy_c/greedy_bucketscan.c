// Bucket-scan greedy fixation sampling on the 64x64 saliency map in fixation_matrix.h.
//
// Same picks as the naive loop: highest-scoring cell that is not blocked (ties go to
// the lower index), then block that cell and its 3x3 neighbourhood.
//
// How: one linked list ("bucket") per score value, built in a single pass, then walk the
// buckets from score 255 down to 0 and take every cell that is not blocked yet.
//   1. link pass   for i = last cell down to 0: put i at the front of bucket score[i]
//                  (going backwards + inserting at the front leaves every bucket in
//                  increasing index order, so the lower index comes first on ties)
//   2. walk        score 255 -> 0, follow each bucket, skip blocked cells, pick the
//                  others, stop after k picks
// The blocked map is stored as bits: one 32-bit word holds 32 cells of a row.
//
// FIX_MAP must be unsigned (uint8_t / unsigned char): plain char is signed on x86.
// Returns (in a0) a checksum over the picked indices, like the naive version.

#include "fixation_matrix.h"

#ifndef NUM_FIXATIONS
#define NUM_FIXATIONS 64
#endif

#define NUM_SCORES     256
#define END_OF_LIST    (-1)
#define WORDS_PER_ROW  ((FIX_DIM + 31) / 32)

int picks[NUM_FIXATIONS];
short head[NUM_SCORES];                 // first cell of each bucket, or END_OF_LIST
short next_cell[FIX_CELLS];             // next cell in the same bucket, or END_OF_LIST
unsigned int blocked_bits[FIX_DIM][WORDS_PER_ROW];

static int is_blocked(int r, int c)
{
    return (blocked_bits[r][c >> 5] >> (c & 31)) & 1u;
}

static void block_cell(int r, int c)
{
    blocked_bits[r][c >> 5] |= 1u << (c & 31);
}

// Returns how many fixations were actually picked (<= k).
int greedy_bucketscan(int k)
{
    // 1. link pass
    for (int v = 0; v < NUM_SCORES; v++)
        head[v] = END_OF_LIST;
    for (int i = FIX_CELLS - 1; i >= 0; i--) {
        int v = FIX_MAP[i];
        next_cell[i] = head[v];
        head[v] = (short)i;
    }

    for (int r = 0; r < FIX_DIM; r++)
        for (int w = 0; w < WORDS_PER_ROW; w++)
            blocked_bits[r][w] = 0;

    // 2. walk the buckets, highest score first
    int n = 0;
    for (int v = NUM_SCORES - 1; v >= 0 && n < k; v--) {
        for (int best = head[v]; best != END_OF_LIST && n < k; best = next_cell[best]) {
            int r = best / FIX_DIM;
            int c = best % FIX_DIM;
            if (is_blocked(r, c))
                continue;

            picks[n++] = best;

            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int rr = r + dr;
                    int cc = c + dc;
                    if (rr < 0 || rr >= FIX_DIM || cc < 0 || cc >= FIX_DIM)
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
    int n = greedy_bucketscan(NUM_FIXATIONS);

    unsigned int checksum = n;          // unsigned: wraps the same way, without signed overflow
    for (int i = 0; i < n; i++)
        checksum = checksum * 31u + (unsigned int)picks[i];
    return (int)checksum;
}
