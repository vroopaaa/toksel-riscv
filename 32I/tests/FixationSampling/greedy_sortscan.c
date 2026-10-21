// Sort-once-and-scan greedy fixation sampling on the 64x64 saliency map in fixation_matrix.h.
//
// Same picks as the naive loop: highest-scoring cell that is not blocked (ties go to
// the lower index), then block that cell and its 3x3 neighbourhood.
//
// How: counting-sort all cells once into one array (score descending, index ascending),
// then scan that array front to back and take every cell that is not blocked yet.
//   1. histogram   count how many cells have each score
//   2. offsets     turn the counts into start positions, highest score first
//   3. scatter     for i = 0 up to the last cell: order[offset[score[i]]++] = i
//                  (going forwards keeps equal scores in raster order, so the lower
//                  index comes first on ties)
//   4. scan        walk order[], skip blocked cells, pick the others, stop after k picks
// The blocked map is stored as bits: one 32-bit word holds 32 cells of a row.
//
// FIX_MAP must be unsigned (uint8_t / unsigned char): plain char is signed on x86.
// Returns (in a0) a checksum over the picked indices, like the naive version.

#include "fixation_matrix.h"

#ifndef NUM_FIXATIONS
#define NUM_FIXATIONS 64
#endif

#define NUM_SCORES     256
#define WORDS_PER_ROW  ((FIX_DIM + 31) / 32)

int picks[NUM_FIXATIONS];
int offset[NUM_SCORES];                 // count, then start position, per score
short order[FIX_CELLS];                 // cell indices, best first
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
int greedy_sortscan(int k)
{
    // 1. histogram
    for (int v = 0; v < NUM_SCORES; v++)
        offset[v] = 0;
    for (int i = 0; i < FIX_CELLS; i++)
        offset[FIX_MAP[i]]++;

    // 2. offsets: highest score gets position 0
    int acc = 0;
    for (int v = NUM_SCORES - 1; v >= 0; v--) {
        int count = offset[v];
        offset[v] = acc;
        acc += count;
    }

    // 3. stable scatter
    for (int i = 0; i < FIX_CELLS; i++)
        order[offset[FIX_MAP[i]]++] = (short)i;

    for (int r = 0; r < FIX_DIM; r++)
        for (int w = 0; w < WORDS_PER_ROW; w++)
            blocked_bits[r][w] = 0;

    // 4. scan
    int n = 0;
    for (int j = 0; j < FIX_CELLS && n < k; j++) {
        int best = order[j];
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
