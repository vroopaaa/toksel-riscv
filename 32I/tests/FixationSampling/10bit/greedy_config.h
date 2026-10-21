// Which score map and K a greedy_*.c file runs on.
//
// run_cycles.py includes one of data/sampler_testdata_*.h and defines FX_MAP and
// NUM_FIXATIONS itself. The defaults below let `emul 32I greedy_x.c -O2` run a file
// on its own (32x32 grid, first image, K = 128).
//
// Scores are 10-bit (0..1023) stored one per uint16_t; the upper 6 bits are zero.

#ifndef FX_MAP
#include "data/sampler_testdata_32x32.h"
#define FX_MAP img_000000039769_F_q10
#endif

#ifndef NUM_FIXATIONS
#define NUM_FIXATIONS 128
#endif

#define FX_CELLS (FX_GRID * FX_GRID)
