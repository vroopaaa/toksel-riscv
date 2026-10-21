// BLK instruction test: named corner cases, then a long random sequence compared step by
// step against a plain-C model of the block map. Returns 0 in a0 if everything agrees,
// otherwise 1..N = the failing corner case, 1000 + i = random op i disagreed,
// 100000 + idx = the final raster sweep disagreed at cell idx.

#include "../toksel_isa.h"

// ---- plain-C model: 1 = pickable, stored 64 columns per row whatever D is ----
static unsigned char model[64 * 64];
static unsigned model_dim;

static void model_clr(unsigned dim)
{
    model_dim = dim;
    for (unsigned r = 0; r < 64; r++)
        for (unsigned c = 0; c < 64; c++)
            model[r * 64 + c] = (r < dim && c < dim);
}

static int model_blk(unsigned idx)
{
    unsigned c = idx % model_dim;                    // upper bits ignored: idx mod D*D
    unsigned r = (idx / model_dim) % model_dim;
    if (!model[r * 64 + c])
        return 0;
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            int rr = (int)r + dr, cc = (int)c + dc;
            if (rr >= 0 && rr < (int)model_dim && cc >= 0 && cc < (int)model_dim)
                model[rr * 64 + cc] = 0;
        }
    return 1;
}

// ---- corner cases: after blk_clr(dim), blk(claim) must return 1, then blk(probe) ----
struct corner { unsigned dim, claim, probe; int expect; };

static const struct corner corners[] = {
    // 64x64: top-left corner, clipped
    {64,    0,    1, 0}, {64,    0,   65, 0}, {64,    0,    2, 1}, {64,    0,  128, 1},
    // top-right: no wrap into the next row
    {64,   63,   62, 0}, {64,   63,  126, 0}, {64,   63,   64, 1}, {64,   63,    0, 1},
    // bottom-left / bottom-right: no wrap into the previous row
    {64, 4032, 3969, 0}, {64, 4032, 4095, 1}, {64, 4032, 3967, 1},
    {64, 4095, 4030, 0}, {64, 4095, 4094, 0}, {64, 4095, 4032, 1},
    // interior: all 8 neighbours blocked, the ring outside them free
    {64,  65,    0, 0}, {64,  65,    2, 0}, {64,  65,  130, 0}, {64,  65,    3, 1}, {64,  65,  192, 1},
    // the claimed cell itself: a second claim returns 0
    {64, 100,  100, 0},
    // 32x32: same corners at D = 32
    {32,    0,   33, 0}, {32,   31,   62, 0}, {32,   31,   32, 1}, {32,   32,   31, 1},
    {32,  992,  961, 0}, {32,  992, 1023, 1}, {32, 1023,  990, 0}, {32, 1023,  992, 1},
    // upper bits ignored (index taken mod D*D)
    {64, 4096 + 5, 5, 0}, {64, 0xFFFFF000u + 70, 70, 0}, {32, 1024 + 40, 40, 0},
};

static unsigned rng = 12345;
static unsigned next_rand(void)
{
    rng = rng * 1103515245u + 12345u;
    return rng >> 8;
}

int main(void)
{
    int n_corners = sizeof(corners) / sizeof(corners[0]);
    for (int i = 0; i < n_corners; i++) {
        blk_clr(corners[i].dim);
        if (blk(corners[i].claim) != 1 || blk(corners[i].probe) != corners[i].expect)
            return i + 1;
    }

    // A blocked candidate must not extend the blocked area: claim 65, try 66 (blocked),
    // then 67 must still be free (66's 3x3 would have covered it).
    blk_clr(64);
    if (blk(65) != 1 || blk(66) != 0 || blk(67) != 1)
        return n_corners + 1;

    // Random ops against the model; a clear every 300 ops, alternating D.
    unsigned dim = 64;
    blk_clr(dim);
    model_clr(dim);
    for (int i = 0; i < 6000; i++) {
        if (i % 300 == 299) {
            dim = (dim == 64) ? 32 : 64;
            blk_clr(dim);
            model_clr(dim);
            continue;
        }
        unsigned r = next_rand();
        unsigned idx = (r & 1) ? (r >> 1) : ((r >> 1) % (dim * dim));   // half with garbage upper bits
        if (blk(idx) != model_blk(idx))
            return 1000 + i;
    }

    // Final raster sweep over every cell: hardware and model must still agree.
    for (unsigned idx = 0; idx < dim * dim; idx++)
        if (blk(idx) != model_blk(idx))
            return 100000 + idx;
    return 0;
}
