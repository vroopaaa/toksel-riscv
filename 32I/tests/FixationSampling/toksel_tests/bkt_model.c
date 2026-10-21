// BKT instruction test: named cases, then a long random sequence of bkt.clr / bkt.ins /
// bkt.head (with BLK ops mixed in) compared step by step against a plain-C model.
// Returns 0 in a0 if everything agrees, otherwise 1..N = the failing named case,
// 1000 + i = random op i disagreed.

#include "../toksel_isa.h"

// ---- plain-C model ----
static unsigned short model_head[1024];
static unsigned char model_valid[1024];

static void model_clr(void)
{
    for (int v = 0; v < 1024; v++)
        model_valid[v] = 0;
}

static int model_ins(unsigned cell, unsigned score)
{
    unsigned v = score % 1024;                       // upper bits ignored
    int old = model_valid[v] ? model_head[v] : -1;
    model_head[v] = (unsigned short)(cell % 4096);
    model_valid[v] = 1;
    return old;
}

static int model_head_of(unsigned score)
{
    unsigned v = score % 1024;
    return model_valid[v] ? model_head[v] : -1;
}

static unsigned rng = 777;
static unsigned next_rand(void)
{
    rng = rng * 1103515245u + 12345u;
    return rng >> 8;
}

int main(void)
{
    // named cases
    bkt_clr();
    if (bkt_head(0) != -1 || bkt_head(1023) != -1)        return 1;  // empty after clr
    if (bkt_ins(5, 7) != -1)                               return 2;  // first push: old head empty
    if (bkt_head(7) != 5)                                  return 3;
    if (bkt_ins(9, 7) != 5 || bkt_head(7) != 9)            return 4;  // second push links to the first
    if (bkt_head(6) != -1 || bkt_head(8) != -1)            return 5;  // neighbours untouched
    if (bkt_ins(4096 + 3, 1024 + 7) != 9)                  return 6;  // upper bits ignored: cell 3, score 7
    if (bkt_head(7) != 3 || bkt_head(2048 + 7) != 3)       return 7;
    if (bkt_ins(4095, 1023) != -1 || bkt_head(1023) != 4095) return 8; // largest index, not read as -1
    if (bkt_ins(0, 0) != -1 || bkt_head(0) != 0)           return 9;  // index 0 is a real head
    blk_clr(64);                                                      // BLK clear leaves buckets alone
    if (bkt_head(7) != 3)                                  return 10;
    if (blk(65) != 1)                                      return 11;
    bkt_clr();                                                        // BKT clear leaves the block map alone
    if (bkt_head(7) != -1 || blk(66) != 0)                 return 12;

    // random ops against the model
    bkt_clr();
    model_clr();
    blk_clr(64);
    for (int i = 0; i < 6000; i++) {
        unsigned r = next_rand();
        unsigned cell = (r & 1) ? next_rand() : next_rand() % 4096;   // half with garbage upper bits
        unsigned score = (r & 2) ? next_rand() : next_rand() % 16;    // half crowded into 16 buckets
        switch ((r >> 2) % 16) {
            case 0:
                bkt_clr();
                model_clr();
                break;
            case 1:                                                    // unrelated BLK traffic
                blk(cell);
                break;
            case 2: case 3: case 4: case 5: case 6: case 7: case 8:
                if (bkt_ins(cell, score) != model_ins(cell, score))
                    return 1000 + i;
                break;
            default:
                if (bkt_head(score) != model_head_of(score))
                    return 1000 + i;
                break;
        }
    }

    // every bucket must still agree
    for (unsigned v = 0; v < 1024; v++)
        if (bkt_head(v) != model_head_of(v))
            return 100000 + v;
    return 0;
}
