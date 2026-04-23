/*
 * rng.c -- Xorshift64 pseudo-random number generator
 *
 * Phase 4 implementation.
 *
 * Algorithm: xorshift64 (Marsaglia, 2003).
 * Period: 2^64 - 1.  Not cryptographic — designed for simulation.
 *
 * Seed 0 is remapped to a non-zero constant to avoid the degenerate
 * all-zeros fixed point of xorshift.
 */

#include "rng.h"

/* ---- rng_seed --------------------------------------------------------- */

void rng_seed(RNGState *rng, uint64_t seed)
{
    /* xorshift requires non-zero state; remap 0 to a large prime */
    rng->state = (seed != 0) ? seed : UINT64_C(0x5DEECE66D);
}

/* ---- rng_next_int ----------------------------------------------------- */

uint64_t rng_next_int(RNGState *rng)
{
    uint64_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng->state = x;
    return x;
}

/* ---- rng_next_double -------------------------------------------------- */

double rng_next_double(RNGState *rng)
{
    /* Use upper 53 bits for full double mantissa precision.
     * Division by 2^53 yields [0.0, 1.0). */
    uint64_t v = rng_next_int(rng);
    return (double)(v >> 11) * (1.0 / 9007199254740992.0);  /* 1/2^53 */
}
