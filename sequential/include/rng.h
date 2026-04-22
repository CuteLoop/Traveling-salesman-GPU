/*
 * rng.h -- Deterministic pseudo-random number generator (Xorshift64)
 *
 * Phase 4 public API.
 *
 * One RNGState per individual — no shared global generator.
 * Direct analog of curandState for the GPU port.
 *
 * Ownership:
 *   RNGState is a value type (uint64_t state).  No heap allocation.
 *   Copyable by value.
 */

#ifndef GA_TSP_RNG_H
#define GA_TSP_RNG_H

#include <stdint.h>

/* ---- RNGState --------------------------------------------------------- */

typedef struct {
    uint64_t state;
} RNGState;

/* ---- function declarations -------------------------------------------- */

/*
 * rng_seed -- initialize the RNG state with the given seed.
 *
 * Seed 0 is remapped internally to avoid the degenerate zero state
 * of xorshift.
 */
void rng_seed(RNGState *rng, uint64_t seed);

/*
 * rng_next_int -- return the next 64-bit pseudo-random integer.
 *
 * Advances the state by one step.
 */
uint64_t rng_next_int(RNGState *rng);

/*
 * rng_next_double -- return a uniform double in [0.0, 1.0).
 *
 * Uses the upper 53 bits of rng_next_int() for full double precision.
 */
double rng_next_double(RNGState *rng);

#endif /* GA_TSP_RNG_H */
