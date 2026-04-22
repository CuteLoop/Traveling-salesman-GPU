/*
 * crossover.h -- Order Crossover (OX1) for TSP permutations
 *
 * Phase 6 public API.
 *
 * Ownership:
 *   Neither function allocates heap memory.  Child must be pre-allocated
 *   (via tour_alloc) for n cities.  Parents are read-only.
 *
 * CUDA-Readiness:
 *   One offspring per thread.  Boolean lookup buffer is a C99 VLA (maps
 *   to per-thread shared memory on GPU).  No global state.
 */

#ifndef GA_TSP_CROSSOVER_H
#define GA_TSP_CROSSOVER_H

#include "ga.h"
#include "tour.h"
#include "rng.h"

/* ---- function declarations -------------------------------------------- */

/*
 * crossover_ox1 -- produce one child via Order Crossover (OX1).
 *
 * Copies a random segment from parent A, then fills remaining positions
 * with cities from parent B in B's order.  Child is always a valid
 * permutation of {0, ..., n-1}.
 *
 * Sets child->length and child->fitness to 0.0 (must be re-evaluated).
 * Returns GA_OK on success, GA_ERR_INVALID on NULL or n <= 0.
 * Zero heap allocations (uses C99 VLA for the used-city buffer).
 */
int crossover_ox1(Tour *child, const Tour *a, const Tour *b,
                  int n, RNGState *rng);

/*
 * apply_crossover -- crossover wrapper honoring probability p_c.
 *
 * Draws rng_next_double(rng).  If the draw >= p_c, copies parent A
 * into child (no crossover).  Otherwise, calls crossover_ox1.
 *
 * Returns GA_OK on success.
 */
int apply_crossover(Tour *child, const Tour *a, const Tour *b,
                    int n, double p_c, RNGState *rng);

#endif /* GA_TSP_CROSSOVER_H */
