/*
 * init.h -- Tour initialization and population seeding
 *
 * Phase 4 public API.
 *
 * Ownership:
 *   tour_random_init and population_init write into caller-owned Tours.
 *   The caller must have called tour_alloc before passing tours in.
 *   No heap allocation inside these functions.
 */

#ifndef GA_TSP_INIT_H
#define GA_TSP_INIT_H

#include "ga.h"
#include "tour.h"
#include "rng.h"

/* ---- function declarations -------------------------------------------- */

/*
 * tour_random_init -- fill a pre-allocated tour with a random permutation
 * of {0, 1, ..., n-1} using strict Fisher-Yates shuffle.
 *
 * Resets length and fitness to 0.0.
 * Returns GA_OK on success, GA_ERR_INVALID on NULL or n <= 0.
 */
int tour_random_init(Tour *t, int n, RNGState *rng);

/*
 * population_init -- initialize N pre-allocated tours, each using its
 * own RNGState from the rng_states array.
 *
 * rng_states must point to an array of at least N seeded states.
 * Maps rng_states[i] -> pop[i].  Direct curandState analog.
 *
 * Returns GA_OK on success, GA_ERR_INVALID on NULL.
 */
int population_init(Tour *pop, int N, int n, RNGState *rng_states);

#endif /* GA_TSP_INIT_H */
