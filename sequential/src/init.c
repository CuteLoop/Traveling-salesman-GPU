/*
 * init.c -- Tour random initialization and population seeding
 *
 * Phase 4 implementation.
 *
 * Uses Fisher-Yates (Knuth) shuffle for O(n) uniform permutations.
 * Each individual gets its own RNGState — no shared mutable state.
 */

#include "init.h"

#include <stddef.h>

/* ---- tour_random_init ------------------------------------------------- */

int tour_random_init(Tour *t, int n, RNGState *rng)
{
    if (t == NULL || rng == NULL)  return GA_ERR_INVALID;
    if (t->cities == NULL)        return GA_ERR_INVALID;
    if (n <= 0)                   return GA_ERR_INVALID;

    /* Fill identity permutation */
    for (int i = 0; i < n; i++) {
        t->cities[i] = i;
    }

    /* Fisher-Yates shuffle: iterate from n-1 down to 1 */
    for (int i = n - 1; i > 0; i--) {
        /* j = random index in [0, i] */
        uint64_t r = rng_next_int(rng);
        int j = (int)(r % (uint64_t)(i + 1));

        /* swap cities[i] and cities[j] */
        int tmp       = t->cities[i];
        t->cities[i]  = t->cities[j];
        t->cities[j]  = tmp;
    }

    t->length  = 0.0;
    t->fitness = 0.0;
    return GA_OK;
}

/* ---- population_init -------------------------------------------------- */

int population_init(Tour *pop, int N, int n, RNGState *rng_states)
{
    if (pop == NULL || rng_states == NULL) return GA_ERR_INVALID;
    if (N <= 0 || n <= 0)                 return GA_ERR_INVALID;

    for (int i = 0; i < N; i++) {
        int rc = tour_random_init(&pop[i], n, &rng_states[i]);
        if (rc != GA_OK) return rc;
    }
    return GA_OK;
}
