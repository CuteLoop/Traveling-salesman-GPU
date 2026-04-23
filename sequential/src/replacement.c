/*
 * replacement.c -- Generational replacement: build next population
 *
 * Phase 8 implementation.
 *
 * Assembles next_pop from elites (deep-copied into slots [0..e-1])
 * followed by offspring (deep-copied into slots [e..N-1]).
 *
 * GPU: replace memcpy-based assembly with kernel launches.
 */

#include "replacement.h"

#include <stddef.h>

/* ---- build_next_generation -------------------------------------------- */

ga_status_t build_next_generation(Tour *next_pop,
                                  const Tour *elites, int e,
                                  const Tour *offspring, int offspring_count,
                                  int N, int n)
{
    /* --- argument validation ------------------------------------------ */
    if (next_pop == NULL)                 return GA_ERR_INVALID;
    if (N <= 0)                           return GA_ERR_INVALID;
    if (n <= 0)                           return GA_ERR_INVALID;
    if (e < 0 || e > N)                  return GA_ERR_INVALID;
    if (offspring_count != N - e)         return GA_ERR_INVALID;
    if (e > 0 && elites == NULL)          return GA_ERR_INVALID;
    if (offspring_count > 0 && offspring == NULL)
                                          return GA_ERR_INVALID;

    /* --- deep-copy elites into slots [0..e-1] ------------------------- */
    for (int i = 0; i < e; i++) {
        tour_copy(&next_pop[i], &elites[i], n);
    }

    /* --- deep-copy offspring into slots [e..N-1] ----------------------- */
    for (int i = 0; i < offspring_count; i++) {
        tour_copy(&next_pop[e + i], &offspring[i], n);
    }

    return GA_OK;
}
