/*
 * elitism.h -- Elite extraction from a population
 *
 * Phase 8 public API.
 *
 * Uses a local EliteRef struct + qsort to find the top-e individuals
 * without any global variables (C99-safe).
 *
 * GPU note: replace qsort-based extraction with a bitonic Top-K kernel.
 */

#ifndef TSP_ELITISM_H
#define TSP_ELITISM_H

#include "ga.h"
#include "tour.h"

/*
 * extract_elites -- copy the top-e individuals (by fitness, descending)
 *                   into the pre-allocated elites[] array.
 *
 * Parameters:
 *   elites  Pre-allocated array of e Tour structs (each with tour_alloc'd cities).
 *   pop     Source population (read-only, not modified).
 *   N       Population size.
 *   e       Number of elites to extract (0 <= e <= N).
 *   n       Number of cities per tour.
 *
 * Tie-breaking: when two individuals share the same fitness, the one
 * with the lower original index in pop[] comes first.
 *
 * Returns GA_OK on success, GA_ERR_INVALID on bad arguments,
 *         GA_ERR_ALLOC if internal temporary allocation fails.
 */
ga_status_t extract_elites(Tour *elites, const Tour *pop, int N, int e, int n);

#endif /* TSP_ELITISM_H */
