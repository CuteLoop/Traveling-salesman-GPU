/*
 * replacement.h -- Generational replacement: build next population
 *
 * Phase 8 public API.
 *
 * Combines elites and offspring via deep copy into a new population.
 * No pointer aliasing between source and destination arrays.
 *
 * GPU note: replace memcpy-based assembly with kernel launches.
 */

#ifndef TSP_REPLACEMENT_H
#define TSP_REPLACEMENT_H

#include "ga.h"
#include "tour.h"

/*
 * build_next_generation -- assemble next_pop from elites + offspring.
 *
 * Layout: next_pop[0..e-1]       = deep copies of elites[0..e-1]
 *         next_pop[e..N-1]       = deep copies of offspring[0..offspring_count-1]
 *
 * Parameters:
 *   next_pop         Pre-allocated array of N Tour structs (tour_alloc'd).
 *   elites           Array of e elite tours (may be NULL when e == 0).
 *   e                Number of elites.
 *   offspring        Array of offspring tours.
 *   offspring_count  Must equal N - e.
 *   N                Population size.
 *   n                Number of cities per tour.
 *
 * Returns GA_OK on success, GA_ERR_INVALID on bad arguments.
 */
ga_status_t build_next_generation(Tour *next_pop,
                                  const Tour *elites, int e,
                                  const Tour *offspring, int offspring_count,
                                  int N, int n);

#endif /* TSP_REPLACEMENT_H */
