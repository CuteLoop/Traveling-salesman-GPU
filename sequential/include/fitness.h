/*
 * fitness.h -- Tour fitness evaluation (single + batch)
 *
 * Phase 3 public API.
 *
 * Ownership:
 *   Neither function allocates heap memory.  The caller owns all arrays.
 *   Both functions write into pre-existing Tour fields (length, fitness).
 */

#ifndef GA_TSP_FITNESS_H
#define GA_TSP_FITNESS_H

#include "ga.h"
#include "tour.h"
#include "instance.h"

/* ---- function declarations -------------------------------------------- */

/*
 * tour_evaluate -- compute closed-tour length and fitness for a single tour.
 *
 * Reads the distance matrix from inst; writes t->length and t->fitness.
 *   length  = sum of edges in the closed tour (including return edge).
 *   fitness = 1.0 / length  when length > 0;  0.0 when length == 0.
 *
 * Returns GA_OK on success, GA_ERR_INVALID if pointers are NULL or n <= 0.
 * No heap allocation.
 */
int tour_evaluate(Tour *t, const TSPInstance *inst);

/*
 * population_evaluate -- batch-evaluate an array of N tours.
 *
 * Calls tour_evaluate for each individual in pop[0..N-1].
 * THIS is the function replaced by a CUDA kernel in Phase GPU-1.
 *
 * Returns GA_OK if all evaluations succeed.
 * No heap allocation.  No statistics gathering.
 */
int population_evaluate(Tour *pop, int N, const TSPInstance *inst);

#endif /* GA_TSP_FITNESS_H */
