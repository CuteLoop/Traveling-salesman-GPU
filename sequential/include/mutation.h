/*
 * mutation.h -- Swap and inversion mutation operators
 *
 * Phase 7 public API.
 *
 * Both core operators mutate in place with zero heap allocation.
 * Separate functions / separate kernel launches — no mixed-operator
 * branching to prevent future CUDA warp divergence.
 *
 * GPU note: each operator becomes its own kernel.
 */

#ifndef GA_TSP_MUTATION_H
#define GA_TSP_MUTATION_H

#include "tour.h"
#include "rng.h"
#include "ga.h"

/* ---- Core mutation operators (100% application rate if called) --------- */

/*
 * mutate_swap -- swap two randomly chosen distinct cities in place.
 *
 * Draws two distinct indices in [0, n-1], swaps them.
 * Sets t->length = 0.0, t->fitness = 0.0 (invalidated).
 * O(1) time, zero heap allocation.
 *
 * Returns GA_OK on success, GA_ERR_INVALID on bad arguments.
 */
int mutate_swap(Tour *t, int n, RNGState *rng);

/*
 * mutate_invert -- reverse a random sub-segment in place.
 *
 * Draws two indices l, r in [0, n-1], ensures l <= r, then reverses
 * t->cities[l..r] using a two-pointer loop.
 * Sets t->length = 0.0, t->fitness = 0.0 (invalidated).
 * O(n) worst case, zero heap allocation.
 *
 * Returns GA_OK on success, GA_ERR_INVALID on bad arguments.
 */
int mutate_invert(Tour *t, int n, RNGState *rng);

/* ---- Wrappers honoring mutation probability p_m ----------------------- */

/*
 * apply_mutation_swap -- draw rng_next_double; if <= p_m, call mutate_swap.
 */
int apply_mutation_swap(Tour *t, int n, double p_m, RNGState *rng);

/*
 * apply_mutation_invert -- draw rng_next_double; if <= p_m, call mutate_invert.
 */
int apply_mutation_invert(Tour *t, int n, double p_m, RNGState *rng);

#endif /* GA_TSP_MUTATION_H */
