/*
 * selection.h -- Tournament selection for parent pairing
 *
 * Phase 5 public API.
 *
 * Ownership:
 *   Neither function allocates heap memory.  The population is read-only.
 *   Selected indices are written to caller-owned pointers.
 *
 * CUDA-Readiness:
 *   Tournament selection requires no global information — each contest is
 *   fully independent.  Maps to one GPU thread per selection with zero
 *   synchronization overhead.
 */

#ifndef GA_TSP_SELECTION_H
#define GA_TSP_SELECTION_H

#include "ga.h"
#include "tour.h"
#include "rng.h"

/* ---- function declarations -------------------------------------------- */

/*
 * tournament_select -- select one individual via k-tournament.
 *
 * Samples k candidates WITHOUT replacement from pop[0..N-1].
 * Returns the index of the candidate with the highest fitness.
 * Ties are broken deterministically: the first drawn candidate wins.
 *
 * Output: *out_index receives the winning index in [0, N).
 * Returns GA_OK on success, GA_ERR_INVALID on NULL / invalid args.
 * Zero heap allocations (uses C99 VLA for the drawn-index buffer).
 */
int tournament_select(const Tour *pop, int N, int k,
                      RNGState *rng, int *out_index);

/*
 * select_parent_pair -- select two distinct parent indices.
 *
 * Calls tournament_select twice; redraws the second if it equals the first.
 * Requires N >= 2.
 *
 * Output: *out_a and *out_b receive two distinct indices in [0, N).
 * Returns GA_OK on success, GA_ERR_INVALID on NULL / N < 2.
 */
int select_parent_pair(const Tour *pop, int N, int k,
                       RNGState *rng, int *out_a, int *out_b);

#endif /* GA_TSP_SELECTION_H */
