/*
 * selection.c -- Tournament selection (without replacement)
 *
 * Phase 5 implementation.
 *
 * Sampling strategy: k candidates drawn WITHOUT replacement within a
 * single tournament using a C99 VLA tracking buffer.  This prevents
 * degenerate tournaments where the same individual is sampled twice.
 *
 * Tie-breaking: deterministic — the first drawn candidate wins when
 * two or more candidates have equal fitness.
 *
 * Zero heap allocations.
 */

#include "selection.h"

#include <stddef.h>
#include <stdbool.h>

/* ---- tournament_select ------------------------------------------------ */

int tournament_select(const Tour *pop, int N, int k,
                      RNGState *rng, int *out_index)
{
    if (pop == NULL || rng == NULL || out_index == NULL)
        return GA_ERR_INVALID;
    if (N <= 0 || k <= 0)
        return GA_ERR_INVALID;

    /* Clamp k to N to avoid impossible sampling */
    int eff_k = (k > N) ? N : k;

    /* C99 VLA: track drawn indices to enforce without-replacement */
    int drawn[eff_k];

    int best_idx = -1;
    double best_fit = -1.0;

    for (int d = 0; d < eff_k; d++) {
        int candidate;

        /* Draw a candidate not yet in drawn[0..d-1] */
        for (;;) {
            candidate = (int)(rng_next_int(rng) % (uint64_t)N);

            /* Check for duplicates */
            bool dup = false;
            for (int j = 0; j < d; j++) {
                if (drawn[j] == candidate) { dup = true; break; }
            }
            if (!dup) break;
        }

        drawn[d] = candidate;

        /* Keep the best — ties broken by first encountered (strict >) */
        if (best_idx < 0 || pop[candidate].fitness > best_fit) {
            best_fit = pop[candidate].fitness;
            best_idx = candidate;
        }
    }

    *out_index = best_idx;
    return GA_OK;
}

/* ---- select_parent_pair ----------------------------------------------- */

int select_parent_pair(const Tour *pop, int N, int k,
                       RNGState *rng, int *out_a, int *out_b)
{
    if (pop == NULL || rng == NULL || out_a == NULL || out_b == NULL)
        return GA_ERR_INVALID;
    if (N < 2)
        return GA_ERR_INVALID;

    int rc = tournament_select(pop, N, k, rng, out_a);
    if (rc != GA_OK) return rc;

    /* Redraw second parent until it differs from the first */
    for (;;) {
        rc = tournament_select(pop, N, k, rng, out_b);
        if (rc != GA_OK) return rc;
        if (*out_b != *out_a) break;
    }

    return GA_OK;
}
