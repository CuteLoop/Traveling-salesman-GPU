/*
 * crossover.c -- Order Crossover (OX1) implementation
 *
 * Phase 6 implementation.
 *
 * OX1 steps:
 *   1. Draw random segment [l, r] from parent A (ensure l <= r).
 *   2. Copy A[l..r] directly into child[l..r].
 *   3. Mark those cities as used (C99 VLA bool used[n]).
 *   4. Starting from position (r+1) % n in parent B, iterate through B.
 *   5. Place unused cities from B into child starting at (r+1) % n,
 *      wrapping around until the child is fully filled.
 *   6. Set child's length and fitness to 0.0.
 *
 * Zero heap allocations.  C99 VLA for the used-city buffer.
 */

#include "crossover.h"

#include <stdbool.h>
#include <string.h>

/* ---- crossover_ox1 ---------------------------------------------------- */

int crossover_ox1(Tour *child, const Tour *a, const Tour *b,
                  int n, RNGState *rng)
{
    if (child == NULL || a == NULL || b == NULL || rng == NULL)
        return GA_ERR_INVALID;
    if (child->cities == NULL || a->cities == NULL || b->cities == NULL)
        return GA_ERR_INVALID;
    if (n <= 0)
        return GA_ERR_INVALID;

    /* Trivial case: n == 1 */
    if (n == 1) {
        child->cities[0] = a->cities[0];
        child->length  = 0.0;
        child->fitness = 0.0;
        return GA_OK;
    }

    /* 1. Draw two random indices and ensure l <= r */
    int l = (int)(rng_next_int(rng) % (uint64_t)n);
    int r = (int)(rng_next_int(rng) % (uint64_t)n);
    if (l > r) { int tmp = l; l = r; r = tmp; }

    /* 2. Copy segment A[l..r] into child[l..r] */
    for (int i = l; i <= r; i++) {
        child->cities[i] = a->cities[i];
    }

    /* 3. Mark used cities — C99 VLA, zero heap */
    bool used[n];
    memset(used, 0, (size_t)n * sizeof(bool));
    for (int i = l; i <= r; i++) {
        used[a->cities[i]] = true;
    }

    /* 4-5. Fill remaining positions with cities from B in B's order */
    int child_pos = (r + 1) % n;
    int b_pos     = (r + 1) % n;

    int filled = r - l + 1;  /* cities already placed */
    while (filled < n) {
        int city = b->cities[b_pos];
        if (!used[city]) {
            child->cities[child_pos] = city;
            used[city] = true;
            child_pos = (child_pos + 1) % n;
            filled++;
        }
        b_pos = (b_pos + 1) % n;
    }

    /* 6. Reset evaluation fields */
    child->length  = 0.0;
    child->fitness = 0.0;

    return GA_OK;
}

/* ---- apply_crossover -------------------------------------------------- */

int apply_crossover(Tour *child, const Tour *a, const Tour *b,
                    int n, double p_c, RNGState *rng)
{
    if (child == NULL || a == NULL || rng == NULL)
        return GA_ERR_INVALID;

    double draw = rng_next_double(rng);
    if (draw >= p_c) {
        /* No crossover — copy parent A */
        tour_copy(child, a, n);
        child->length  = 0.0;
        child->fitness = 0.0;
        return GA_OK;
    }

    return crossover_ox1(child, a, b, n, rng);
}
