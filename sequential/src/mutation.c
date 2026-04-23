/*
 * mutation.c -- Swap and inversion mutation operators
 *
 * Phase 7 implementation.
 *
 * Zero heap allocations.  All operations are in-place on t->cities.
 * length/fitness are invalidated (set to 0.0) after every mutation.
 *
 * GPU: each operator becomes its own kernel — no mixed branching.
 */

#include "mutation.h"

/* ---- mutate_swap ------------------------------------------------------ */

int mutate_swap(Tour *t, int n, RNGState *rng)
{
    if (t == NULL || t->cities == NULL || rng == NULL)
        return GA_ERR_INVALID;
    if (n < 2)
        return GA_ERR_INVALID;

    /* Draw two distinct indices */
    int i = (int)(rng_next_int(rng) % (uint64_t)n);
    int j = (int)(rng_next_int(rng) % (uint64_t)(n - 1));
    if (j >= i) j++;   /* maps [0, n-2] → [0, n-1] \ {i} */

    /* Swap */
    int tmp = t->cities[i];
    t->cities[i] = t->cities[j];
    t->cities[j] = tmp;

    /* Invalidate cached evaluation */
    t->length  = 0.0;
    t->fitness = 0.0;

    return GA_OK;
}

/* ---- mutate_invert ---------------------------------------------------- */

int mutate_invert(Tour *t, int n, RNGState *rng)
{
    if (t == NULL || t->cities == NULL || rng == NULL)
        return GA_ERR_INVALID;
    if (n < 2)
        return GA_ERR_INVALID;

    /* Draw two random indices */
    int l = (int)(rng_next_int(rng) % (uint64_t)n);
    int r = (int)(rng_next_int(rng) % (uint64_t)n);

    /* Ensure l <= r */
    if (l > r) {
        int tmp = l;
        l = r;
        r = tmp;
    }

    /* Reverse the sub-segment [l..r] in place — two-pointer loop */
    while (l < r) {
        int tmp = t->cities[l];
        t->cities[l] = t->cities[r];
        t->cities[r] = tmp;
        l++;
        r--;
    }

    /* Invalidate cached evaluation */
    t->length  = 0.0;
    t->fitness = 0.0;

    return GA_OK;
}

/* ---- apply_mutation_swap ---------------------------------------------- */

int apply_mutation_swap(Tour *t, int n, double p_m, RNGState *rng)
{
    if (t == NULL || rng == NULL)
        return GA_ERR_INVALID;

    double r = rng_next_double(rng);
    if (r <= p_m) {
        return mutate_swap(t, n, rng);
    }
    return GA_OK;
}

/* ---- apply_mutation_invert -------------------------------------------- */

int apply_mutation_invert(Tour *t, int n, double p_m, RNGState *rng)
{
    if (t == NULL || rng == NULL)
        return GA_ERR_INVALID;

    double r = rng_next_double(rng);
    if (r <= p_m) {
        return mutate_invert(t, n, rng);
    }
    return GA_OK;
}
