/*
 * tour.c -- Tour allocation, cleanup, copy, validation
 *
 * Phase 2 implementation.
 */

#include "tour.h"

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ---- tour_alloc -------------------------------------------------------- */

ga_status_t tour_alloc(Tour *t, int n)
{
    t->cities  = malloc((size_t)n * sizeof(int));
    t->length  = 0.0;
    t->fitness = 0.0;
    if (t->cities == NULL) return GA_ERR_ALLOC;
    return GA_OK;
}

/* ---- tour_free --------------------------------------------------------- */

void tour_free(Tour *t)
{
    if (t == NULL) return;
    free(t->cities);
    t->cities  = NULL;
    t->length  = 0.0;
    t->fitness = 0.0;
}

/* ---- tour_copy --------------------------------------------------------- */

void tour_copy(Tour *dst, const Tour *src, int n)
{
    memcpy(dst->cities, src->cities, (size_t)n * sizeof(int));
    dst->length  = src->length;
    dst->fitness = src->fitness;
}

/* ---- tour_validate ----------------------------------------------------- */

bool tour_validate(const Tour *t, int n)
{
    if (t == NULL)          return false;
    if (t->cities == NULL)  return false;
    if (n <= 0)             return false;

    /* C99 VLA -- O(n) stack, no heap allocation */
    bool seen[n];
    memset(seen, 0, (size_t)n * sizeof(bool));

    for (int i = 0; i < n; i++) {
        int c = t->cities[i];
        if (c < 0 || c >= n)  return false;   /* out of range */
        if (seen[c])           return false;   /* duplicate    */
        seen[c] = true;
    }
    return true;
}
