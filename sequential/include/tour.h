/*
 * tour.h -- Tour representation, validation, copy
 *
 * Phase 2 public API.
 *
 * Ownership:
 *   Tour owns the cities array.  tour_free() must be called once
 *   for every tour touched by tour_alloc().
 */

#ifndef TSP_TOUR_H
#define TSP_TOUR_H

#include <stdbool.h>
#include "ga.h"

/* ---- Tour ---------------------------------------------------------------
 *
 * Fields:
 *   cities   Heap-allocated array of n city indices (0-indexed permutation).
 *            NULL when uninitialised or freed.
 *   length   Total closed-tour distance.  0.0 until evaluated.
 *   fitness  Inverse of length (or similar).  0.0 until evaluated.
 * --------------------------------------------------------------------- */
typedef struct {
    int    *cities;
    double  length;
    double  fitness;
} Tour;

/* ---- function declarations -------------------------------------------- */

/*
 * tour_alloc -- allocate a tour for n cities.
 *
 * Sets cities to heap array of n ints, length and fitness to 0.0.
 * Returns GA_OK on success, GA_ERR_ALLOC on malloc failure.
 */
ga_status_t tour_alloc(Tour *t, int n);

/*
 * tour_free -- release the cities array.
 *
 * Nullifies cities pointer, resets length/fitness to 0.0.
 * Safe on NULL, safe on repeated calls.
 */
void tour_free(Tour *t);

/*
 * tour_copy -- deep-copy src into dst (both must be allocated for n cities).
 *
 * Copies all n city indices and the length/fitness scalars.
 * dst->cities must already be allocated (via tour_alloc).
 * No aliasing: dst->cities != src->cities after the call.
 */
void tour_copy(Tour *dst, const Tour *src, int n);

/*
 * tour_validate -- check that the tour is a valid permutation of [0, n).
 *
 * Pure function.  O(n) time, O(n) stack space (C99 VLA).
 * No heap allocation.
 *
 * Returns true  if t is non-NULL, t->cities is non-NULL, n >= 1,
 *               and cities is exactly the set {0, 1, ..., n-1}.
 * Returns false otherwise.
 */
bool tour_validate(const Tour *t, int n);

#endif /* TSP_TOUR_H */
