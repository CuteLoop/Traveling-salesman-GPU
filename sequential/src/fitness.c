/*
 * fitness.c -- Tour fitness evaluation (single + batch)
 *
 * Phase 3 implementation.
 *
 * Closed-loop tour length: sum of dist(cities[k], cities[k+1]) for
 * k = 0..n-2, plus dist(cities[n-1], cities[0]) to close the loop.
 *
 * No heap allocation.  Pure function on pre-allocated data.
 */

#include "fitness.h"
#include "instance.h"
#include "tour.h"

#include <stddef.h>

/* ---- tour_evaluate ---------------------------------------------------- */

int tour_evaluate(Tour *t, const TSPInstance *inst)
{
    if (t == NULL || inst == NULL)       return GA_ERR_INVALID;
    if (t->cities == NULL)               return GA_ERR_INVALID;
    if (inst->dist == NULL)              return GA_ERR_INVALID;
    if (inst->n <= 0)                    return GA_ERR_INVALID;

    int n = inst->n;

    /* Special case: single-city tour has zero length, zero fitness */
    if (n == 1) {
        t->length  = 0.0;
        t->fitness = 0.0;
        return GA_OK;
    }

    double total = 0.0;
    for (int k = 0; k < n - 1; k++) {
        total += DIST(inst, t->cities[k], t->cities[k + 1]);
    }
    /* Close the loop */
    total += DIST(inst, t->cities[n - 1], t->cities[0]);

    t->length  = total;
    t->fitness = (total > 0.0) ? (1.0 / total) : 0.0;

    return GA_OK;
}

/* ---- population_evaluate ---------------------------------------------- */

int population_evaluate(Tour *pop, int N, const TSPInstance *inst)
{
    if (pop == NULL || inst == NULL) return GA_ERR_INVALID;
    if (N <= 0)                     return GA_ERR_INVALID;

    for (int i = 0; i < N; i++) {
        int rc = tour_evaluate(&pop[i], inst);
        if (rc != GA_OK) return rc;
    }
    return GA_OK;
}
