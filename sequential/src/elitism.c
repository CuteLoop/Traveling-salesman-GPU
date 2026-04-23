/*
 * elitism.c -- Elite extraction from a population
 *
 * Phase 8 implementation.
 *
 * Uses a local EliteRef struct + qsort to find the top-e individuals
 * without any global variables (C99-safe).
 *
 * GPU: replace with bitonic Top-K kernel
 */

#include "elitism.h"

#include <stdlib.h>
#include <string.h>

/* ---- EliteRef -----------------------------------------------------------
 *
 * Local helper for qsort.  Bundles original index + fitness so the
 * comparator needs no global state.  This is the C99-safe alternative
 * to qsort_r / qsort_s.
 * --------------------------------------------------------------------- */
typedef struct {
    int    index;
    double fitness;
} EliteRef;

/* Comparator: descending fitness, tie-break by ascending index. */
static int cmp_elite_desc(const void *a, const void *b)
{
    const EliteRef *ea = (const EliteRef *)a;
    const EliteRef *eb = (const EliteRef *)b;

    if (ea->fitness > eb->fitness) return -1;
    if (ea->fitness < eb->fitness) return  1;
    /* Equal fitness: lower original index first */
    if (ea->index < eb->index) return -1;
    if (ea->index > eb->index) return  1;
    return 0;
}

/* ---- extract_elites --------------------------------------------------- */

ga_status_t extract_elites(Tour *elites, const Tour *pop,
                           int N, int e, int n)
{
    /* --- argument validation -------------------------------------------*/
    if (e == 0) return GA_OK;                  /* nothing to do          */
    if (pop == NULL)       return GA_ERR_INVALID;
    if (elites == NULL)    return GA_ERR_INVALID;
    if (N <= 0)            return GA_ERR_INVALID;
    if (e < 0 || e > N)   return GA_ERR_INVALID;
    if (n <= 0)            return GA_ERR_INVALID;

    /* --- build EliteRef array (temporary) ----------------------------- */
    EliteRef *refs = malloc((size_t)N * sizeof(EliteRef));
    if (refs == NULL) return GA_ERR_ALLOC;

    for (int i = 0; i < N; i++) {
        refs[i].index   = i;
        refs[i].fitness = pop[i].fitness;
    }

    /* --- sort descending by fitness ----------------------------------- */
    /* GPU: replace with bitonic Top-K kernel */
    qsort(refs, (size_t)N, sizeof(EliteRef), cmp_elite_desc);

    /* --- deep-copy top-e into elites[] -------------------------------- */
    for (int i = 0; i < e; i++) {
        tour_copy(&elites[i], &pop[refs[i].index], n);
    }

    free(refs);
    return GA_OK;
}
