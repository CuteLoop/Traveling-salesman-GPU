/*
 * ga_driver.c -- GA main loop: orchestrates Phases 1-8 modules
 *
 * Phase 9 implementation.
 *
 * The driver contains NO raw GA logic.  It strictly orchestrates the
 * APIs from Phases 1-8.  This is the seam where future CUDA kernel
 * launches replace serial loops.
 *
 * Memory ownership:
 *   ga_run allocates and frees all internal buffers (pop, next_pop,
 *   offspring, elites, rng_states).  The caller owns cfg, inst, stats,
 *   and out_best.
 *
 * Double-buffering:
 *   pop and next_pop are swapped via O(1) pointer swap each generation.
 *   No deep copies for the generational transition.
 */

#include "ga_driver.h"
#include "rng.h"
#include "init.h"
#include "fitness.h"
#include "selection.h"
#include "crossover.h"
#include "mutation.h"
#include "elitism.h"
#include "replacement.h"

#include <stdlib.h>
#include <float.h>

/* ---- internal helpers ------------------------------------------------- */

/*
 * Allocate an array of N tours, each with n cities.
 * On failure, frees any partially allocated tours and returns NULL.
 */
static Tour *alloc_pop(int N, int n)
{
    Tour *arr = malloc((size_t)N * sizeof(Tour));
    if (arr == NULL) return NULL;

    for (int i = 0; i < N; i++) {
        if (tour_alloc(&arr[i], n) != GA_OK) {
            /* Cleanup partial allocation */
            for (int j = 0; j < i; j++) tour_free(&arr[j]);
            free(arr);
            return NULL;
        }
    }
    return arr;
}

/* Free an array of N tours, then the array itself. */
static void free_pop(Tour *arr, int N)
{
    if (arr == NULL) return;
    for (int i = 0; i < N; i++) tour_free(&arr[i]);
    free(arr);
}

/*
 * Compute population statistics (distance-based, not fitness).
 * Writes best/mean/worst distance into stats at index g.
 * Returns the index of the best individual.
 */
static int log_stats(const Tour *pop, int N, GAStats *stats, int g)
{
    double sum  = 0.0;
    double best = DBL_MAX;
    double wrst = 0.0;
    int best_idx = 0;

    for (int i = 0; i < N; i++) {
        double d = pop[i].length;
        sum += d;
        if (d < best) { best = d; best_idx = i; }
        if (d > wrst) { wrst = d; }
    }

    stats->best[g]  = best;
    stats->mean[g]  = sum / (double)N;
    stats->worst[g] = wrst;
    stats->generations_logged = g + 1;

    return best_idx;
}

/* ---- ga_run ----------------------------------------------------------- */

int ga_run(const GAConfig *cfg, const TSPInstance *inst,
           GAStats *stats, Tour *out_best)
{
    /* --- argument validation ------------------------------------------ */
    if (cfg == NULL || inst == NULL || stats == NULL || out_best == NULL)
        return GA_ERR_INVALID;

    int N = cfg->population_size;
    int e = cfg->elite_count;
    int n = inst->n;
    int G = cfg->generations;
    int oc = N - e;  /* offspring count */

    if (N < 2)                    return GA_ERR_INVALID;
    if (e < 0 || e > N)          return GA_ERR_INVALID;
    if (cfg->tournament_k < 1)   return GA_ERR_INVALID;
    if (cfg->crossover_prob < 0.0 || cfg->crossover_prob > 1.0)
                                  return GA_ERR_INVALID;
    if (cfg->mutation_prob < 0.0 || cfg->mutation_prob > 1.0)
                                  return GA_ERR_INVALID;
    if (G < 0)                    return GA_ERR_INVALID;

    /* --- allocate internal buffers ------------------------------------ */
    Tour *pop      = alloc_pop(N, n);
    Tour *next_pop = alloc_pop(N, n);
    Tour *offspring = (oc > 0) ? alloc_pop(oc, n) : NULL;
    Tour *elites    = (e  > 0) ? alloc_pop(e, n)  : NULL;

    if (pop == NULL || next_pop == NULL ||
        (oc > 0 && offspring == NULL) ||
        (e  > 0 && elites == NULL)) {
        free_pop(pop, N);
        free_pop(next_pop, N);
        free_pop(offspring, oc);
        free_pop(elites, e);
        return GA_ERR_ALLOC;
    }

    /* Allocate per-individual RNG states */
    RNGState *rng_states = malloc((size_t)N * sizeof(RNGState));
    if (rng_states == NULL) {
        free_pop(pop, N);
        free_pop(next_pop, N);
        free_pop(offspring, oc);
        free_pop(elites, e);
        return GA_ERR_ALLOC;
    }

    /* --- seed RNG states ---------------------------------------------- */
    for (int i = 0; i < N; i++) {
        rng_seed(&rng_states[i], cfg->seed + (uint64_t)i);
    }

    /* --- Generation 0: init + eval + log ------------------------------ */
    GA_RETURN_IF_ERR(population_init(pop, N, n, rng_states));
    GA_RETURN_IF_ERR(population_evaluate(pop, N, inst));

    int best_idx = log_stats(pop, N, stats, 0);

    /* Track absolute best distance across all generations */
    double abs_best_dist = pop[best_idx].length;
    tour_copy(out_best, &pop[best_idx], n);

    /* --- generational loop -------------------------------------------- */
    for (int g = 1; g <= G; g++) {

        /* a. Extract elites */
        if (e > 0) {
            GA_RETURN_IF_ERR(extract_elites(elites, pop, N, e, n));
        }

        /* b. Generate offspring: select + crossover + mutate */
        for (int i = 0; i < oc; i++) {
            int pa, pb;
            GA_RETURN_IF_ERR(select_parent_pair(pop, N, cfg->tournament_k,
                                                &rng_states[i], &pa, &pb));
            GA_RETURN_IF_ERR(apply_crossover(&offspring[i], &pop[pa], &pop[pb],
                                             n, cfg->crossover_prob,
                                             &rng_states[i]));
            GA_RETURN_IF_ERR(apply_mutation_swap(&offspring[i], n,
                                                 cfg->mutation_prob,
                                                 &rng_states[i]));
        }

        /* c. Evaluate offspring */
        GA_RETURN_IF_ERR(population_evaluate(offspring, oc, inst));

        /* d. Build next generation */
        GA_RETURN_IF_ERR(build_next_generation(next_pop, elites, e,
                                               offspring, oc, N, n));

        /* e. O(1) pointer swap — no deep copy */
        {
            Tour *tmp = pop;
            pop = next_pop;
            next_pop = tmp;
        }

        /* f. Log stats and update absolute best */
        best_idx = log_stats(pop, N, stats, g);
        if (pop[best_idx].length < abs_best_dist) {
            abs_best_dist = pop[best_idx].length;
            tour_copy(out_best, &pop[best_idx], n);
        }
    }

    /* --- cleanup ------------------------------------------------------ */
    free_pop(pop, N);
    free_pop(next_pop, N);
    free_pop(offspring, oc);
    free_pop(elites, e);
    free(rng_states);

    return GA_OK;
}
