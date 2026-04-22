/*
 * ga_driver.h -- GA configuration, statistics, and driver
 *
 * Phase 9 public API.
 *
 * Named ga_driver.h to avoid collision with ga.h (status codes).
 *
 * The driver orchestrates Phases 1-8 module APIs.  It contains no raw
 * GA logic — only coordination.  This is the seam where future CUDA
 * kernel launches replace serial loops.
 *
 * Memory ownership:
 *   GAStats owns the best/mean/worst arrays.  Caller must call
 *   ga_stats_free() after use.  ga_run() writes into caller-supplied
 *   stats and out_best; it manages all internal buffers.
 */

#ifndef GA_TSP_DRIVER_H
#define GA_TSP_DRIVER_H

#include <stdint.h>
#include "ga.h"
#include "instance.h"
#include "tour.h"

/* ---- GAConfig --------------------------------------------------------- */

typedef struct {
    int      population_size;   /* N individuals                          */
    int      elite_count;       /* e elites preserved per generation      */
    int      tournament_k;      /* k for tournament selection             */
    double   crossover_prob;    /* p_c in [0.0, 1.0]                     */
    double   mutation_prob;     /* p_m in [0.0, 1.0]                     */
    int      generations;       /* G generational iterations              */
    uint64_t seed;              /* base RNG seed                          */
} GAConfig;

/* ---- GAStats ----------------------------------------------------------
 *
 * Logs per-generation best, mean, and worst DISTANCE (not fitness).
 * Generation 0 is the initial (un-evolved) population.
 * A run of G generations produces G+1 log entries.
 * --------------------------------------------------------------------- */

typedef struct {
    int     generations_logged; /* number of entries written (should be G+1) */
    double *best;               /* [generations_logged] best distance        */
    double *mean;               /* [generations_logged] mean distance        */
    double *worst;              /* [generations_logged] worst distance       */
} GAStats;

/* ---- function declarations -------------------------------------------- */

/*
 * ga_stats_alloc -- allocate arrays for count log entries.
 *
 * count should be cfg->generations + 1 (to include Generation 0).
 * Returns GA_OK on success, GA_ERR_ALLOC on failure.
 */
int ga_stats_alloc(GAStats *stats, int count);

/*
 * ga_stats_free -- release all memory owned by stats.
 *
 * Safe on NULL, safe on repeated calls.
 */
void ga_stats_free(GAStats *stats);

/*
 * ga_stats_write_csv -- write statistics to a CSV file.
 *
 * Header: generation,best,mean,worst
 * One row per logged generation.
 *
 * Returns GA_OK on success, GA_ERR_IO on file error.
 */
int ga_stats_write_csv(const GAStats *stats, const char *path);

/*
 * ga_run -- execute the full GA loop.
 *
 * Orchestrates init → eval → [extract_elites → select+cross → mutate →
 * eval offspring → build_next_gen → log stats] × G.
 *
 * Parameters:
 *   cfg       Configuration (read-only).
 *   inst      Loaded TSP instance with distance matrix (read-only).
 *   stats     Pre-allocated GAStats (via ga_stats_alloc, count = G+1).
 *   out_best  Pre-allocated Tour (via tour_alloc, n = inst->n).
 *             Receives the best tour found across all generations.
 *
 * Returns GA_OK on success, error code on failure.
 * All internal memory is freed before return.
 */
int ga_run(const GAConfig *cfg, const TSPInstance *inst,
           GAStats *stats, Tour *out_best);

#endif /* GA_TSP_DRIVER_H */
