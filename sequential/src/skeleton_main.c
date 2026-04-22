/*
 * skeleton_main.c -- Walking skeleton: init + evaluate + print best
 *
 * Phase 4 integration executable.
 *
 * Usage:
 *   skeleton --instance <path.tsp> --pop <N> --seed <val>
 *
 * Behaviour:
 *   1. Parse CLI args
 *   2. Load TSPInstance + distance matrix
 *   3. Allocate N tours + N RNGStates
 *   4. Seed each RNGState deterministically (base_seed + i)
 *   5. population_init
 *   6. population_evaluate
 *   7. Find and print best tour length
 *   8. Free all memory, exit 0
 *
 * No crossover, selection, or mutation.
 */

#include "ga.h"
#include "instance.h"
#include "tour.h"
#include "fitness.h"
#include "rng.h"
#include "init.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- simple CLI parser ------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --instance <file.tsp> --pop <size> --seed <val>\n",
            prog);
}

int main(int argc, char *argv[])
{
    const char *instance_path = NULL;
    int    pop_size  = 0;
    uint64_t seed    = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--instance") == 0 && i + 1 < argc) {
            instance_path = argv[++i];
        } else if (strcmp(argv[i], "--pop") == 0 && i + 1 < argc) {
            pop_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        }
    }

    if (instance_path == NULL || pop_size <= 0) {
        usage(argv[0]);
        return 1;
    }

    /* 1. Load instance */
    TSPInstance inst;
    ga_status_t s = tsp_instance_load(instance_path, &inst);
    if (s != GA_OK) {
        fprintf(stderr, "Error: failed to load instance '%s'\n",
                instance_path);
        return 1;
    }

    s = tsp_instance_build_distance_matrix(&inst);
    if (s != GA_OK) {
        fprintf(stderr, "Error: failed to build distance matrix\n");
        tsp_instance_free(&inst);
        return 1;
    }

    int N = pop_size;
    int n = inst.n;

    /* 2. Allocate population + RNG states */
    Tour     *pop    = malloc((size_t)N * sizeof(Tour));
    RNGState *states = malloc((size_t)N * sizeof(RNGState));
    if (pop == NULL || states == NULL) {
        fprintf(stderr, "Error: allocation failed\n");
        free(pop);
        free(states);
        tsp_instance_free(&inst);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        s = tour_alloc(&pop[i], n);
        if (s != GA_OK) {
            /* Free already-allocated tours */
            for (int j = 0; j < i; j++) tour_free(&pop[j]);
            free(pop);
            free(states);
            tsp_instance_free(&inst);
            return 1;
        }
        rng_seed(&states[i], seed + (uint64_t)i);
    }

    /* 3. Initialize population */
    s = population_init(pop, N, n, states);
    if (s != GA_OK) {
        fprintf(stderr, "Error: population_init failed\n");
        for (int i = 0; i < N; i++) tour_free(&pop[i]);
        free(pop);
        free(states);
        tsp_instance_free(&inst);
        return 1;
    }

    /* 4. Evaluate population */
    int rc = population_evaluate(pop, N, &inst);
    if (rc != GA_OK) {
        fprintf(stderr, "Error: population_evaluate failed\n");
        for (int i = 0; i < N; i++) tour_free(&pop[i]);
        free(pop);
        free(states);
        tsp_instance_free(&inst);
        return 1;
    }

    /* 5. Find best tour */
    double best_len = pop[0].length;
    int    best_idx = 0;
    for (int i = 1; i < N; i++) {
        if (pop[i].length < best_len) {
            best_len = pop[i].length;
            best_idx = i;
        }
    }

    printf("Walking Skeleton Complete. Best length: %.6f\n", best_len);
    printf("  (individual %d of %d, seed = %llu, n = %d)\n",
           best_idx, N, (unsigned long long)seed, n);

    /* 6. Cleanup */
    for (int i = 0; i < N; i++) tour_free(&pop[i]);
    free(pop);
    free(states);
    tsp_instance_free(&inst);

    return 0;
}
