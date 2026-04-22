/*
 * main.c — ga-tsp CLI driver
 *
 * Phase 9: Full GA execution with command-line arguments.
 *
 * Usage:
 *   ga-tsp --instance <file> [--pop N] [--gen G] [--seed S]
 *          [--elites E] [--pc P] [--pm P] [--csv <file>]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ga.h"
#include "instance.h"
#include "tour.h"
#include "ga_driver.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --instance <file> [options]\n"
        "  --instance <file>   TSP instance file (required)\n"
        "  --pop <N>           Population size (default: 100)\n"
        "  --gen <G>           Generations (default: 200)\n"
        "  --seed <S>          RNG seed (default: 42)\n"
        "  --elites <E>        Elite count (default: 2)\n"
        "  --tk <K>            Tournament k (default: 3)\n"
        "  --pc <P>            Crossover probability (default: 0.9)\n"
        "  --pm <P>            Mutation probability (default: 0.1)\n"
        "  --csv <file>        CSV output file (default: results.csv)\n",
        prog);
}

int main(int argc, char *argv[])
{
    /* Defaults */
    const char *instance_path = NULL;
    const char *csv_path      = "results.csv";
    GAConfig cfg;
    cfg.population_size = 100;
    cfg.elite_count     = 2;
    cfg.tournament_k    = 3;
    cfg.crossover_prob  = 0.9;
    cfg.mutation_prob   = 0.1;
    cfg.generations     = 200;
    cfg.seed            = 42;

    /* Parse CLI */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--instance") == 0 && i + 1 < argc) {
            instance_path = argv[++i];
        } else if (strcmp(argv[i], "--pop") == 0 && i + 1 < argc) {
            cfg.population_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc) {
            cfg.generations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--elites") == 0 && i + 1 < argc) {
            cfg.elite_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tk") == 0 && i + 1 < argc) {
            cfg.tournament_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pc") == 0 && i + 1 < argc) {
            cfg.crossover_prob = atof(argv[++i]);
        } else if (strcmp(argv[i], "--pm") == 0 && i + 1 < argc) {
            cfg.mutation_prob = atof(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (instance_path == NULL) {
        fprintf(stderr, "Error: --instance is required.\n");
        usage(argv[0]);
        return 1;
    }

    /* Load instance */
    TSPInstance inst = {0};
    ga_status_t rc = tsp_instance_load(instance_path, &inst);
    if (rc != GA_OK) {
        fprintf(stderr, "Error: failed to load instance '%s' (code %d)\n",
                instance_path, rc);
        tsp_instance_free(&inst);
        return 1;
    }
    rc = tsp_instance_build_distance_matrix(&inst);
    if (rc != GA_OK) {
        fprintf(stderr, "Error: failed to build distance matrix (code %d)\n", rc);
        tsp_instance_free(&inst);
        return 1;
    }

    /* Allocate stats and best tour */
    GAStats stats = {0};
    rc = ga_stats_alloc(&stats, cfg.generations + 1);
    if (rc != GA_OK) {
        fprintf(stderr, "Error: stats allocation failed (code %d)\n", rc);
        tsp_instance_free(&inst);
        return 1;
    }

    Tour best = {0};
    rc = tour_alloc(&best, inst.n);
    if (rc != GA_OK) {
        fprintf(stderr, "Error: tour allocation failed (code %d)\n", rc);
        ga_stats_free(&stats);
        tsp_instance_free(&inst);
        return 1;
    }

    /* Run GA */
    printf("ga-tsp: instance=%s  n=%d  pop=%d  gen=%d  seed=%llu\n",
           instance_path, inst.n, cfg.population_size, cfg.generations,
           (unsigned long long)cfg.seed);

    rc = ga_run(&cfg, &inst, &stats, &best);
    if (rc != GA_OK) {
        fprintf(stderr, "Error: ga_run failed (code %d)\n", rc);
        tour_free(&best);
        ga_stats_free(&stats);
        tsp_instance_free(&inst);
        return 1;
    }

    /* Report */
    printf("Best distance: %.6f\n", best.length);
    printf("Best tour (0-based indices):\n");
    for (int i = 0; i < inst.n; i++) {
        printf("%d ", best.cities[i]);
    }
    printf("%d\n", best.cities[0]);

    /* Export CSV */
    rc = ga_stats_write_csv(&stats, csv_path);
    if (rc == GA_OK) {
        printf("Statistics written to %s (%d generations)\n",
               csv_path, stats.generations_logged);
    } else {
        fprintf(stderr, "Warning: CSV export failed (code %d)\n", rc);
    }

    /* Cleanup */
    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);

    return 0;
}
