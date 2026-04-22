/*
 * test_regression.c -- Phase 10: Numeric regression test
 *
 * Ensures the GA produces bit-reproducible results across builds.
 * Uses smoke_20.tsp, pop=100, gen=200, seed=42.
 *
 * Reg-01: best distance must match locked value within 1e-6.
 */

#include "ga.h"
#include "instance.h"
#include "tour.h"
#include "ga_driver.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(msg, expr) do { \
    tests_run++; \
    if (expr) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while (0)

static const char *FIXTURE = "tests/fixtures/smoke_20.tsp";

/*
 * LOCKED REGRESSION VALUE
 *
 * Obtained from: ga-tsp --instance smoke_20.tsp --pop 100 --gen 200 --seed 42
 * Build: gcc -std=c99 -Wall -Wextra -Wpedantic (debug, no optimizations)
 * Date locked: 2026-04-15
 *
 * If this value changes, either:
 *   (a) a GA operator was modified (intentional — update the value), or
 *   (b) a bug was introduced (investigate immediately).
 */
static const double LOCKED_BEST_DISTANCE = 77.492495;
static const double REGRESSION_TOLERANCE = 1e-6;

/* ---- Reg-01: deterministic best distance ------------------------------ */
static void test_reg01(void)
{
    TSPInstance inst = {0};
    ga_status_t rc = tsp_instance_load(FIXTURE, &inst);
    ASSERT("Reg-01 load ok", rc == GA_OK);
    rc = tsp_instance_build_distance_matrix(&inst);
    ASSERT("Reg-01 dist ok", rc == GA_OK);

    GAConfig cfg;
    cfg.population_size = 100;
    cfg.elite_count     = 2;
    cfg.tournament_k    = 3;
    cfg.crossover_prob  = 0.9;
    cfg.mutation_prob   = 0.1;
    cfg.generations     = 200;
    cfg.seed            = 42;

    GAStats stats = {0};
    rc = ga_stats_alloc(&stats, cfg.generations + 1);
    ASSERT("Reg-01 stats alloc ok", rc == GA_OK);

    Tour best = {0};
    rc = tour_alloc(&best, inst.n);
    ASSERT("Reg-01 tour alloc ok", rc == GA_OK);

    rc = ga_run(&cfg, &inst, &stats, &best);
    ASSERT("Reg-01 ga_run ok", rc == GA_OK);

    double diff = fabs(best.length - LOCKED_BEST_DISTANCE);
    printf("  Reg-01 best distance: %.15f\n", best.length);
    printf("  Reg-01 locked value:  %.15f\n", LOCKED_BEST_DISTANCE);
    printf("  Reg-01 difference:    %.15e\n", diff);
    ASSERT("Reg-01 distance within tolerance", diff < REGRESSION_TOLERANCE);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- Reg-02: stats monotonicity preserved under any build ------------- */
static void test_reg02(void)
{
    TSPInstance inst = {0};
    ga_status_t rc = tsp_instance_load(FIXTURE, &inst);
    ASSERT("Reg-02 load ok", rc == GA_OK);
    rc = tsp_instance_build_distance_matrix(&inst);
    ASSERT("Reg-02 dist ok", rc == GA_OK);

    GAConfig cfg;
    cfg.population_size = 100;
    cfg.elite_count     = 2;
    cfg.tournament_k    = 3;
    cfg.crossover_prob  = 0.9;
    cfg.mutation_prob   = 0.1;
    cfg.generations     = 200;
    cfg.seed            = 42;

    GAStats stats = {0};
    rc = ga_stats_alloc(&stats, cfg.generations + 1);
    ASSERT("Reg-02 stats alloc", rc == GA_OK);

    Tour best = {0};
    rc = tour_alloc(&best, inst.n);
    ASSERT("Reg-02 tour alloc", rc == GA_OK);

    rc = ga_run(&cfg, &inst, &stats, &best);
    ASSERT("Reg-02 ga_run ok", rc == GA_OK);

    /* Best distance must be non-increasing across generations */
    int monotonic = 1;
    for (int g = 1; g < stats.generations_logged; g++) {
        if (stats.best[g] > stats.best[g - 1] + 1e-12) {
            printf("  Reg-02 monotonicity violated at gen %d: %.6f > %.6f\n",
                   g, stats.best[g], stats.best[g - 1]);
            monotonic = 0;
            break;
        }
    }
    ASSERT("Reg-02 best distance non-increasing", monotonic);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- Reg-03: multiple seeds all produce valid results ----------------- */
static void test_reg03(void)
{
    static const uint64_t seeds[] = {42, 123, 999, 5555, 9876};
    static const int NSEEDS = 5;

    TSPInstance inst = {0};
    ga_status_t rc = tsp_instance_load(FIXTURE, &inst);
    ASSERT("Reg-03 load ok", rc == GA_OK);
    rc = tsp_instance_build_distance_matrix(&inst);
    ASSERT("Reg-03 dist ok", rc == GA_OK);

    int all_ok = 1;
    for (int s = 0; s < NSEEDS; s++) {
        GAConfig cfg;
        cfg.population_size = 50;
        cfg.elite_count     = 2;
        cfg.tournament_k    = 3;
        cfg.crossover_prob  = 0.9;
        cfg.mutation_prob   = 0.1;
        cfg.generations     = 50;
        cfg.seed            = seeds[s];

        GAStats stats = {0};
        rc = ga_stats_alloc(&stats, cfg.generations + 1);
        if (rc != GA_OK) { all_ok = 0; continue; }

        Tour best = {0};
        rc = tour_alloc(&best, inst.n);
        if (rc != GA_OK) { ga_stats_free(&stats); all_ok = 0; continue; }

        rc = ga_run(&cfg, &inst, &stats, &best);
        if (rc != GA_OK) { all_ok = 0; }
        if (best.length <= 0.0 || !isfinite(best.length)) { all_ok = 0; }

        tour_free(&best);
        ga_stats_free(&stats);
    }
    ASSERT("Reg-03 all seeds produce valid results", all_ok);

    tsp_instance_free(&inst);
}

/* ---- Reg-04: release build reproduces same value (self-check) --------- */
static void test_reg04(void)
{
    /* Run twice with same config, verify identical output */
    TSPInstance inst = {0};
    ga_status_t rc = tsp_instance_load(FIXTURE, &inst);
    ASSERT("Reg-04 load ok", rc == GA_OK);
    rc = tsp_instance_build_distance_matrix(&inst);
    ASSERT("Reg-04 dist ok", rc == GA_OK);

    GAConfig cfg;
    cfg.population_size = 100;
    cfg.elite_count     = 2;
    cfg.tournament_k    = 3;
    cfg.crossover_prob  = 0.9;
    cfg.mutation_prob   = 0.1;
    cfg.generations     = 200;
    cfg.seed            = 42;

    double distances[2];
    for (int run = 0; run < 2; run++) {
        GAStats stats = {0};
        rc = ga_stats_alloc(&stats, cfg.generations + 1);
        ASSERT("Reg-04 stats alloc", rc == GA_OK);

        Tour best = {0};
        rc = tour_alloc(&best, inst.n);
        ASSERT("Reg-04 tour alloc", rc == GA_OK);

        rc = ga_run(&cfg, &inst, &stats, &best);
        ASSERT("Reg-04 ga_run ok", rc == GA_OK);

        distances[run] = best.length;

        tour_free(&best);
        ga_stats_free(&stats);
    }
    ASSERT("Reg-04 two runs produce identical distance",
           fabs(distances[0] - distances[1]) < 1e-15);

    tsp_instance_free(&inst);
}

int main(void)
{
    printf("test_regression:\n");
    test_reg01();
    test_reg02();
    test_reg03();
    test_reg04();
    printf("test_regression: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
