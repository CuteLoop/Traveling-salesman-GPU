/*
 * test_ga_smoke.c -- Phase 9 tests: GA driver integration smoke tests
 *
 * Written BEFORE implementation (TDD).
 * Uses smoke_20.tsp fixture for all tests.
 */

#include "ga.h"
#include "instance.h"
#include "tour.h"
#include "ga_driver.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(msg, expr) do { \
    tests_run++; \
    if (expr) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while (0)

/* ---- helpers ---------------------------------------------------------- */

static const char *FIXTURE = "tests/fixtures/smoke_20.tsp";
static const char *CSV_TMP = "tests/fixtures/_test_results.csv";

static GAConfig default_config(void)
{
    GAConfig cfg;
    cfg.population_size = 30;
    cfg.elite_count     = 2;
    cfg.tournament_k    = 3;
    cfg.crossover_prob  = 0.9;
    cfg.mutation_prob   = 0.1;
    cfg.generations     = 20;
    cfg.seed            = 42;
    return cfg;
}

/* ---- G-01: GA runs without crashing and returns GA_OK ----------------- */
static void test_g01(void)
{
    TSPInstance inst = {0};
    ga_status_t rc_load = tsp_instance_load(FIXTURE, &inst);
    ASSERT("G-01 load ok", rc_load == GA_OK);
    tsp_instance_build_distance_matrix(&inst);

    GAConfig cfg = default_config();
    GAStats  stats = {0};
    int rc_alloc = ga_stats_alloc(&stats, cfg.generations + 1);
    ASSERT("G-01 stats alloc ok", rc_alloc == GA_OK);

    Tour best = {0};
    tour_alloc(&best, inst.n);

    int rc = ga_run(&cfg, &inst, &stats, &best);
    ASSERT("G-01 ga_run returns GA_OK", rc == GA_OK);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- G-02: Final best distance is finite and positive ----------------- */
static void test_g02(void)
{
    TSPInstance inst = {0};
    tsp_instance_load(FIXTURE, &inst);
    tsp_instance_build_distance_matrix(&inst);

    GAConfig cfg = default_config();
    GAStats  stats = {0};
    ga_stats_alloc(&stats, cfg.generations + 1);

    Tour best = {0};
    tour_alloc(&best, inst.n);

    ga_run(&cfg, &inst, &stats, &best);

    double final_best = stats.best[stats.generations_logged - 1];
    ASSERT("G-02 final best is finite", isfinite(final_best));
    ASSERT("G-02 final best is positive", final_best > 0.0);
    ASSERT("G-02 out_best.length matches", best.length == final_best);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- G-03: Best-so-far distance is monotonically non-increasing ------- */
static void test_g03(void)
{
    TSPInstance inst = {0};
    tsp_instance_load(FIXTURE, &inst);
    tsp_instance_build_distance_matrix(&inst);

    GAConfig cfg = default_config();
    cfg.generations = 50;   /* more generations to test convergence */
    GAStats  stats = {0};
    ga_stats_alloc(&stats, cfg.generations + 1);

    Tour best = {0};
    tour_alloc(&best, inst.n);

    ga_run(&cfg, &inst, &stats, &best);

    int monotonic = 1;
    for (int g = 1; g < stats.generations_logged; g++) {
        if (stats.best[g] > stats.best[g - 1] + 1e-9) {
            monotonic = 0;
            printf("    G-03 violation: gen %d best=%.6f > gen %d best=%.6f\n",
                   g, stats.best[g], g - 1, stats.best[g - 1]);
            break;
        }
    }
    ASSERT("G-03 best distance is monotonically non-increasing", monotonic);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- G-04: Stats contain exactly generations+1 entries ---------------- */
static void test_g04(void)
{
    TSPInstance inst = {0};
    tsp_instance_load(FIXTURE, &inst);
    tsp_instance_build_distance_matrix(&inst);

    GAConfig cfg = default_config();
    cfg.generations = 15;
    GAStats  stats = {0};
    ga_stats_alloc(&stats, cfg.generations + 1);

    Tour best = {0};
    tour_alloc(&best, inst.n);

    ga_run(&cfg, &inst, &stats, &best);

    ASSERT("G-04 generations_logged == generations+1",
           stats.generations_logged == cfg.generations + 1);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- G-05: CSV export writes expected rows ---------------------------- */
static void test_g05(void)
{
    TSPInstance inst = {0};
    tsp_instance_load(FIXTURE, &inst);
    tsp_instance_build_distance_matrix(&inst);

    GAConfig cfg = default_config();
    cfg.generations = 10;
    GAStats  stats = {0};
    ga_stats_alloc(&stats, cfg.generations + 1);

    Tour best = {0};
    tour_alloc(&best, inst.n);

    ga_run(&cfg, &inst, &stats, &best);
    int rc = ga_stats_write_csv(&stats, CSV_TMP);
    ASSERT("G-05 CSV write returns GA_OK", rc == GA_OK);

    /* Count lines: 1 header + (generations+1) data rows */
    FILE *f = fopen(CSV_TMP, "r");
    ASSERT("G-05 CSV file opens", f != NULL);

    int lines = 0;
    if (f) {
        char buf[256];
        while (fgets(buf, sizeof(buf), f)) lines++;
        fclose(f);
    }

    /* Expected: 1 header + (generations+1) data = generations+2 */
    ASSERT("G-05 CSV has generations+2 lines",
           lines == cfg.generations + 2);

    /* Cleanup temp file */
    remove(CSV_TMP);

    tour_free(&best);
    ga_stats_free(&stats);
    tsp_instance_free(&inst);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_ga_smoke\n");

    test_g01();
    test_g02();
    test_g03();
    test_g04();
    test_g05();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
