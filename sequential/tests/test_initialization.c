/*
 * test_initialization.c -- Phase 4 tests: tour init, population init,
 *                          walking skeleton integration
 *
 * Uses square_4.tsp fixture.
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "instance.h"
#include "tour.h"
#include "fitness.h"
#include "rng.h"
#include "init.h"

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

#define APPROX_EQ(a, b) (fabs((a) - (b)) < 1e-9)

/* ---- helpers ---------------------------------------------------------- */

static TSPInstance load_square(void)
{
    TSPInstance inst;
    ga_status_t s = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    if (s != GA_OK) { printf("FATAL: load square_4.tsp failed\n"); exit(1); }
    s = tsp_instance_build_distance_matrix(&inst);
    if (s != GA_OK) { printf("FATAL: dist matrix failed\n"); exit(1); }
    return inst;
}

/* ---- I-01: Single generated tour passes tour_validate ----------------- */
static void test_i01(void)
{
    Tour t;
    tour_alloc(&t, 4);

    RNGState rng;
    rng_seed(&rng, 42);

    int rc = tour_random_init(&t, 4, &rng);
    ASSERT("I-01 tour_random_init returns GA_OK", rc == GA_OK);
    ASSERT("I-01 random tour validates", tour_validate(&t, 4));

    tour_free(&t);
}

/* ---- I-02: Population size matches ------------------------------------ */
static void test_i02(void)
{
    int N = 10;
    int n = 4;
    Tour *pop = malloc((size_t)N * sizeof(Tour));
    RNGState *states = malloc((size_t)N * sizeof(RNGState));

    for (int i = 0; i < N; i++) {
        tour_alloc(&pop[i], n);
        rng_seed(&states[i], 100 + (uint64_t)i);
    }

    int rc = population_init(pop, N, n, states);
    ASSERT("I-02 population_init returns GA_OK", rc == GA_OK);

    /* All tours allocated and valid */
    int all_valid = 1;
    for (int i = 0; i < N; i++) {
        if (!tour_validate(&pop[i], n)) { all_valid = 0; break; }
    }
    ASSERT("I-02 all N tours valid", all_valid);

    for (int i = 0; i < N; i++) tour_free(&pop[i]);
    free(pop);
    free(states);
}

/* ---- I-03: All individuals valid -------------------------------------- */
static void test_i03(void)
{
    int N = 50;
    int n = 4;
    Tour *pop = malloc((size_t)N * sizeof(Tour));
    RNGState *states = malloc((size_t)N * sizeof(RNGState));

    for (int i = 0; i < N; i++) {
        tour_alloc(&pop[i], n);
        rng_seed(&states[i], 200 + (uint64_t)i);
    }

    population_init(pop, N, n, states);

    int all_valid = 1;
    for (int i = 0; i < N; i++) {
        if (!tour_validate(&pop[i], n)) { all_valid = 0; break; }
    }
    ASSERT("I-03 all 50 individuals valid", all_valid);

    for (int i = 0; i < N; i++) tour_free(&pop[i]);
    free(pop);
    free(states);
}

/* ---- I-04: Same seed array reproduces exact same population ----------- */
static void test_i04(void)
{
    int N = 10;
    int n = 4;

    Tour *pop_a = malloc((size_t)N * sizeof(Tour));
    Tour *pop_b = malloc((size_t)N * sizeof(Tour));
    RNGState *states_a = malloc((size_t)N * sizeof(RNGState));
    RNGState *states_b = malloc((size_t)N * sizeof(RNGState));

    for (int i = 0; i < N; i++) {
        tour_alloc(&pop_a[i], n);
        tour_alloc(&pop_b[i], n);
        rng_seed(&states_a[i], 300 + (uint64_t)i);
        rng_seed(&states_b[i], 300 + (uint64_t)i);
    }

    population_init(pop_a, N, n, states_a);
    population_init(pop_b, N, n, states_b);

    int all_match = 1;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < n; k++) {
            if (pop_a[i].cities[k] != pop_b[i].cities[k]) {
                all_match = 0;
                break;
            }
        }
        if (!all_match) break;
    }
    ASSERT("I-04 same seeds => same population", all_match);

    for (int i = 0; i < N; i++) { tour_free(&pop_a[i]); tour_free(&pop_b[i]); }
    free(pop_a); free(pop_b);
    free(states_a); free(states_b);
}

/* ---- I-05: Walking skeleton integration (init + eval + best) ---------- */
static void test_i05(void)
{
    TSPInstance inst = load_square();

    int N = 20;
    int n = inst.n;
    Tour *pop = malloc((size_t)N * sizeof(Tour));
    RNGState *states = malloc((size_t)N * sizeof(RNGState));

    for (int i = 0; i < N; i++) {
        tour_alloc(&pop[i], n);
        rng_seed(&states[i], 42 + (uint64_t)i);
    }

    int rc1 = population_init(pop, N, n, states);
    ASSERT("I-05 population_init OK", rc1 == GA_OK);

    int rc2 = population_evaluate(pop, N, &inst);
    ASSERT("I-05 population_evaluate OK", rc2 == GA_OK);

    /* Find best */
    double best_len = pop[0].length;
    for (int i = 1; i < N; i++) {
        if (pop[i].length < best_len) best_len = pop[i].length;
    }
    ASSERT("I-05 best length > 0", best_len > 0.0);
    ASSERT("I-05 best length finite", !isinf(best_len) && !isnan(best_len));

    printf("    I-05 best length = %.6f\n", best_len);

    for (int i = 0; i < N; i++) tour_free(&pop[i]);
    free(pop);
    free(states);
    tsp_instance_free(&inst);
}

/* ---- I-NULL: NULL safety ---------------------------------------------- */
static void test_null_safety(void)
{
    int rc1 = tour_random_init(NULL, 4, NULL);
    ASSERT("I-NULL tour_random_init(NULL,...) => ERR", rc1 != GA_OK);

    int rc2 = population_init(NULL, 10, 4, NULL);
    ASSERT("I-NULL population_init(NULL,...) => ERR", rc2 != GA_OK);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_initialization\n");

    test_i01();
    test_i02();
    test_i03();
    test_i04();
    test_i05();
    test_null_safety();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
