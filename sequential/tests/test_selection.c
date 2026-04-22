/*
 * test_selection.c -- Phase 5 tests: tournament selection & parent pairs
 *
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "tour.h"
#include "rng.h"
#include "selection.h"

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

/* ---- helpers ---------------------------------------------------------- */

/*
 * Build a dummy population of N tours where pop[i].fitness = (i+1).
 * So pop[N-1] is always the fittest.  No actual cities needed for
 * selection — only fitness matters.
 */
static Tour *make_dummy_pop(int N, int n)
{
    Tour *pop = malloc((size_t)N * sizeof(Tour));
    for (int i = 0; i < N; i++) {
        tour_alloc(&pop[i], n);
        /* Fill identity so tour is valid if anyone checks */
        for (int k = 0; k < n; k++) pop[i].cities[k] = k;
        pop[i].length  = 1.0 / (double)(i + 1);
        pop[i].fitness = (double)(i + 1);
    }
    return pop;
}

static void free_pop(Tour *pop, int N)
{
    for (int i = 0; i < N; i++) tour_free(&pop[i]);
    free(pop);
}

/* ---- S-01: Returned index in [0, N) ---------------------------------- */
static void test_s01(void)
{
    int N = 20, n = 4;
    Tour *pop = make_dummy_pop(N, n);
    RNGState rng;
    rng_seed(&rng, 42);

    int all_in_range = 1;
    for (int trial = 0; trial < 1000; trial++) {
        int idx = -1;
        int rc = tournament_select(pop, N, 2, &rng, &idx);
        if (rc != GA_OK || idx < 0 || idx >= N) {
            all_in_range = 0;
            break;
        }
    }
    ASSERT("S-01 index in [0, N) for 1000 trials", all_in_range);

    free_pop(pop, N);
}

/* ---- S-02: Selection does not mutate the population -------------------- */
static void test_s02(void)
{
    int N = 10, n = 4;
    Tour *pop = make_dummy_pop(N, n);
    RNGState rng;
    rng_seed(&rng, 77);

    /* Snapshot fitness values */
    double orig[10];
    for (int i = 0; i < N; i++) orig[i] = pop[i].fitness;

    for (int trial = 0; trial < 500; trial++) {
        int idx;
        tournament_select(pop, N, 3, &rng, &idx);
    }

    int unchanged = 1;
    for (int i = 0; i < N; i++) {
        if (pop[i].fitness != orig[i]) { unchanged = 0; break; }
    }
    ASSERT("S-02 population unchanged after 500 selections", unchanged);

    free_pop(pop, N);
}

/* ---- S-03: k=2, fittest wins majority of 1000 trials ------------------ */
static void test_s03(void)
{
    int N = 10, n = 4;
    Tour *pop = make_dummy_pop(N, n);
    /* pop[9].fitness == 10.0 is the highest */
    RNGState rng;
    rng_seed(&rng, 123);

    int win_count = 0;
    int trials = 1000;
    for (int t = 0; t < trials; t++) {
        int idx;
        tournament_select(pop, N, 2, &rng, &idx);
        if (idx == N - 1) win_count++;
    }

    /* With k=2, the fittest (index 9) should be selected whenever it
     * appears in the tournament.  P(appears) = 1 - (9/10)^2 = 0.19.
     * So ~190 wins expected.  We check for >= 100 as a soft bound. */
    ASSERT("S-03 k=2 fittest wins >= 100/1000", win_count >= 100);

    free_pop(pop, N);
}

/* ---- S-04: k=1 is uniformly distributed (no bias) --------------------- */
static void test_s04(void)
{
    int N = 10, n = 4;
    Tour *pop = make_dummy_pop(N, n);
    RNGState rng;
    rng_seed(&rng, 999);

    int counts[10] = {0};
    int trials = 10000;
    for (int t = 0; t < trials; t++) {
        int idx;
        tournament_select(pop, N, 1, &rng, &idx);
        counts[idx]++;
    }

    /* With k=1, each individual has P=1/N=0.1.  Expected ~1000 each.
     * Allow [500, 1500] as a generous statistical bound. */
    int all_uniform = 1;
    for (int i = 0; i < N; i++) {
        if (counts[i] < 500 || counts[i] > 1500) {
            all_uniform = 0;
            printf("    S-04 counts[%d] = %d (expected ~1000)\n", i, counts[i]);
            break;
        }
    }
    ASSERT("S-04 k=1 uniform distribution", all_uniform);

    free_pop(pop, N);
}

/* ---- S-05: Deterministic tie-breaking (first drawn wins) -------------- */
static void test_s05(void)
{
    /* Create a population where ALL individuals have the same fitness */
    int N = 5, n = 4;
    Tour *pop = make_dummy_pop(N, n);
    for (int i = 0; i < N; i++) {
        pop[i].fitness = 1.0;
        pop[i].length  = 1.0;
    }

    RNGState rng_a, rng_b;
    rng_seed(&rng_a, 55);
    rng_seed(&rng_b, 55);

    /* Same seed must produce identical selections even with ties */
    int all_match = 1;
    for (int trial = 0; trial < 200; trial++) {
        int idx_a, idx_b;
        tournament_select(pop, N, 3, &rng_a, &idx_a);
        tournament_select(pop, N, 3, &rng_b, &idx_b);
        if (idx_a != idx_b) { all_match = 0; break; }
    }
    ASSERT("S-05 tie-breaking deterministic (same seed)", all_match);

    free_pop(pop, N);
}

/* ---- S-06: select_parent_pair returns two distinct indices ------------- */
static void test_s06(void)
{
    int N = 10, n = 4;
    Tour *pop = make_dummy_pop(N, n);
    RNGState rng;
    rng_seed(&rng, 314);

    int all_distinct = 1;
    for (int trial = 0; trial < 1000; trial++) {
        int a = -1, b = -1;
        int rc = select_parent_pair(pop, N, 2, &rng, &a, &b);
        if (rc != GA_OK || a == b || a < 0 || b < 0 || a >= N || b >= N) {
            all_distinct = 0;
            break;
        }
    }
    ASSERT("S-06 parent pair always distinct", all_distinct);

    free_pop(pop, N);
}

/* ---- S-NULL: NULL / invalid argument safety --------------------------- */
static void test_null_safety(void)
{
    int idx;
    int rc1 = tournament_select(NULL, 10, 2, NULL, &idx);
    ASSERT("S-NULL tournament_select(NULL,...) => ERR", rc1 != GA_OK);

    int a, b;
    int rc2 = select_parent_pair(NULL, 10, 2, NULL, &a, &b);
    ASSERT("S-NULL select_parent_pair(NULL,...) => ERR", rc2 != GA_OK);

    /* N < 2 should fail for pair selection */
    Tour pop[1];
    tour_alloc(&pop[0], 4);
    pop[0].fitness = 1.0;
    RNGState rng;
    rng_seed(&rng, 1);
    int rc3 = select_parent_pair(pop, 1, 1, &rng, &a, &b);
    ASSERT("S-NULL pair with N=1 => ERR", rc3 != GA_OK);
    tour_free(&pop[0]);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_selection\n");

    test_s01();
    test_s02();
    test_s03();
    test_s04();
    test_s05();
    test_s06();
    test_null_safety();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
