/*
 * test_replacement.c -- Phase 8 tests: generational replacement
 *
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "tour.h"
#include "replacement.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(msg, expr) do { \
    tests_run++; \
    if (expr) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while (0)

#define APPROX_EQ(a, b) (fabs((a) - (b)) < 1e-9)

/* ---- helpers ---------------------------------------------------------- */

static Tour *make_tours(int count, int n, double base_fitness)
{
    Tour *arr = malloc((size_t)count * sizeof(Tour));
    for (int i = 0; i < count; i++) {
        tour_alloc(&arr[i], n);
        for (int k = 0; k < n; k++) arr[i].cities[k] = k;
        arr[i].fitness = base_fitness + (double)i;
        arr[i].length  = 1.0 / arr[i].fitness;
    }
    return arr;
}

static Tour *alloc_tours(int count, int n)
{
    Tour *arr = malloc((size_t)count * sizeof(Tour));
    for (int i = 0; i < count; i++) {
        tour_alloc(&arr[i], n);
    }
    return arr;
}

static void free_tours(Tour *arr, int count)
{
    for (int i = 0; i < count; i++) tour_free(&arr[i]);
    free(arr);
}

/* ---- Re-01: next_pop has N valid tours -------------------------------- */
static void test_re01(void)
{
    int N = 10, n = 4, e = 3;
    int offspring_count = N - e;

    Tour *elites   = make_tours(e, n, 100.0);           /* 100, 101, 102 */
    Tour *offspring = make_tours(offspring_count, n, 1.0); /* 1..7         */
    Tour *next_pop  = alloc_tours(N, n);

    int rc = build_next_generation(next_pop, elites, e,
                                   offspring, offspring_count, N, n);
    ASSERT("Re-01 returns GA_OK", rc == GA_OK);

    /* All N slots should have non-NULL cities */
    int valid = 1;
    for (int i = 0; i < N; i++) {
        if (next_pop[i].cities == NULL) { valid = 0; break; }
    }
    ASSERT("Re-01 all N slots populated", valid);

    free_tours(next_pop, N);
    free_tours(offspring, offspring_count);
    free_tours(elites, e);
}

/* ---- Re-02: all tours pass tour_validate ------------------------------ */
static void test_re02(void)
{
    int N = 8, n = 5, e = 2;
    int offspring_count = N - e;

    Tour *elites   = make_tours(e, n, 50.0);
    Tour *offspring = make_tours(offspring_count, n, 1.0);
    Tour *next_pop  = alloc_tours(N, n);

    build_next_generation(next_pop, elites, e,
                          offspring, offspring_count, N, n);

    int all_valid = 1;
    for (int i = 0; i < N; i++) {
        if (!tour_validate(&next_pop[i], n)) { all_valid = 0; break; }
    }
    ASSERT("Re-02 all tours pass tour_validate", all_valid);

    free_tours(next_pop, N);
    free_tours(offspring, offspring_count);
    free_tours(elites, e);
}

/* ---- Re-03: elites occupy slots [0..e-1] ------------------------------ */
static void test_re03(void)
{
    int N = 10, n = 4, e = 3;
    int offspring_count = N - e;

    Tour *elites   = make_tours(e, n, 100.0);           /* 100, 101, 102 */
    Tour *offspring = make_tours(offspring_count, n, 1.0);
    Tour *next_pop  = alloc_tours(N, n);

    build_next_generation(next_pop, elites, e,
                          offspring, offspring_count, N, n);

    int elites_match = 1;
    for (int i = 0; i < e; i++) {
        if (!APPROX_EQ(next_pop[i].fitness, elites[i].fitness)) {
            elites_match = 0;
            break;
        }
    }
    ASSERT("Re-03 elites in slots [0..e-1]", elites_match);

    free_tours(next_pop, N);
    free_tours(offspring, offspring_count);
    free_tours(elites, e);
}

/* ---- Re-04: no pointer aliasing (deep copies) ------------------------- */
static void test_re04(void)
{
    int N = 6, n = 4, e = 2;
    int offspring_count = N - e;

    Tour *elites   = make_tours(e, n, 50.0);
    Tour *offspring = make_tours(offspring_count, n, 1.0);
    Tour *next_pop  = alloc_tours(N, n);

    build_next_generation(next_pop, elites, e,
                          offspring, offspring_count, N, n);

    /* Verify no pointer aliasing: next_pop cities should not share
       pointers with elites or offspring */
    int no_alias = 1;
    for (int i = 0; i < e; i++) {
        if (next_pop[i].cities == elites[i].cities) { no_alias = 0; break; }
    }
    for (int i = 0; i < offspring_count; i++) {
        if (next_pop[e + i].cities == offspring[i].cities) { no_alias = 0; break; }
    }
    ASSERT("Re-04 no pointer aliasing", no_alias);

    free_tours(next_pop, N);
    free_tours(offspring, offspring_count);
    free_tours(elites, e);
}

/* ---- Re-05: e=0 fills entirely from offspring ------------------------- */
static void test_re05(void)
{
    int N = 5, n = 4;
    int offspring_count = N;

    Tour *offspring = make_tours(offspring_count, n, 10.0);  /* 10..14 */
    Tour *next_pop  = alloc_tours(N, n);

    int rc = build_next_generation(next_pop, NULL, 0,
                                   offspring, offspring_count, N, n);
    ASSERT("Re-05 e=0 returns GA_OK", rc == GA_OK);

    int match = 1;
    for (int i = 0; i < N; i++) {
        if (!APPROX_EQ(next_pop[i].fitness, offspring[i].fitness)) {
            match = 0;
            break;
        }
    }
    ASSERT("Re-05 all slots from offspring", match);

    free_tours(next_pop, N);
    free_tours(offspring, offspring_count);
}

/* ---- Re-06: offspring_count mismatch => ERR --------------------------- */
static void test_re06(void)
{
    int N = 8, n = 4, e = 2;
    int bad_count = N;  /* should be N - e = 6, not N = 8 */

    Tour *elites    = make_tours(e, n, 50.0);
    Tour *offspring = make_tours(bad_count, n, 1.0);
    Tour *next_pop  = alloc_tours(N, n);

    int rc = build_next_generation(next_pop, elites, e,
                                   offspring, bad_count, N, n);
    ASSERT("Re-06 mismatched offspring_count => ERR", rc != GA_OK);

    free_tours(next_pop, N);
    free_tours(offspring, bad_count);
    free_tours(elites, e);
}

/* ---- Re-NULL: NULL / invalid argument safety -------------------------- */
static void test_null_safety(void)
{
    int N = 5, n = 4, e = 2;
    int oc = N - e;
    Tour *elites   = make_tours(e, n, 50.0);
    Tour *offspring = make_tours(oc, n, 1.0);
    Tour *next_pop  = alloc_tours(N, n);

    int rc1 = build_next_generation(NULL, elites, e, offspring, oc, N, n);
    ASSERT("Re-NULL next_pop=NULL => ERR", rc1 != GA_OK);

    int rc2 = build_next_generation(next_pop, elites, e, NULL, oc, N, n);
    ASSERT("Re-NULL offspring=NULL => ERR", rc2 != GA_OK);

    int rc3 = build_next_generation(next_pop, elites, e, offspring, oc, 0, n);
    ASSERT("Re-NULL N=0 => ERR", rc3 != GA_OK);

    free_tours(next_pop, N);
    free_tours(offspring, oc);
    free_tours(elites, e);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_replacement\n");

    test_re01();
    test_re02();
    test_re03();
    test_re04();
    test_re05();
    test_re06();
    test_null_safety();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
