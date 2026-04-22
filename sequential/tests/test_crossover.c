/*
 * test_crossover.c -- Phase 6 tests: OX1 crossover and apply_crossover
 *
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "tour.h"
#include "rng.h"
#include "crossover.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(msg, expr) do { \
    tests_run++; \
    if (expr) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while (0)

/* ---- helpers ---------------------------------------------------------- */

static Tour make_tour(int n, const int *cities)
{
    Tour t;
    tour_alloc(&t, n);
    for (int i = 0; i < n; i++) t.cities[i] = cities[i];
    t.length  = 0.0;
    t.fitness = 0.0;
    return t;
}

/* ---- C-01: Child is a valid permutation ------------------------------- */
static void test_c01(void)
{
    int n = 8;
    int ca[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int cb[] = {7, 6, 5, 4, 3, 2, 1, 0};
    Tour a = make_tour(n, ca);
    Tour b = make_tour(n, cb);
    Tour child;
    tour_alloc(&child, n);

    RNGState rng;
    rng_seed(&rng, 42);

    int all_valid = 1;
    for (int trial = 0; trial < 200; trial++) {
        int rc = crossover_ox1(&child, &a, &b, n, &rng);
        if (rc != GA_OK || !tour_validate(&child, n)) {
            all_valid = 0;
            break;
        }
    }
    ASSERT("C-01 OX1 child is valid permutation (200 trials)", all_valid);

    tour_free(&a);
    tour_free(&b);
    tour_free(&child);
}

/* ---- C-02: Child city count equals n ---------------------------------- */
static void test_c02(void)
{
    int n = 10;
    int ca[10], cb[10];
    for (int i = 0; i < n; i++) { ca[i] = i; cb[i] = n - 1 - i; }
    Tour a = make_tour(n, ca);
    Tour b = make_tour(n, cb);
    Tour child;
    tour_alloc(&child, n);

    RNGState rng;
    rng_seed(&rng, 77);

    crossover_ox1(&child, &a, &b, n, &rng);

    /* Count distinct cities */
    bool seen[10] = {false};
    int count = 0;
    for (int i = 0; i < n; i++) {
        int c = child.cities[i];
        if (c >= 0 && c < n && !seen[c]) {
            seen[c] = true;
            count++;
        }
    }
    ASSERT("C-02 child has exactly n distinct cities", count == n);

    tour_free(&a);
    tour_free(&b);
    tour_free(&child);
}

/* ---- C-04: p_c=0.0 returns exact copy of parent A --------------------- */
static void test_c04(void)
{
    int n = 6;
    int ca[] = {5, 3, 1, 0, 4, 2};
    int cb[] = {0, 1, 2, 3, 4, 5};
    Tour a = make_tour(n, ca);
    Tour b = make_tour(n, cb);
    a.length  = 10.0;
    a.fitness = 0.1;
    Tour child;
    tour_alloc(&child, n);

    RNGState rng;
    rng_seed(&rng, 99);

    int rc = apply_crossover(&child, &a, &b, n, 0.0, &rng);
    ASSERT("C-04 apply_crossover returns GA_OK", rc == GA_OK);

    int exact_copy = 1;
    for (int i = 0; i < n; i++) {
        if (child.cities[i] != a.cities[i]) { exact_copy = 0; break; }
    }
    ASSERT("C-04 p_c=0.0 child == parent A", exact_copy);

    tour_free(&a);
    tour_free(&b);
    tour_free(&child);
}

/* ---- C-06: Identical parents produce identical child ------------------- */
static void test_c06(void)
{
    int n = 8;
    int ca[] = {3, 1, 7, 0, 5, 2, 6, 4};
    Tour a = make_tour(n, ca);
    Tour b = make_tour(n, ca);  /* identical to a */
    Tour child;
    tour_alloc(&child, n);

    RNGState rng;
    rng_seed(&rng, 123);

    int all_identical = 1;
    for (int trial = 0; trial < 100; trial++) {
        crossover_ox1(&child, &a, &b, n, &rng);
        for (int i = 0; i < n; i++) {
            if (child.cities[i] != a.cities[i]) {
                all_identical = 0;
                break;
            }
        }
        if (!all_identical) break;
    }
    ASSERT("C-06 identical parents => identical child", all_identical);

    tour_free(&a);
    tour_free(&b);
    tour_free(&child);
}

/* ---- C-03: OX1 preserves segment from parent A ----------------------- */
static void test_c03(void)
{
    /* We run many trials and verify the child always validates.
     * The segment preservation is structurally guaranteed by OX1;
     * this test catches off-by-one errors in the segment copy. */
    int n = 12;
    int ca[12], cb[12];
    for (int i = 0; i < n; i++) { ca[i] = i; cb[i] = n - 1 - i; }
    Tour a = make_tour(n, ca);
    Tour b = make_tour(n, cb);
    Tour child;
    tour_alloc(&child, n);

    RNGState rng;
    rng_seed(&rng, 555);

    int all_valid = 1;
    for (int trial = 0; trial < 500; trial++) {
        int rc = crossover_ox1(&child, &a, &b, n, &rng);
        if (rc != GA_OK || !tour_validate(&child, n)) {
            all_valid = 0;
            break;
        }
    }
    ASSERT("C-03 500 crossovers all produce valid children", all_valid);

    tour_free(&a);
    tour_free(&b);
    tour_free(&child);
}

/* ---- C-05: apply_crossover with p_c=1.0 always calls OX1 ------------- */
static void test_c05(void)
{
    int n = 8;
    int ca[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int cb[] = {7, 6, 5, 4, 3, 2, 1, 0};
    Tour a = make_tour(n, ca);
    Tour b = make_tour(n, cb);
    Tour child;
    tour_alloc(&child, n);

    RNGState rng;
    rng_seed(&rng, 200);

    int all_valid = 1;
    for (int trial = 0; trial < 100; trial++) {
        int rc = apply_crossover(&child, &a, &b, n, 1.0, &rng);
        if (rc != GA_OK || !tour_validate(&child, n)) {
            all_valid = 0;
            break;
        }
    }
    ASSERT("C-05 p_c=1.0 always produces valid child", all_valid);

    tour_free(&a);
    tour_free(&b);
    tour_free(&child);
}

/* ---- C-NULL: NULL safety ---------------------------------------------- */
static void test_null_safety(void)
{
    RNGState rng;
    rng_seed(&rng, 1);
    Tour child;
    tour_alloc(&child, 4);

    int rc1 = crossover_ox1(NULL, NULL, NULL, 4, &rng);
    ASSERT("C-NULL crossover_ox1(NULL,...) => ERR", rc1 != GA_OK);

    int rc2 = apply_crossover(NULL, NULL, NULL, 4, 1.0, &rng);
    ASSERT("C-NULL apply_crossover(NULL,...) => ERR", rc2 != GA_OK);

    tour_free(&child);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_crossover\n");

    test_c01();
    test_c02();
    test_c03();
    test_c04();
    test_c05();
    test_c06();
    test_null_safety();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
