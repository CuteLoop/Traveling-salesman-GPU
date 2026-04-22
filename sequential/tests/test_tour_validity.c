/*
 * test_tour_validity.c -- Phase 2: tour representation & validation tests
 *
 * Covers: V-01 .. V-08
 * TDD: written before src/tour.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ga.h"
#include "tour.h"

/* ---- lightweight test helpers ------------------------------------------- */
static int tests_run    = 0;
static int tests_passed = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        tests_run++;                                                \
        if (cond) { tests_passed++; }                               \
        else { fprintf(stderr, "FAIL [%s:%d] %s\n",                \
                       __FILE__, __LINE__, msg); }                  \
    } while (0)

/* ---- V-01: valid permutation [0,1,2,3] passes -------------------------- */
static void test_valid_permutation(void)
{
    Tour t;
    tour_alloc(&t, 4);
    t.cities[0] = 0; t.cities[1] = 1; t.cities[2] = 2; t.cities[3] = 3;
    CHECK(tour_validate(&t, 4) == true,  "V-01: valid permutation passes");
    tour_free(&t);
}

/* ---- V-02: duplicate city [0,1,1,3] fails ------------------------------ */
static void test_duplicate_city(void)
{
    Tour t;
    tour_alloc(&t, 4);
    t.cities[0] = 0; t.cities[1] = 1; t.cities[2] = 1; t.cities[3] = 3;
    CHECK(tour_validate(&t, 4) == false, "V-02: duplicate city fails");
    tour_free(&t);
}

/* ---- V-03: missing city [0,1,3,3] fails -------------------------------- */
static void test_missing_city(void)
{
    Tour t;
    tour_alloc(&t, 4);
    t.cities[0] = 0; t.cities[1] = 1; t.cities[2] = 3; t.cities[3] = 3;
    CHECK(tour_validate(&t, 4) == false, "V-03: missing city fails");
    tour_free(&t);
}

/* ---- V-04: out-of-range index [0,1,2,99] fails ------------------------- */
static void test_out_of_range(void)
{
    Tour t;
    tour_alloc(&t, 4);
    t.cities[0] = 0; t.cities[1] = 1; t.cities[2] = 2; t.cities[3] = 99;
    CHECK(tour_validate(&t, 4) == false, "V-04: out-of-range index fails");
    tour_free(&t);
}

/* ---- V-05: negative index [0,-1,2,3] fails ----------------------------- */
static void test_negative_index(void)
{
    Tour t;
    tour_alloc(&t, 4);
    t.cities[0] = 0; t.cities[1] = -1; t.cities[2] = 2; t.cities[3] = 3;
    CHECK(tour_validate(&t, 4) == false, "V-05: negative index fails");
    tour_free(&t);
}

/* ---- V-06: deep copy preserves values ---------------------------------- */
static void test_copy_values(void)
{
    Tour src, dst;
    tour_alloc(&src, 4);
    src.cities[0] = 3; src.cities[1] = 2; src.cities[2] = 1; src.cities[3] = 0;
    src.length  = 4.0;
    src.fitness = 0.25;

    tour_alloc(&dst, 4);
    tour_copy(&dst, &src, 4);

    int match = 1;
    for (int i = 0; i < 4; i++) {
        if (dst.cities[i] != src.cities[i]) { match = 0; break; }
    }
    CHECK(match,                     "V-06a: city values preserved");
    CHECK(dst.length  == src.length, "V-06b: length preserved");
    CHECK(dst.fitness == src.fitness,"V-06c: fitness preserved");

    tour_free(&src);
    tour_free(&dst);
}

/* ---- V-07: deep copy does not alias ------------------------------------ */
static void test_copy_no_alias(void)
{
    Tour src, dst;
    tour_alloc(&src, 4);
    src.cities[0] = 0; src.cities[1] = 1; src.cities[2] = 2; src.cities[3] = 3;

    tour_alloc(&dst, 4);
    tour_copy(&dst, &src, 4);

    CHECK(dst.cities != src.cities, "V-07: deep copy does not alias");

    tour_free(&src);
    tour_free(&dst);
}

/* ---- V-08: trivial size-1 tour [0] passes ------------------------------ */
static void test_trivial_tour(void)
{
    Tour t;
    tour_alloc(&t, 1);
    t.cities[0] = 0;
    CHECK(tour_validate(&t, 1) == true, "V-08: trivial size-1 tour passes");
    tour_free(&t);
}

/* ---- edge: NULL tour fails validate ------------------------------------ */
static void test_null_tour(void)
{
    CHECK(tour_validate(NULL, 4) == false, "V-null: NULL tour fails");
}

/* ---- edge: tour_free NULL is safe -------------------------------------- */
static void test_free_null(void)
{
    tour_free(NULL);   /* must not crash */
    tests_run++;
    tests_passed++;
    fprintf(stdout, "  tour_free(NULL) safe -- PASS\n");
}

/* ---- edge: double free is safe ----------------------------------------- */
static void test_double_free(void)
{
    Tour t;
    tour_alloc(&t, 4);
    t.cities[0] = 0; t.cities[1] = 1; t.cities[2] = 2; t.cities[3] = 3;
    tour_free(&t);
    tour_free(&t);     /* second free must not crash */
    tests_run++;
    tests_passed++;
    fprintf(stdout, "  tour double free safe -- PASS\n");
}

/* ---- main -------------------------------------------------------------- */
int main(void)
{
    test_valid_permutation();
    test_duplicate_city();
    test_missing_city();
    test_out_of_range();
    test_negative_index();
    test_copy_values();
    test_copy_no_alias();
    test_trivial_tour();
    test_null_tour();
    test_free_null();
    test_double_free();

    printf("\ntest_tour_validity: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
