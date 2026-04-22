/*
 * test_distance_matrix.c -- Phase 1: distance matrix tests
 *
 * Covers: D-01 .. D-05
 * TDD: written before tsp_instance_build_distance_matrix implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ga.h"
#include "instance.h"

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

#define CHECK_NEAR(a, b, tol, msg) CHECK(fabs((a)-(b)) < (tol), msg)

/* ---- helpers ------------------------------------------------------------ */
static TSPInstance load_square(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    if (rc != GA_OK) {
        fprintf(stderr, "FATAL: could not load square_4.tsp\n");
        exit(2);
    }
    rc = tsp_instance_build_distance_matrix(&inst);
    if (rc != GA_OK) {
        fprintf(stderr, "FATAL: could not build distance matrix\n");
        tsp_instance_free(&inst);
        exit(2);
    }
    return inst;
}

/* ---- D-01: diagonal is zero -------------------------------------------- */
static void test_diagonal_zero(void)
{
    TSPInstance inst = load_square();
    for (int i = 0; i < inst.n; i++)
        CHECK(DIST(&inst, i, i) == 0.0, "D-01: diagonal is zero");
    tsp_instance_free(&inst);
}

/* ---- D-02: symmetry ---------------------------------------------------- */
static void test_symmetry(void)
{
    TSPInstance inst = load_square();
    for (int i = 0; i < inst.n; i++)
        for (int j = 0; j < inst.n; j++)
            CHECK_NEAR(DIST(&inst, i, j), DIST(&inst, j, i),
                       1e-12, "D-02: symmetric");
    tsp_instance_free(&inst);
}

/* ---- D-03: square edges == 1.0 ----------------------------------------- */
static void test_edge_lengths(void)
{
    TSPInstance inst = load_square();
    CHECK_NEAR(DIST(&inst, 0, 1), 1.0, 1e-9, "D-03a: dist(0,1)==1");
    CHECK_NEAR(DIST(&inst, 1, 2), 1.0, 1e-9, "D-03b: dist(1,2)==1");
    CHECK_NEAR(DIST(&inst, 2, 3), 1.0, 1e-9, "D-03c: dist(2,3)==1");
    CHECK_NEAR(DIST(&inst, 3, 0), 1.0, 1e-9, "D-03d: dist(3,0)==1");
    tsp_instance_free(&inst);
}

/* ---- D-04: square diagonals == sqrt(2) --------------------------------- */
static void test_diagonal_lengths(void)
{
    TSPInstance inst = load_square();
    CHECK_NEAR(DIST(&inst, 0, 2), sqrt(2.0), 1e-9,
               "D-04a: dist(0,2)==sqrt(2)");
    CHECK_NEAR(DIST(&inst, 1, 3), sqrt(2.0), 1e-9,
               "D-04b: dist(1,3)==sqrt(2)");
    tsp_instance_free(&inst);
}

/* ---- D-05: rebuild produces identical values ---------------------------- */
static void test_rebuild_identical(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    if (rc != GA_OK) { tsp_instance_free(&inst); return; }

    tsp_instance_build_distance_matrix(&inst);

    /* save first build */
    int n = inst.n;
    double *first = malloc((size_t)n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) first[i] = inst.dist[i];

    /* free and rebuild */
    free(inst.dist);
    inst.dist = NULL;
    tsp_instance_build_distance_matrix(&inst);

    int match = 1;
    for (int i = 0; i < n * n; i++) {
        if (fabs(inst.dist[i] - first[i]) > 1e-15) { match = 0; break; }
    }
    CHECK(match, "D-05: rebuild produces identical values");

    free(first);
    tsp_instance_free(&inst);
}

/* ---- D-05b: triangle inequality ---------------------------------------- */
static void test_triangle_inequality(void)
{
    TSPInstance inst = load_square();
    for (int i = 0; i < inst.n; i++)
        for (int j = 0; j < inst.n; j++)
            for (int k = 0; k < inst.n; k++)
                CHECK(DIST(&inst, i, k) <=
                      DIST(&inst, i, j) + DIST(&inst, j, k) + 1e-12,
                      "D-05b: triangle inequality");
    tsp_instance_free(&inst);
}

/* ---- D-matrix: build returns GA_OK ------------------------------------- */
static void test_build_returns_ok(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    CHECK(rc == GA_OK, "D-build setup");
    if (rc != GA_OK) { tsp_instance_free(&inst); return; }

    rc = tsp_instance_build_distance_matrix(&inst);
    CHECK(rc == GA_OK, "D-build: returns GA_OK");
    CHECK(inst.dist != NULL, "D-build: dist is non-NULL after build");

    tsp_instance_free(&inst);
}

/* ---- main -------------------------------------------------------------- */
int main(void)
{
    test_build_returns_ok();
    test_diagonal_zero();
    test_symmetry();
    test_edge_lengths();
    test_diagonal_lengths();
    test_rebuild_identical();
    test_triangle_inequality();

    printf("\ntest_distance_matrix: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
