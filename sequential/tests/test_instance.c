/*
 * test_instance.c -- Phase 1: TSP instance parser tests
 *
 * Covers: P-01 .. P-05, M-01, M-05
 * TDD: written before src/instance.c
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

/* ---- P-01 .. P-04: valid load ------------------------------------------ */
static void test_load_square(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);

    CHECK(rc == GA_OK,     "P-01: load returns GA_OK on valid file");
    CHECK(inst.n == 4,     "P-02: city count is 4");

    if (rc != GA_OK) { tsp_instance_free(&inst); return; }

    /* P-03: city count matches DIMENSION header */
    CHECK(inst.n == 4,     "P-03: n matches DIMENSION header");

    /* P-04: spot-check coordinates */
    CHECK_NEAR(inst.coords_x[0], 0.0, 1e-9, "P-04a: city 0 x == 0.0");
    CHECK_NEAR(inst.coords_y[0], 0.0, 1e-9, "P-04b: city 0 y == 0.0");
    CHECK_NEAR(inst.coords_x[1], 1.0, 1e-9, "P-04c: city 1 x == 1.0");
    CHECK_NEAR(inst.coords_y[1], 0.0, 1e-9, "P-04d: city 1 y == 0.0");
    CHECK_NEAR(inst.coords_x[2], 1.0, 1e-9, "P-04e: city 2 x == 1.0");
    CHECK_NEAR(inst.coords_y[2], 1.0, 1e-9, "P-04f: city 2 y == 1.0");
    CHECK_NEAR(inst.coords_x[3], 0.0, 1e-9, "P-04g: city 3 x == 0.0");
    CHECK_NEAR(inst.coords_y[3], 1.0, 1e-9, "P-04h: city 3 y == 1.0");

    /* dist should be NULL until build_distance_matrix is called */
    CHECK(inst.dist == NULL, "P-04i: dist is NULL after load");

    tsp_instance_free(&inst);
}

/* ---- P-02 / F-01: malformed coordinate --------------------------------- */
static void test_malformed_bad_coord(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load(
        "tests/fixtures/malformed_bad_coord.tsp", &inst);
    CHECK(rc != GA_OK,     "F-01: malformed coord returns error");
    tsp_instance_free(&inst);
    CHECK(inst.n == 0,     "F-01b: n is 0 after failed load + free");
}

/* ---- F-02: missing NODE_COORD_SECTION ---------------------------------- */
static void test_missing_section(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load(
        "tests/fixtures/malformed_missing_section.tsp", &inst);
    CHECK(rc != GA_OK,     "F-02: missing section returns error");
    tsp_instance_free(&inst);
}

/* ---- P-05: empty / nonexistent file ------------------------------------ */
static void test_nonexistent_file(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("no_such_file.tsp", &inst);
    CHECK(rc != GA_OK,     "P-05: nonexistent file returns error");
    tsp_instance_free(&inst);
}

/* ---- M-01: free nullifies pointers ------------------------------------- */
static void test_free_nullifies(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    CHECK(rc == GA_OK,     "M-01 setup: load succeeded");
    if (rc != GA_OK) return;

    tsp_instance_free(&inst);
    CHECK(inst.coords_x == NULL, "M-01a: coords_x NULL after free");
    CHECK(inst.coords_y == NULL, "M-01b: coords_y NULL after free");
    CHECK(inst.dist     == NULL, "M-01c: dist NULL after free");
    CHECK(inst.n        == 0,    "M-01d: n == 0 after free");
}

/* ---- M-05: free on NULL pointer is safe -------------------------------- */
static void test_free_null_safe(void)
{
    tsp_instance_free(NULL);          /* must not crash */
    tests_run++;
    tests_passed++;
    fprintf(stdout, "  M-05: free(NULL) safe -- PASS\n");
}

/* ---- M-05b: double free is safe ---------------------------------------- */
static void test_double_free_safe(void)
{
    TSPInstance inst;
    ga_status_t rc = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    if (rc != GA_OK) { tsp_instance_free(&inst); return; }

    tsp_instance_free(&inst);
    tsp_instance_free(&inst);         /* second free must not crash */
    tests_run++;
    tests_passed++;
    fprintf(stdout, "  M-05b: double free safe -- PASS\n");
}

/* ---- main -------------------------------------------------------------- */
int main(void)
{
    test_load_square();
    test_malformed_bad_coord();
    test_missing_section();
    test_nonexistent_file();
    test_free_nullifies();
    test_free_null_safe();
    test_double_free_safe();

    printf("\ntest_instance: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
