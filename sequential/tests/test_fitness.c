/*
 * test_fitness.c -- Phase 3 tests: tour evaluation and batch population eval
 *
 * Uses square_4.tsp fixture (unit square: (0,0),(1,0),(1,1),(0,1)).
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "instance.h"
#include "tour.h"
#include "fitness.h"

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

/* ---- Helpers ---------------------------------------------------------- */

static TSPInstance load_square(void)
{
    TSPInstance inst;
    ga_status_t s = tsp_instance_load("tests/fixtures/square_4.tsp", &inst);
    if (s != GA_OK) { printf("FATAL: cannot load square_4.tsp\n"); exit(1); }
    s = tsp_instance_build_distance_matrix(&inst);
    if (s != GA_OK) { printf("FATAL: cannot build dist matrix\n"); exit(1); }
    return inst;
}

static Tour make_tour(int n, const int *cities)
{
    Tour t;
    tour_alloc(&t, n);
    for (int i = 0; i < n; i++) t.cities[i] = cities[i];
    t.length  = 0.0;
    t.fitness = 0.0;
    return t;
}

/* ---- F-01: Identity tour [0,1,2,3] length == 4.0 --------------------- */
static void test_f01(TSPInstance *inst)
{
    int cities[] = {0, 1, 2, 3};
    Tour t = make_tour(4, cities);

    int rc = tour_evaluate(&t, inst);
    ASSERT("F-01 tour_evaluate returns GA_OK", rc == GA_OK);
    ASSERT("F-01 identity tour length == 4.0", APPROX_EQ(t.length, 4.0));

    tour_free(&t);
}

/* ---- F-02: Same tour evaluated twice gives identical results ---------- */
static void test_f02(TSPInstance *inst)
{
    int cities[] = {0, 1, 2, 3};
    Tour t = make_tour(4, cities);

    tour_evaluate(&t, inst);
    double len1 = t.length;
    double fit1 = t.fitness;

    tour_evaluate(&t, inst);
    double len2 = t.length;
    double fit2 = t.fitness;

    ASSERT("F-02 length stable", APPROX_EQ(len1, len2));
    ASSERT("F-02 fitness stable", APPROX_EQ(fit1, fit2));

    tour_free(&t);
}

/* ---- F-03: Rotation [1,2,3,0] has same closed-tour length ------------ */
static void test_f03(TSPInstance *inst)
{
    int c1[] = {0, 1, 2, 3};
    int c2[] = {1, 2, 3, 0};
    Tour t1 = make_tour(4, c1);
    Tour t2 = make_tour(4, c2);

    tour_evaluate(&t1, inst);
    tour_evaluate(&t2, inst);

    ASSERT("F-03 rotation length match", APPROX_EQ(t1.length, t2.length));

    tour_free(&t1);
    tour_free(&t2);
}

/* ---- F-04: population_evaluate matches tour_evaluate entry-wise ------- */
static void test_f04(TSPInstance *inst)
{
    int c0[] = {0, 1, 2, 3};
    int c1[] = {0, 1, 3, 2};   /* diagonal tour */
    int c2[] = {3, 2, 1, 0};   /* reverse */
    Tour pop[3];
    pop[0] = make_tour(4, c0);
    pop[1] = make_tour(4, c1);
    pop[2] = make_tour(4, c2);

    /* individual evals */
    Tour ref[3];
    ref[0] = make_tour(4, c0);
    ref[1] = make_tour(4, c1);
    ref[2] = make_tour(4, c2);
    for (int i = 0; i < 3; i++) tour_evaluate(&ref[i], inst);

    /* batch eval */
    int rc = population_evaluate(pop, 3, inst);
    ASSERT("F-04 population_evaluate returns GA_OK", rc == GA_OK);

    int all_match = 1;
    for (int i = 0; i < 3; i++) {
        if (!APPROX_EQ(pop[i].length, ref[i].length)) all_match = 0;
        if (!APPROX_EQ(pop[i].fitness, ref[i].fitness)) all_match = 0;
    }
    ASSERT("F-04 batch matches individual", all_match);

    for (int i = 0; i < 3; i++) { tour_free(&pop[i]); tour_free(&ref[i]); }
}

/* ---- F-05: Fitness == 1/length; shorter tour has higher fitness ------- */
static void test_f05(TSPInstance *inst)
{
    int c_square[]   = {0, 1, 2, 3};   /* length = 4.0 (perimeter) */
    int c_diagonal[] = {0, 1, 3, 2};   /* goes through diagonal, longer */
    Tour t_sq  = make_tour(4, c_square);
    Tour t_dia = make_tour(4, c_diagonal);

    tour_evaluate(&t_sq, inst);
    tour_evaluate(&t_dia, inst);

    ASSERT("F-05 fitness = 1/length (sq)", APPROX_EQ(t_sq.fitness, 1.0 / t_sq.length));
    ASSERT("F-05 fitness = 1/length (dia)", APPROX_EQ(t_dia.fitness, 1.0 / t_dia.length));
    ASSERT("F-05 shorter tour => higher fitness", t_sq.fitness > t_dia.fitness);

    tour_free(&t_sq);
    tour_free(&t_dia);
}

/* ---- F-06: n=1 city tour, length=0, fitness=0 (no crash) ------------- */
static void test_f06(void)
{
    /* Build a trivial 1-city instance by hand */
    TSPInstance inst;
    inst.n = 1;
    inst.coords_x = malloc(sizeof(double));
    inst.coords_y = malloc(sizeof(double));
    inst.dist     = malloc(sizeof(double));
    inst.coords_x[0] = 0.0;
    inst.coords_y[0] = 0.0;
    inst.dist[0]     = 0.0;

    Tour t;
    tour_alloc(&t, 1);
    t.cities[0] = 0;

    int rc = tour_evaluate(&t, &inst);
    ASSERT("F-06 n=1 returns GA_OK", rc == GA_OK);
    ASSERT("F-06 n=1 length == 0.0", APPROX_EQ(t.length, 0.0));
    ASSERT("F-06 n=1 fitness == 0.0", APPROX_EQ(t.fitness, 0.0));

    tour_free(&t);
    free(inst.coords_x);
    free(inst.coords_y);
    free(inst.dist);
}

/* ---- F-07: Reversed tour [3,2,1,0] matches forward tour length -------- */
static void test_f07(TSPInstance *inst)
{
    int c_fwd[] = {0, 1, 2, 3};
    int c_rev[] = {3, 2, 1, 0};
    Tour t_fwd = make_tour(4, c_fwd);
    Tour t_rev = make_tour(4, c_rev);

    tour_evaluate(&t_fwd, inst);
    tour_evaluate(&t_rev, inst);

    ASSERT("F-07 reversed tour same length", APPROX_EQ(t_fwd.length, t_rev.length));

    tour_free(&t_fwd);
    tour_free(&t_rev);
}

/* ---- F-NULL: NULL pointer safety -------------------------------------- */
static void test_null_safety(TSPInstance *inst)
{
    int rc1 = tour_evaluate(NULL, inst);
    ASSERT("F-NULL tour_evaluate(NULL, inst) => ERR", rc1 != GA_OK);

    Tour t;
    tour_alloc(&t, 4);
    int rc2 = tour_evaluate(&t, NULL);
    ASSERT("F-NULL tour_evaluate(t, NULL) => ERR", rc2 != GA_OK);

    int rc3 = population_evaluate(NULL, 3, inst);
    ASSERT("F-NULL population_evaluate(NULL,...) => ERR", rc3 != GA_OK);

    tour_free(&t);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_fitness\n");

    TSPInstance inst = load_square();

    test_f01(&inst);
    test_f02(&inst);
    test_f03(&inst);
    test_f04(&inst);
    test_f05(&inst);
    test_f06();
    test_f07(&inst);
    test_null_safety(&inst);

    tsp_instance_free(&inst);

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
