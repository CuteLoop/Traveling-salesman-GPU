/*
 * test_elitism.c -- Phase 8 tests: elite extraction
 *
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "tour.h"
#include "elitism.h"

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

/*
 * Build a population of N tours with unique fitnesses.
 * pop[i].fitness = (i + 1) * 1.0, so pop[N-1] is the fittest.
 * Cities are identity permutations (valid tours).
 */
static Tour *make_pop(int N, int n)
{
    Tour *pop = malloc((size_t)N * sizeof(Tour));
    for (int i = 0; i < N; i++) {
        tour_alloc(&pop[i], n);
        for (int k = 0; k < n; k++) pop[i].cities[k] = k;
        pop[i].length  = 1.0 / (double)(i + 1);
        pop[i].fitness = (double)(i + 1);
    }
    return pop;
}

static Tour *alloc_elites(int e, int n)
{
    Tour *elites = malloc((size_t)e * sizeof(Tour));
    for (int i = 0; i < e; i++) {
        tour_alloc(&elites[i], n);
    }
    return elites;
}

static void free_tours(Tour *arr, int count)
{
    for (int i = 0; i < count; i++) tour_free(&arr[i]);
    free(arr);
}

/* ---- E-01: extract_elites returns exactly e individuals --------------- */
static void test_e01(void)
{
    int N = 10, n = 4, e = 3;
    Tour *pop    = make_pop(N, n);
    Tour *elites = alloc_elites(e, n);

    int rc = extract_elites(elites, pop, N, e, n);
    ASSERT("E-01 returns GA_OK", rc == GA_OK);

    /* All e elite slots should have non-NULL cities */
    int count = 0;
    for (int i = 0; i < e; i++) {
        if (elites[i].cities != NULL) count++;
    }
    ASSERT("E-01 exactly e=3 elites populated", count == e);

    free_tours(elites, e);
    free_tours(pop, N);
}

/* ---- E-02: Elites are strictly the top-e by fitness ------------------- */
static void test_e02(void)
{
    int N = 10, n = 4, e = 3;
    Tour *pop    = make_pop(N, n);
    Tour *elites = alloc_elites(e, n);

    extract_elites(elites, pop, N, e, n);

    /* Top 3 fitnesses: 10.0, 9.0, 8.0 (pop[9], pop[8], pop[7]) */
    int has_10 = 0, has_9 = 0, has_8 = 0;
    for (int i = 0; i < e; i++) {
        if (APPROX_EQ(elites[i].fitness, 10.0)) has_10 = 1;
        if (APPROX_EQ(elites[i].fitness, 9.0))  has_9  = 1;
        if (APPROX_EQ(elites[i].fitness, 8.0))  has_8  = 1;
    }
    ASSERT("E-02 top-3 fitnesses present", has_10 && has_9 && has_8);

    free_tours(elites, e);
    free_tours(pop, N);
}

/* ---- E-03: Best individual correctly copied --------------------------- */
static void test_e03(void)
{
    int N = 10, n = 4, e = 1;
    Tour *pop    = make_pop(N, n);
    Tour *elites = alloc_elites(e, n);

    /* Make pop[5] the absolute best */
    pop[5].fitness = 999.0;
    pop[5].length  = 1.0 / 999.0;

    extract_elites(elites, pop, N, e, n);

    ASSERT("E-03 best individual fitness == 999.0",
           APPROX_EQ(elites[0].fitness, 999.0));

    /* Verify deep copy: cities are valid */
    int valid = 1;
    for (int k = 0; k < n; k++) {
        if (elites[0].cities[k] != pop[5].cities[k]) { valid = 0; break; }
    }
    ASSERT("E-03 best individual cities deep-copied", valid);

    free_tours(elites, e);
    free_tours(pop, N);
}

/* ---- E-04: e=0 runs safely ------------------------------------------- */
static void test_e04(void)
{
    int N = 5, n = 4;
    Tour *pop = make_pop(N, n);

    /* No elites array needed for e=0, but pass a dummy */
    int rc = extract_elites(NULL, pop, N, 0, n);
    ASSERT("E-04 e=0 returns GA_OK", rc == GA_OK);

    free_tours(pop, N);
}

/* ---- E-05: e=N deep-copies entire population -------------------------- */
static void test_e05(void)
{
    int N = 5, n = 4;
    Tour *pop    = make_pop(N, n);
    Tour *elites = alloc_elites(N, n);

    int rc = extract_elites(elites, pop, N, N, n);
    ASSERT("E-05 e=N returns GA_OK", rc == GA_OK);

    /* All fitnesses should appear */
    int all_present = 1;
    for (int want = 1; want <= N; want++) {
        int found = 0;
        for (int i = 0; i < N; i++) {
            if (APPROX_EQ(elites[i].fitness, (double)want)) { found = 1; break; }
        }
        if (!found) { all_present = 0; break; }
    }
    ASSERT("E-05 all N individuals present in elites", all_present);

    free_tours(elites, N);
    free_tours(pop, N);
}

/* ---- E-06: Equal fitness ties resolve by lower original index --------- */
static void test_e06(void)
{
    int N = 5, n = 4, e = 2;
    Tour *pop = make_pop(N, n);

    /* Make pop[1] and pop[3] both have the highest fitness */
    pop[1].fitness = 100.0;
    pop[1].length  = 0.01;
    pop[3].fitness = 100.0;
    pop[3].length  = 0.01;

    /* Set unique city patterns to distinguish them */
    pop[1].cities[0] = 1;  /* pop[1] starts with city 1 */
    pop[3].cities[0] = 3;  /* pop[3] starts with city 3 */

    Tour *elites = alloc_elites(e, n);
    extract_elites(elites, pop, N, e, n);

    /* Both elites should have fitness 100.0 */
    ASSERT("E-06 elite[0] fitness == 100.0", APPROX_EQ(elites[0].fitness, 100.0));
    ASSERT("E-06 elite[1] fitness == 100.0", APPROX_EQ(elites[1].fitness, 100.0));

    /* Lower original index (1) should come first */
    ASSERT("E-06 tie-break: lower index first (elite[0] from pop[1])",
           elites[0].cities[0] == 1);
    ASSERT("E-06 tie-break: higher index second (elite[1] from pop[3])",
           elites[1].cities[0] == 3);

    free_tours(elites, e);
    free_tours(pop, N);
}

/* ---- E-NULL: NULL / invalid argument safety --------------------------- */
static void test_null_safety(void)
{
    int N = 5, n = 4, e = 2;
    Tour *pop = make_pop(N, n);
    Tour *elites = alloc_elites(e, n);

    int rc1 = extract_elites(elites, NULL, N, e, n);
    ASSERT("E-NULL pop=NULL => ERR", rc1 != GA_OK);

    int rc2 = extract_elites(elites, pop, N, -1, n);
    ASSERT("E-NULL e=-1 => ERR", rc2 != GA_OK);

    int rc3 = extract_elites(elites, pop, N, N + 1, n);
    ASSERT("E-NULL e>N => ERR", rc3 != GA_OK);

    int rc4 = extract_elites(elites, pop, 0, e, n);
    ASSERT("E-NULL N=0 => ERR", rc4 != GA_OK);

    free_tours(elites, e);
    free_tours(pop, N);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_elitism\n");

    test_e01();
    test_e02();
    test_e03();
    test_e04();
    test_e05();
    test_e06();
    test_null_safety();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
