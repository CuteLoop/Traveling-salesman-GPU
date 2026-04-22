/*
 * test_rng.c -- Phase 4 tests: RNG determinism and isolation
 *
 * Written BEFORE implementation (TDD).
 */

#include "rng.h"

#include <stdio.h>
#include <math.h>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(msg, expr) do { \
    tests_run++; \
    if (expr) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while (0)

/* ---- R-01: Same seed produces identical integer stream (1000 draws) --- */
static void test_r01(void)
{
    RNGState a, b;
    rng_seed(&a, 42);
    rng_seed(&b, 42);

    int all_match = 1;
    for (int i = 0; i < 1000; i++) {
        uint64_t va = rng_next_int(&a);
        uint64_t vb = rng_next_int(&b);
        if (va != vb) { all_match = 0; break; }
    }
    ASSERT("R-01 same seed => same integer stream (1000)", all_match);
}

/* ---- R-02: Same seed produces identical double stream ----------------- */
static void test_r02(void)
{
    RNGState a, b;
    rng_seed(&a, 42);
    rng_seed(&b, 42);

    int all_match = 1;
    for (int i = 0; i < 1000; i++) {
        double da = rng_next_double(&a);
        double db = rng_next_double(&b);
        if (fabs(da - db) > 1e-15) { all_match = 0; break; }
    }
    ASSERT("R-02 same seed => same double stream (1000)", all_match);
}

/* ---- R-03: Different seeds diverge quickly ---------------------------- */
static void test_r03(void)
{
    RNGState a, b;
    rng_seed(&a, 42);
    rng_seed(&b, 99);

    int differ_count = 0;
    for (int i = 0; i < 10; i++) {
        if (rng_next_int(&a) != rng_next_int(&b)) differ_count++;
    }
    ASSERT("R-03 different seeds diverge (>= 9/10 differ)", differ_count >= 9);
}

/* ---- R-04: RNG state is fully contained in struct --------------------- */
static void test_r04(void)
{
    RNGState a, b;
    rng_seed(&a, 123);

    /* Advance a by 500 draws */
    for (int i = 0; i < 500; i++) rng_next_int(&a);

    /* Copy raw bytes of a into b */
    b = a;

    /* Both must produce identical sequences from here */
    int all_match = 1;
    for (int i = 0; i < 500; i++) {
        if (rng_next_int(&a) != rng_next_int(&b)) { all_match = 0; break; }
    }
    ASSERT("R-04 struct copy preserves state", all_match);
}

/* ---- R-05: rng_next_double always in [0.0, 1.0) ---------------------- */
static void test_r05(void)
{
    RNGState rng;
    rng_seed(&rng, 7);

    int in_range = 1;
    for (int i = 0; i < 100000; i++) {
        double d = rng_next_double(&rng);
        if (d < 0.0 || d >= 1.0) { in_range = 0; break; }
    }
    ASSERT("R-05 100k doubles in [0, 1)", in_range);
}

/* ---- R-06: Seed 0 works (no degenerate zero-state) -------------------- */
static void test_r06(void)
{
    RNGState rng;
    rng_seed(&rng, 0);

    /* Must produce non-zero output (xorshift requires non-zero state) */
    uint64_t v = rng_next_int(&rng);
    ASSERT("R-06 seed 0 produces non-zero output", v != 0);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_rng\n");

    test_r01();
    test_r02();
    test_r03();
    test_r04();
    test_r05();
    test_r06();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
