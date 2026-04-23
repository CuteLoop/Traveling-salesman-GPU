/*
 * test_mutation.c -- Phase 7 tests: swap and inversion mutation
 *
 * Written BEFORE implementation (TDD).
 */

#include "ga.h"
#include "tour.h"
#include "rng.h"
#include "mutation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(msg, expr) do { \
    tests_run++; \
    if (expr) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while (0)

/* ---- helpers ---------------------------------------------------------- */

/* Build an identity tour: cities = {0, 1, ..., n-1} */
static void make_identity(Tour *t, int n)
{
    tour_alloc(t, n);
    for (int k = 0; k < n; k++) t->cities[k] = k;
    t->length  = 100.0;
    t->fitness = 0.01;
}

/* Compare two int arrays for equality */
static int arrays_equal(const int *a, const int *b, int n)
{
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

/* Simple insertion sort for small int arrays (used for multiset check) */
static void sort_ints(int *arr, int n)
{
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/* ---- M-01: Mutated tour is a valid permutation ------------------------ */
static void test_m01(void)
{
    int n = 10;
    RNGState rng;
    rng_seed(&rng, 42);

    Tour t;
    make_identity(&t, n);

    int rc1 = mutate_swap(&t, n, &rng);
    ASSERT("M-01a swap returns GA_OK", rc1 == GA_OK);
    ASSERT("M-01a swap result is valid permutation", tour_validate(&t, n));

    /* Reset and test invert */
    for (int k = 0; k < n; k++) t.cities[k] = k;
    int rc2 = mutate_invert(&t, n, &rng);
    ASSERT("M-01b invert returns GA_OK", rc2 == GA_OK);
    ASSERT("M-01b invert result is valid permutation", tour_validate(&t, n));

    tour_free(&t);
}

/* ---- M-02: City multiset unchanged after mutation --------------------- */
static void test_m02(void)
{
    int n = 15;
    RNGState rng;
    rng_seed(&rng, 99);

    Tour t;
    make_identity(&t, n);

    /* Save sorted before */
    int before[15];
    memcpy(before, t.cities, (size_t)n * sizeof(int));
    sort_ints(before, n);

    /* Swap mutation */
    mutate_swap(&t, n, &rng);
    int after_swap[15];
    memcpy(after_swap, t.cities, (size_t)n * sizeof(int));
    sort_ints(after_swap, n);
    ASSERT("M-02a swap preserves city multiset", arrays_equal(before, after_swap, n));

    /* Reset and invert */
    for (int k = 0; k < n; k++) t.cities[k] = k;
    mutate_invert(&t, n, &rng);
    int after_inv[15];
    memcpy(after_inv, t.cities, (size_t)n * sizeof(int));
    sort_ints(after_inv, n);
    ASSERT("M-02b invert preserves city multiset", arrays_equal(before, after_inv, n));

    tour_free(&t);
}

/* ---- M-03: p_m = 0.0 → tour unchanged -------------------------------- */
static void test_m03(void)
{
    int n = 10;
    RNGState rng;
    rng_seed(&rng, 7);

    Tour t;
    make_identity(&t, n);

    int orig[10];
    memcpy(orig, t.cities, (size_t)n * sizeof(int));

    /* apply_mutation_swap with p_m=0.0: must not mutate */
    for (int trial = 0; trial < 100; trial++) {
        apply_mutation_swap(&t, n, 0.0, &rng);
    }
    ASSERT("M-03a p_m=0.0 swap leaves tour unchanged",
           arrays_equal(orig, t.cities, n));

    /* Reset and test invert */
    for (int k = 0; k < n; k++) t.cities[k] = k;
    for (int trial = 0; trial < 100; trial++) {
        apply_mutation_invert(&t, n, 0.0, &rng);
    }
    ASSERT("M-03b p_m=0.0 invert leaves tour unchanged",
           arrays_equal(orig, t.cities, n));

    tour_free(&t);
}

/* ---- M-04: p_m = 1.0 → tour changes in at least 1/100 trials --------- */
static void test_m04(void)
{
    int n = 10;
    RNGState rng;
    rng_seed(&rng, 55);

    Tour t;
    make_identity(&t, n);

    int orig[10];
    memcpy(orig, t.cities, (size_t)n * sizeof(int));

    /* Swap: at least one of 100 trials must change */
    int changed_swap = 0;
    for (int trial = 0; trial < 100; trial++) {
        for (int k = 0; k < n; k++) t.cities[k] = k;
        apply_mutation_swap(&t, n, 1.0, &rng);
        if (!arrays_equal(orig, t.cities, n)) { changed_swap = 1; break; }
    }
    ASSERT("M-04a p_m=1.0 swap changes tour in ≤100 trials", changed_swap);

    /* Invert: at least one of 100 trials must change */
    int changed_inv = 0;
    for (int trial = 0; trial < 100; trial++) {
        for (int k = 0; k < n; k++) t.cities[k] = k;
        apply_mutation_invert(&t, n, 1.0, &rng);
        if (!arrays_equal(orig, t.cities, n)) { changed_inv = 1; break; }
    }
    ASSERT("M-04b p_m=1.0 invert changes tour in ≤100 trials", changed_inv);

    tour_free(&t);
}

/* ---- M-05: Inversion reverses the target segment correctly ------------ */
static void test_m05(void)
{
    int n = 8;
    RNGState rng;
    rng_seed(&rng, 123);

    Tour t;
    make_identity(&t, n);
    /* cities = {0, 1, 2, 3, 4, 5, 6, 7} */

    /*
     * We can't control which indices RNG picks, so we run multiple
     * trials and verify each result is a valid permutation with a
     * contiguous reversed segment.
     */
    int all_valid = 1;
    for (int trial = 0; trial < 50; trial++) {
        for (int k = 0; k < n; k++) t.cities[k] = k;
        mutate_invert(&t, n, &rng);
        if (!tour_validate(&t, n)) { all_valid = 0; break; }

        /* Find the reversed segment: count how many positions differ */
        int first_diff = -1, last_diff = -1;
        for (int k = 0; k < n; k++) {
            if (t.cities[k] != k) {
                if (first_diff == -1) first_diff = k;
                last_diff = k;
            }
        }
        /* If no difference, l == r (single element), that's fine */
        if (first_diff == -1) continue;

        /* Verify the differing segment is a reversal of the identity */
        int segment_ok = 1;
        for (int k = first_diff; k <= last_diff; k++) {
            if (t.cities[k] != (first_diff + last_diff - k)) {
                segment_ok = 0;
                break;
            }
        }
        if (!segment_ok) { all_valid = 0; break; }
    }
    ASSERT("M-05 inversion reverses a contiguous segment correctly", all_valid);

    tour_free(&t);
}

/* ---- M-06: Swap mutation changes exactly two positions ---------------- */
static void test_m06(void)
{
    int n = 20;
    RNGState rng;
    rng_seed(&rng, 77);

    Tour t;
    make_identity(&t, n);

    int all_hamming2 = 1;
    for (int trial = 0; trial < 100; trial++) {
        for (int k = 0; k < n; k++) t.cities[k] = k;
        mutate_swap(&t, n, &rng);

        /* Count differing positions (Hamming distance) */
        int hamming = 0;
        for (int k = 0; k < n; k++) {
            if (t.cities[k] != k) hamming++;
        }
        /* Hamming distance must be exactly 2 (or 0 if same index drawn,
           but with n=20 and a good RNG this is vanishingly rare) */
        if (hamming != 0 && hamming != 2) { all_hamming2 = 0; break; }
    }
    ASSERT("M-06 swap changes exactly 0 or 2 positions", all_hamming2);

    tour_free(&t);
}

/* ---- M-07: Mutation safely handles n <= 1 ----------------------------- */
static void test_m07(void)
{
    RNGState rng;
    rng_seed(&rng, 1);

    /* n = 1: single city, nothing to mutate */
    Tour t1;
    tour_alloc(&t1, 1);
    t1.cities[0] = 0;
    t1.length = 0.0;
    t1.fitness = 0.0;

    int rc1 = mutate_swap(&t1, 1, &rng);
    ASSERT("M-07a swap n=1 returns GA_OK or ERR_INVALID gracefully",
           rc1 == GA_OK || rc1 == GA_ERR_INVALID);

    int rc2 = mutate_invert(&t1, 1, &rng);
    ASSERT("M-07b invert n=1 returns GA_OK or ERR_INVALID gracefully",
           rc2 == GA_OK || rc2 == GA_ERR_INVALID);

    tour_free(&t1);

    /* NULL pointer safety */
    int rc3 = mutate_swap(NULL, 5, &rng);
    ASSERT("M-07c swap NULL tour => ERR", rc3 != GA_OK);

    int rc4 = mutate_invert(NULL, 5, &rng);
    ASSERT("M-07d invert NULL tour => ERR", rc4 != GA_OK);
}

/* ---- M-08: length/fitness invalidated after mutation ------------------ */
static void test_m08(void)
{
    int n = 10;
    RNGState rng;
    rng_seed(&rng, 333);

    Tour t;
    make_identity(&t, n);
    t.length  = 42.0;
    t.fitness = 1.0 / 42.0;

    mutate_swap(&t, n, &rng);
    ASSERT("M-08a swap invalidates length to 0.0", t.length == 0.0);
    ASSERT("M-08b swap invalidates fitness to 0.0", t.fitness == 0.0);

    t.length  = 42.0;
    t.fitness = 1.0 / 42.0;
    mutate_invert(&t, n, &rng);
    ASSERT("M-08c invert invalidates length to 0.0", t.length == 0.0);
    ASSERT("M-08d invert invalidates fitness to 0.0", t.fitness == 0.0);

    tour_free(&t);
}

/* ---- main ------------------------------------------------------------- */

int main(void)
{
    printf("test_mutation\n");

    test_m01();
    test_m02();
    test_m03();
    test_m04();
    test_m05();
    test_m06();
    test_m07();
    test_m08();

    printf("  %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
