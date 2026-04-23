/*
 * stress_mutation.c -- Phase 7 stress test: 50,000 mutation trials
 *
 * Standalone executable.  Applies mutate_swap and mutate_invert
 * randomly with p_m=1.0 on a tour of n=30 cities for 50,000
 * iterations, validating every result.
 *
 * Usage: stress_mutation [--n N] [--trials T] [--seed S]
 *
 * Exit 0 on success.
 */

#include "tour.h"
#include "rng.h"
#include "mutation.h"
#include "ga.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int n      = 30;
    int trials = 50000;
    uint64_t seed = 7;

    /* Simple CLI parsing */
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--n") == 0)      n      = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--trials") == 0)  trials = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--seed") == 0)    seed   = (uint64_t)atoll(argv[i + 1]);
    }

    RNGState rng;
    rng_seed(&rng, seed);

    /* Initialize a random tour */
    Tour t;
    tour_alloc(&t, n);
    for (int k = 0; k < n; k++) t.cities[k] = k;

    /* Shuffle using Fisher-Yates */
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next_int(&rng) % (uint64_t)(i + 1));
        int tmp = t.cities[i];
        t.cities[i] = t.cities[j];
        t.cities[j] = tmp;
    }
    t.length  = 0.0;
    t.fitness = 0.0;

    int violations = 0;

    for (int trial = 0; trial < trials; trial++) {
        /* Randomly choose swap or invert */
        if (rng_next_int(&rng) % 2 == 0) {
            apply_mutation_swap(&t, n, 1.0, &rng);
        } else {
            apply_mutation_invert(&t, n, 1.0, &rng);
        }

        if (!tour_validate(&t, n)) {
            violations++;
            fprintf(stderr, "VIOLATION at trial %d\n", trial);
        }
    }

    printf("stress_mutation: %d multiset violations across %d mutations.\n",
           violations, trials);

    tour_free(&t);

    return (violations == 0) ? 0 : 1;
}
