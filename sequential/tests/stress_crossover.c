/*
 * stress_crossover.c -- Stress test: 100,000 OX1 offspring, all validated
 *
 * Standalone executable.  Usage:
 *   stress_crossover [--n 50] [--trials 100000] [--seed 42]
 *
 * Initializes two random parents of size n, generates <trials> offspring
 * via OX1 (p_c=1.0), and validates every single one.
 * Exits 0 on success.
 */

#include "ga.h"
#include "tour.h"
#include "rng.h"
#include "init.h"
#include "crossover.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int n         = 50;
    int trials    = 100000;
    uint64_t seed = 42;

    /* Simple CLI parsing */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
            trials = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint64_t)atoll(argv[++i]);
    }

    /* Set up two random parents */
    Tour parent_a, parent_b, child;
    tour_alloc(&parent_a, n);
    tour_alloc(&parent_b, n);
    tour_alloc(&child, n);

    RNGState rng_a, rng_b, rng_cross;
    rng_seed(&rng_a, seed);
    rng_seed(&rng_b, seed + 1);
    rng_seed(&rng_cross, seed + 2);

    tour_random_init(&parent_a, n, &rng_a);
    tour_random_init(&parent_b, n, &rng_b);

    int failures = 0;
    for (int t = 0; t < trials; t++) {
        int rc = crossover_ox1(&child, &parent_a, &parent_b, n, &rng_cross);
        if (rc != GA_OK || !tour_validate(&child, n)) {
            failures++;
            printf("FAILURE at trial %d\n", t);
        }
    }

    printf("All %d children valid. %d failures.\n", trials, failures);

    tour_free(&parent_a);
    tour_free(&parent_b);
    tour_free(&child);

    return (failures == 0) ? 0 : 1;
}
