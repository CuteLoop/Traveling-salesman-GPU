/*
 * ga_stats.c -- GAStats lifecycle: alloc, free, CSV export
 *
 * Phase 9 implementation.
 */

#include "ga_driver.h"

#include <stdio.h>
#include <stdlib.h>

/* ---- ga_stats_alloc --------------------------------------------------- */

int ga_stats_alloc(GAStats *stats, int count)
{
    if (stats == NULL || count <= 0) return GA_ERR_INVALID;

    stats->generations_logged = 0;
    stats->best  = calloc((size_t)count, sizeof(double));
    stats->mean  = calloc((size_t)count, sizeof(double));
    stats->worst = calloc((size_t)count, sizeof(double));

    if (!stats->best || !stats->mean || !stats->worst) {
        free(stats->best);  stats->best  = NULL;
        free(stats->mean);  stats->mean  = NULL;
        free(stats->worst); stats->worst = NULL;
        return GA_ERR_ALLOC;
    }
    return GA_OK;
}

/* ---- ga_stats_free ---------------------------------------------------- */

void ga_stats_free(GAStats *stats)
{
    if (stats == NULL) return;
    free(stats->best);   stats->best  = NULL;
    free(stats->mean);   stats->mean  = NULL;
    free(stats->worst);  stats->worst = NULL;
    stats->generations_logged = 0;
}

/* ---- ga_stats_write_csv ----------------------------------------------- */

int ga_stats_write_csv(const GAStats *stats, const char *path)
{
    if (stats == NULL || path == NULL) return GA_ERR_INVALID;

    FILE *f = fopen(path, "w");
    if (f == NULL) return GA_ERR_IO;

    fprintf(f, "generation,best,mean,worst\n");
    for (int g = 0; g < stats->generations_logged; g++) {
        fprintf(f, "%d,%.6f,%.6f,%.6f\n",
                g, stats->best[g], stats->mean[g], stats->worst[g]);
    }

    fclose(f);
    return GA_OK;
}
