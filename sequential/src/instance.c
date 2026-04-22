/*
 * instance.c -- TSP instance loader, distance matrix builder, cleanup
 *
 * Phase 1 implementation.
 *
 * Parsing strategy : fgets + sscanf (line-buffered, clean error recovery).
 * Allocation       : single-pass after reading DIMENSION header.
 * Matrix layout    : flat row-major n*n double array (one malloc).
 */

#include "instance.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- static helpers ---------------------------------------------------- */

/*
 * Zero-initialise the struct so that tsp_instance_free() is always safe,
 * even if the caller bails out before any allocation succeeds.
 */
static void tsp_instance_init(TSPInstance *inst)
{
    inst->n        = 0;
    inst->coords_x = NULL;
    inst->coords_y = NULL;
    inst->dist     = NULL;
}

/* ---- tsp_instance_free ------------------------------------------------- */

void tsp_instance_free(TSPInstance *inst)
{
    if (inst == NULL) return;
    free(inst->coords_x);  inst->coords_x = NULL;
    free(inst->coords_y);  inst->coords_y = NULL;
    free(inst->dist);      inst->dist     = NULL;
    inst->n = 0;
}

/* ---- tsp_instance_load ------------------------------------------------- */

ga_status_t tsp_instance_load(const char *path, TSPInstance *out)
{
    tsp_instance_init(out);

    FILE *fp = fopen(path, "r");
    if (fp == NULL) return GA_ERR_IO;

    char line[512];
    int  dimension_found    = 0;
    int  coord_section_found = 0;
    int  n = 0;

    /* -- scan headers for DIMENSION, stop at NODE_COORD_SECTION or EOF -- */
    while (fgets(line, sizeof(line), fp) != NULL) {
        /* check for DIMENSION */
        if (strstr(line, "DIMENSION") != NULL) {
            /* accept "DIMENSION: 4" or "DIMENSION : 4" */
            const char *colon = strchr(line, ':');
            if (colon != NULL) {
                n = atoi(colon + 1);
                if (n > 0) dimension_found = 1;
            }
        }
        /* check for NODE_COORD_SECTION */
        if (strstr(line, "NODE_COORD_SECTION") != NULL) {
            coord_section_found = 1;
            break;
        }
    }

    if (!dimension_found || !coord_section_found) {
        fclose(fp);
        return GA_ERR_IO;
    }

    /* -- allocate coordinate arrays ------------------------------------- */
    out->coords_x = malloc((size_t)n * sizeof(double));
    out->coords_y = malloc((size_t)n * sizeof(double));
    if (out->coords_x == NULL || out->coords_y == NULL) {
        fclose(fp);
        tsp_instance_free(out);
        return GA_ERR_ALLOC;
    }

    /* -- parse coordinate lines ----------------------------------------- */
    for (int k = 0; k < n; k++) {
        if (fgets(line, sizeof(line), fp) == NULL) {
            fclose(fp);
            tsp_instance_free(out);
            return GA_ERR_IO;
        }
        /* stop early on EOF marker */
        if (strstr(line, "EOF") != NULL) {
            fclose(fp);
            tsp_instance_free(out);
            return GA_ERR_IO;
        }

        int    id;
        double x, y;
        if (sscanf(line, "%d %lf %lf", &id, &x, &y) != 3) {
            fclose(fp);
            tsp_instance_free(out);
            return GA_ERR_IO;
        }
        out->coords_x[id - 1] = x;
        out->coords_y[id - 1] = y;
    }

    fclose(fp);
    out->n = n;
    return GA_OK;
}

/* ---- tsp_instance_build_distance_matrix -------------------------------- */

ga_status_t tsp_instance_build_distance_matrix(TSPInstance *inst)
{
    int n = inst->n;
    inst->dist = malloc((size_t)n * n * sizeof(double));
    if (inst->dist == NULL) return GA_ERR_ALLOC;

    for (int i = 0; i < n; i++) {
        DIST(inst, i, i) = 0.0;
        for (int j = i + 1; j < n; j++) {
            double dx = inst->coords_x[i] - inst->coords_x[j];
            double dy = inst->coords_y[i] - inst->coords_y[j];
            double d  = sqrt(dx * dx + dy * dy);
            DIST(inst, i, j) = d;
            DIST(inst, j, i) = d;
        }
    }
    return GA_OK;
}
