/*
 * instance.h -- TSP instance: struct, loader, distance matrix, cleanup
 *
 * Phase 1 public API.
 *
 * Ownership:
 *   TSPInstance owns all three heap pointers (coords_x, coords_y, dist).
 *   tsp_instance_free() must be called exactly once on any struct touched
 *   by tsp_instance_load(), whether or not the load succeeded.
 */

#ifndef TSP_INSTANCE_H
#define TSP_INSTANCE_H

#include "ga.h"

/* ---- TSPInstance --------------------------------------------------------
 *
 * Fields:
 *   n         Number of cities.  >= 1 when valid; 0 when uninitialised/freed.
 *   coords_x  Heap array of n x-coordinates (0-indexed).  NULL if not yet set.
 *   coords_y  Heap array of n y-coordinates (0-indexed).  NULL if not yet set.
 *   dist      Flat row-major n*n distance matrix.  NULL until built.
 *
 * Invariants (once dist is built):
 *   dist[i*n + i] == 0.0           for all i
 *   dist[i*n + j] == dist[j*n + i] for all i, j  (symmetry)
 *   dist[i*n + j] >= 0.0           for all i, j
 * --------------------------------------------------------------------- */
typedef struct {
    int     n;
    double *coords_x;
    double *coords_y;
    double *dist;
} TSPInstance;

/* Row-major access macro.  i and j must be in [0, inst->n). */
#define DIST(inst, i, j)  ((inst)->dist[(i) * (inst)->n + (j)])

/* ---- function declarations -------------------------------------------- */

/*
 * tsp_instance_load -- parse a TSPLIB EUC_2D coordinate file.
 *
 * On success (GA_OK): out->n >= 1, coords allocated, dist is NULL.
 * On failure:         out is safe to pass to tsp_instance_free().
 */
ga_status_t tsp_instance_load(const char *path, TSPInstance *out);

/*
 * tsp_instance_build_distance_matrix -- fill the n*n Euclidean distance
 * matrix from the already-loaded coordinates.
 *
 * Precondition: inst was successfully loaded (n >= 1, coords valid).
 * On success (GA_OK): inst->dist is a contiguous n*n double array.
 * On failure:         inst->dist is NULL.
 */
ga_status_t tsp_instance_build_distance_matrix(TSPInstance *inst);

/*
 * tsp_instance_free -- release all heap memory owned by the instance.
 *
 * Safe to call on NULL, on a zeroed struct, and on a partially-initialised
 * struct.  Nullifies all pointers and resets n to 0.
 */
void tsp_instance_free(TSPInstance *inst);

#endif /* TSP_INSTANCE_H */
