/*
 * test_smoke.c — Phase 0 smoke test
 *
 * Verifies only that the test harness itself compiles and links against
 * the ga.h header.  Exits 0 on success.  No algorithm logic is tested here.
 */

#include <stdio.h>
#include "ga.h"

int main(void)
{
    /* Confirm status code definitions are reachable */
    ga_status_t s = GA_OK;
    if (s != 0) {
        fprintf(stderr, "FAIL: GA_OK != 0\n");
        return 1;
    }

    if (GA_ERR_ALLOC   == GA_OK ||
        GA_ERR_INVALID == GA_OK ||
        GA_ERR_IO      == GA_OK) {
        fprintf(stderr, "FAIL: error codes collide with GA_OK\n");
        return 1;
    }

    printf("smoke: PASS\n");
    return 0;
}
