#ifndef GA_H
#define GA_H

/*
 * ga.h — Global status codes and common definitions
 *
 * All GA modules return ga_status_t.  GA_OK (0) indicates success; all
 * other values indicate an error.  Functions never rely on errno.
 *
 * Ownership rules:
 *   - Any function whose name ends in _alloc() transfers ownership to the
 *     caller.  The caller must eventually call the matching _free().
 *   - Functions not named _alloc() do not transfer ownership.
 */

/* ---------------------------------------------------------------------------
 * Status codes
 * --------------------------------------------------------------------------*/
typedef enum ga_status {
    GA_OK          = 0,  /* success                                   */
    GA_ERR_ALLOC   = 1,  /* memory allocation failure                 */
    GA_ERR_INVALID = 2,  /* invalid argument or data (null ptr, etc.) */
    GA_ERR_IO      = 3   /* file or I/O error                         */
} ga_status_t;

/* Convenience macro: propagate error up the call stack */
#define GA_RETURN_IF_ERR(expr)          \
    do {                                \
        ga_status_t _s = (expr);        \
        if (_s != GA_OK) return _s;     \
    } while (0)

#endif /* GA_H */
