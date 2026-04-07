# Implementation Roadmap — Sequential GA for TSP
### `ga-tsp` · University of Arizona · Applied Mathematics
**Version:** 1.0 · **Status:** Living Document

---

> **This roadmap is the first half of the GPU roadmap.**  
> Every decision in the serial implementation is a down-payment on the CUDA port.  
> No GPU work begins before Phase 10 exits cleanly.

---

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Sprint Cadence](#2-sprint-cadence)
3. [Phase Overview](#3-phase-overview)
4. [Phase 0 — Repository and Build Backbone](#4-phase-0--repository-and-build-backbone)
5. [Phase 1 — Instance Loading and Distance Matrix](#5-phase-1--instance-loading-and-distance-matrix)
6. [Phase 2 — Tour Representation and Validation](#6-phase-2--tour-representation-and-validation)
7. [Phase 3 — Fitness Evaluation](#7-phase-3--fitness-evaluation)
8. [Phase 4 — RNG, Initialization, and Walking Skeleton](#8-phase-4--rng-initialization-and-walking-skeleton)
9. [Phase 5 — Selection](#9-phase-5--selection)
10. [Phase 6 — Crossover](#10-phase-6--crossover)
11. [Phase 7 — Mutation](#11-phase-7--mutation)
12. [Phase 8 — Elitism and Replacement](#12-phase-8--elitism-and-replacement)
13. [Phase 9 — GA Driver and Statistics](#13-phase-9--ga-driver-and-statistics)
14. [Phase 10 — Baseline Benchmarking and Regression Lock](#14-phase-10--baseline-benchmarking-and-regression-lock)
15. [Post-Serial: GPU Transition Order](#15-post-serial-gpu-transition-order)

---

## 1. Guiding Principles

Three principles govern every implementation decision.

**Correctness before speed.**  
No optimization work begins before invariants and operator correctness are established and tested.  An optimized wrong answer is worse than a slow right one.

**Batch-oriented interfaces.**  
Even in the serial version, modules expose population-level operations — `evaluate_population(...)`, not just `evaluate_tour(...)`.  These interfaces map directly onto GPU kernel launches.  The friction of the CUDA port is determined almost entirely by decisions made here.

**Test-driven evolution.**  
Tests are written before implementation.  A phase is not closed by completing code; it is closed by passing its gate.  If a bug is found and fixed, a test capturing that exact bug is committed before the fix.

---

## 2. Sprint Cadence

| Parameter | Value |
|-----------|-------|
| Sprint length | 4–7 working days |
| Sprint deliverable | One coherent subsystem + a passing phase gate |
| Sprint closeout | Run designated tests; record results in `TEST_PLAN.md` |
| Required build | `make clean && make test` from a fresh checkout |

Phases roughly map to one sprint each.  Phase 10 may extend to two sprints depending on profiling and benchmark scope.

---

## 3. Phase Overview

```
Phase  0  │  Build backbone + empty smoke
Phase  1  │  Instance parser + distance matrix
Phase  2  │  Tour representation + validation           ← ASan required from here
Phase  3  │  Fitness evaluation
Phase  4  │  RNG + initialization + WALKING SKELETON    ← first runnable program
Phase  5  │  Selection
Phase  6  │  Crossover
Phase  7  │  Mutation
Phase  8  │  Elitism + replacement
Phase  9  │  GA driver + statistics + smoke             ← Valgrind required
Phase 10  │  Profiling + -O3 baseline + regression lock ← GPU work may begin after this
```

---

## 4. Phase 0 — Repository and Build Backbone

**Goal:** A reproducible C99 project skeleton with a working build system and test harness, but no algorithm code.

### Deliverables

- [ ] Repository structure initialized
- [ ] `Makefile` with targets: `all`, `test`, `clean`, `asan`
- [ ] Directory layout established (see below)
- [ ] Minimal smoke executable compiles and exits cleanly
- [ ] `README.md` skeleton committed

### Directory Layout

```
ga-tsp/
├── include/          # public headers (.h)
├── src/              # implementation files (.c)
├── tests/
│   ├── fixtures/     # .tsp test files
│   └── *.c           # test sources
├── scripts/          # benchmark and profiling helpers
├── docs/             # LaTeX spec, this roadmap, test plan
├── Makefile
└── README.md
```

### Conventions Established

| Convention | Decision |
|------------|----------|
| C standard | C99 (`-std=c99`) |
| Naming | `snake_case` for all identifiers |
| Error handling | Return codes; no global `errno` reliance |
| Memory ownership | Documented in header comment for every allocating function |
| Status codes | `GA_OK = 0`, `GA_ERR_ALLOC`, `GA_ERR_INVALID`, etc. |

### CUDA-Readiness Notes

- Headers are modular; serial and CUDA implementations share the same interface
- No file-scope globals
- Population-oriented function signatures stubbed now — not retrofitted later

### Phase Gate

```bash
make clean && make all    # must produce zero warnings with -Wall -Wextra
make test                 # empty smoke test exits 0
```

---

## 5. Phase 1 — Instance Loading and Distance Matrix

**Goal:** Load a TSP instance from a coordinate file and construct a correct, contiguous distance matrix.

### Deliverables

- [ ] `TSPInstance` struct (`include/instance.h`)
- [ ] Instance parser supporting coordinate-format input
- [ ] Distance matrix builder (Euclidean, row-major flat array)
- [ ] `tsp_instance_free()` — complete memory cleanup
- [ ] `tests/fixtures/square_4.tsp` and `malformed.tsp` created

### Key Design Decisions

```c
typedef struct {
    int      n;           /* number of cities                        */
    double  *coords_x;    /* [n]   x-coordinates                     */
    double  *coords_y;    /* [n]   y-coordinates                     */
    double  *dist;        /* [n*n] row-major distance matrix (flat)  */
} TSPInstance;

/* Lookup macro — inlined for GPU porting compatibility */
#define DIST(inst, i, j)  ((inst)->dist[(i)*(inst)->n + (j)])
```

> **Why flat?**  `double **dist` requires $n+1$ separate allocations, none contiguous.  
> A flat `double *dist` with a macro is a single `cudaMemcpy` away from device memory.

### CUDA-Readiness Notes

- Flat layout → single `cudaMalloc` + `cudaMemcpy` for device mirror
- `DIST` macro is zero-cost abstraction; replace backing pointer for GPU
- Matrix is treated as read-only after construction → candidate for constant memory

### Phase Gate

```bash
make test SUITE=instance       # P-01 through P-05
make test SUITE=distance       # D-01 through D-05
```

All 10 cases must pass.  Leak check:

```bash
valgrind --leak-check=full ./tests/test_instance
# Expected: "0 bytes in 0 blocks"
```

---

## 6. Phase 2 — Tour Representation and Validation

**Goal:** Define candidate tours, enforce memory ownership, and provide a validator used as the correctness oracle for all subsequent phases.

### Deliverables

- [ ] `Tour` struct (`include/tour.h`)
- [ ] `tour_alloc()` / `tour_free()`
- [ ] `tour_copy()` — deep copy, no aliasing
- [ ] `tour_validate()` — $O(n)$ permutation check, returns bool
- [ ] `tests/test_tour_validity.c` written *before* implementation

### Key Design Decisions

```c
typedef struct {
    int    *cities;   /* permutation of {0, ..., n-1}  */
    double  length;   /* precomputed tour length        */
    double  fitness;  /* 1.0 / length                  */
} Tour;
```

The validator is the most important function in the codebase for debugging.  It must be callable in $O(n)$ with no dynamic allocation, so it can be used freely inside debug assertions.

```c
bool tour_validate(const Tour *t, int n);
/* Returns false on: duplicate city, missing city, out-of-range index. */
/* No heap allocation. Safe to call from GPU validation wrapper.       */
```

### CUDA-Readiness Notes

- `tour_validate` is pure — no side effects; maps to a device-side checker kernel
- No embedded function pointers or callbacks in the struct
- `tour_copy` serves as the model for `cudaMemcpy`-based device copy later

### Phase Gate

```bash
make test SUITE=tour_validity   # V-01 through V-07
make asan SUITE=tour_validity   # Must produce zero ASan aborts
```

> **ASan is mandatory from this phase onward.**  Every subsequent phase gate requires `make asan` to pass alongside `make test`.

---

## 7. Phase 3 — Fitness Evaluation

**Goal:** Compute closed-tour length correctly and provide a batch evaluation interface for the entire population.

### Deliverables

- [ ] `tour_evaluate(Tour *t, const TSPInstance *inst)` — single tour
- [ ] `population_evaluate(Tour *pop, int N, const TSPInstance *inst)` — batch
- [ ] `tests/test_fitness.c` written before implementation

### Key Design Decisions

```c
/* Single tour: reads dist matrix, writes t->length and t->fitness. */
void tour_evaluate(Tour *t, const TSPInstance *inst);

/* Batch: calls tour_evaluate for each individual in pop[0..N-1].   */
/* THIS is the function replaced by a CUDA kernel in Phase GPU-1.   */
void population_evaluate(Tour *pop, int N, const TSPInstance *inst);
```

The batch function is explicitly designed as a future kernel target.  It must not allocate heap memory internally — all state lives in the caller-supplied arrays.

### CUDA-Readiness Notes

- `population_evaluate` → CUDA kernel: one thread block per tour, parallel reduction within block
- No global state inside evaluation → pure function; trivially threadsafe
- Distance matrix access pattern: `dist[cities[k]*n + cities[k+1]]` — predictable, cache-friendly even on GPU

### Phase Gate

```bash
make test SUITE=fitness    # F-01 through F-06
make asan SUITE=fitness
```

---

## 8. Phase 4 — RNG, Initialization, and Walking Skeleton

**Goal:** Generate reproducible random tours, seed the initial population, and produce the **first runnable program** — a walking skeleton that initializes and evaluates a population without any evolution.

### Deliverables

- [ ] `RNGState` struct + `rng_seed()`, `rng_next_int()`, `rng_next_double()`
- [ ] `tour_random_init(Tour *t, int n, RNGState *rng)` — Fisher–Yates
- [ ] `population_init(Tour *pop, int N, int n, RNGState *rng)`
- [ ] **Walking skeleton driver** (`src/skeleton_main.c`)
- [ ] `tests/test_rng.c`, `tests/test_initialization.c`

### RNG Design

```c
typedef struct { uint64_t state; } RNGState;

void     rng_seed(RNGState *rng, uint64_t seed);
uint64_t rng_next_int(RNGState *rng);
double   rng_next_double(RNGState *rng);   /* returns [0.0, 1.0) */
```

One `RNGState` per individual.  No shared global generator.  This is a direct `curandState` analog.

### Walking Skeleton

The walking skeleton must be running by end of this phase.  Its purpose is to catch integration failures **before** the evolution operators are built, not after.

```
Walking Skeleton behavior:
  1. Parse CLI: instance file, population size, seed
  2. Load TSPInstance
  3. Initialize population (random tours)
  4. Evaluate population (tour lengths)
  5. Find and print best tour length
  6. Free all memory
  7. Exit 0
```

**There is no crossover, selection, or mutation.**  The skeleton exists purely to validate that Phases 1–4 integrate correctly.  This is test I-05 in the test plan.

### CUDA-Readiness Notes

- Per-individual `RNGState` → `curandState` per thread; no shared state to serialize
- `population_init` batch interface → future device-side initialization kernel
- Walking skeleton validates data flow from parser → device memory → evaluation → host readback

### Phase Gate

```bash
make test SUITE=rng              # R-01 through R-04
make test SUITE=initialization   # I-01 through I-05
make asan SUITE=rng
make asan SUITE=initialization

# Walking skeleton integration check
./skeleton --instance tests/fixtures/smoke_20.tsp --pop 50 --seed 42
# Must exit 0 and print a finite best-tour length
```

---

## 9. Phase 5 — Selection

**Goal:** Implement parent selection; prefer tournament selection for its parallel scaling properties.

### Deliverables

- [ ] `tournament_select(Tour *pop, int N, int k, RNGState *rng)` → returns index
- [ ] `select_parent_pair(...)` → selects two distinct parents
- [ ] `tests/test_selection.c`

### Design Note: Why Tournament, Not Roulette

| Property | Tournament ($k=2$) | Roulette Wheel |
|----------|--------------------|----------------|
| Per-individual work | $O(k)$ | $O(\log N)$ prefix scan |
| Inter-thread communication | None | Global reduction required |
| GPU scaling | Embarrassingly parallel | Bottleneck at prefix sum |
| Selection pressure tuning | Adjust $k$ | Adjust fitness scaling |

Tournament selection is chosen because it requires no global information.  Each contest is fully independent and can be assigned to a single GPU thread with zero synchronization overhead.

### Phase Gate

```bash
make test SUITE=selection    # S-01 through S-04
make asan SUITE=selection
```

---

## 10. Phase 6 — Crossover

**Goal:** Implement Order Crossover (OX1) that produces valid permutation offspring.

### Deliverables

- [ ] `crossover_ox1(Tour *child, const Tour *a, const Tour *b, int n, RNGState *rng)`
- [ ] Crossover wrapper honoring probability `p_c`
- [ ] `tests/test_crossover.c`

### OX1 Implementation Notes

The sequential implementation uses a boolean lookup buffer of size $n$ (stack-allocated for small $n$, heap otherwise).  The GPU version will replace this with register-held bitmasks for $n \leq 64$ or shared memory for $n \leq 1024$.

```
OX1 steps:
  1. Draw random segment [l, r] from parent A
  2. Copy A[l..r] directly into child positions [l..r]
  3. Mark those cities as used (lookup buffer)
  4. Traverse parent B from position r+1 (mod n)
  5. Append unused cities in B's order to remaining child slots
```

Key constraint: `child` must pass `tour_validate()` after every call, with no exceptions.

### CUDA-Readiness Notes

- One offspring per thread: inputs are read-only parent arrays + RNG state
- Bitmask membership replaces boolean array → no divergence from cache-miss-dependent branches
- Internal lookup buffer must not be a global — stack or per-thread shared memory

### Phase Gate

```bash
make test SUITE=crossover    # C-01 through C-06
make asan SUITE=crossover

# Stress test: 100,000 crossover calls, all children validated
./tests/stress_crossover --n 50 --trials 100000 --seed 42
# Must report: "All 100000 children valid. 0 failures."
```

---

## 11. Phase 7 — Mutation

**Goal:** Add swap and inversion mutation operators that preserve permutation validity under all probability settings.

### Deliverables

- [ ] `mutate_swap(Tour *t, int n, RNGState *rng)` — $O(1)$
- [ ] `mutate_invert(Tour *t, int n, RNGState *rng)` — $O(n)$
- [ ] Mutation wrapper honoring probability `p_m`
- [ ] `tests/test_mutation.c`

### Design Note: Operator Isolation

Mutation type is a compile-time or configuration-time setting, **not** a runtime branch inside a single function.  Mixed operator branching inside one call creates warp divergence when ported to GPU.  Separate functions, separate kernel launches.

### Phase Gate

```bash
make test SUITE=mutation    # M-01 through M-06
make asan SUITE=mutation

# Validate city multiset preservation over 50,000 trials
./tests/stress_mutation --n 30 --trials 50000 --seed 7
# Must report: "0 multiset violations."
```

---

## 12. Phase 8 — Elitism and Replacement

**Goal:** Preserve top individuals across generations and build the next-generation population correctly.

### Deliverables

- [ ] `extract_elites(Tour *pop, int N, int e, Tour *elites)` — partial sort
- [ ] `build_next_generation(Tour *next, Tour *elites, int e, Tour *offspring, int N)`
- [ ] `tests/test_elitism.c`, `tests/test_replacement.c`

### Design Note: Sorting Strategy

The serial baseline uses introselect (C stdlib `qsort` on the fitness array) for simplicity and correctness.  This is documented with a `/* GPU: replace with bitonic Top-K kernel */` comment so the replacement site is unambiguous.

### Phase Gate

```bash
make test SUITE=elitism        # E-01 through E-05
make test SUITE=replacement    # Re-01 through Re-04
make asan SUITE=elitism
make asan SUITE=replacement
```

---

## 13. Phase 9 — GA Driver and Statistics

**Goal:** Integrate all modules into a working sequential GA that runs to completion on real TSP instances.

### Deliverables

- [ ] `GAConfig` struct (population size, elite count, tournament $k$, $p_c$, $p_m$, generations, seed)
- [ ] `GAStats` struct (per-generation best, mean, worst fitness)
- [ ] `ga_run(GAConfig *cfg, TSPInstance *inst, GAStats *stats)` — main loop
- [ ] CSV statistics export
- [ ] `tests/test_ga_smoke.c`

### Driver Structure

```c
int ga_run(const GAConfig *cfg, const TSPInstance *inst, GAStats *stats) {
    /* 1. Initialize population                  */
    /* 2. Evaluate population                    */
    /* 3. For g in [1..G]:                       */
    /*      a. Extract elites                    */
    /*      b. Generate offspring (select+cross) */
    /*      c. Mutate offspring                  */
    /*      d. Evaluate offspring                */
    /*      e. Build next generation             */
    /*      f. Log statistics                    */
    /* 4. Return best tour                       */
    /* 5. Free all intermediate memory           */
}
```

The driver has no knowledge of crossover or mutation internals.  It operates through the module interfaces established in Phases 5–8.  This is the seam where future CUDA kernel launches replace serial loops.

### Phase Gate

```bash
make test SUITE=ga_smoke    # G-01 through G-05

# Full Valgrind run — mandatory
valgrind --leak-check=full \
  ./ga --instance tests/fixtures/smoke_20.tsp \
       --pop 100 --gen 200 --seed 42
# Expected: "0 bytes in 0 blocks" leaked
```

---

## 14. Phase 10 — Baseline Benchmarking and Regression Lock

**Goal:** Establish the maximum single-threaded CPU performance and freeze the serial baseline as the correctness reference for the GPU implementation.

> **Critical.**  Comparing a GPU implementation against an unoptimized CPU build (`-O0`, no SIMD) produces artificially inflated speedup numbers.  This phase exists to prevent that.  The baseline must be genuinely competitive before the comparison is scientifically meaningful.

### Deliverables

- [ ] Benchmark scripts (`scripts/benchmark.sh`)
- [ ] Fixed-seed experiment set (seeds × instances matrix)
- [ ] Baseline result archive (`results/serial_baseline.csv`)
- [ ] Regression tolerances documented in `TEST_PLAN.md` (Reg-01)
- [ ] Profiling report identifying top-3 hotspots

### Optimization Build

```bash
# Must pass all regression tests with this build
CFLAGS="-O3 -march=native -ffast-math -DNDEBUG" make all
```

> **`-ffast-math` caveat.**  Verify that enabling `-ffast-math` does not change regression outputs by more than `1e-6` relative to the `-O0` build.  If it does, remove it and document why.

### Profiling

```bash
# Option A: gprof
gcc -O2 -pg -o ga_profile ...
./ga_profile --instance benchmark_100.tsp --pop 500 --gen 1000 --seed 42
gprof ga_profile gmon.out | head -40

# Option B: perf (Linux, UA HPC cluster)
perf record -g ./ga --instance benchmark_100.tsp --pop 500 --gen 1000 --seed 42
perf report
```

**Expected hotspot:** `population_evaluate` and the inner distance-lookup loop should account for > 60% of runtime.  If they do not, investigate before proceeding to GPU work.

### Phase Gate

```bash
# All prior suites must still pass under -O3 build
make test BUILD=release    # Reg-01 through Reg-05

# Regression comparison
./scripts/compare_baseline.sh results/serial_baseline.csv results/current.csv
# Must report: all distances within 0.1% tolerance
```

---

## 15. Post-Serial: GPU Transition Order

Once Phase 10 exits cleanly, the GPU port proceeds in this order.  Each step replaces exactly one serial module with a CUDA kernel; everything else remains unchanged.

| GPU Phase | Target | Serial Seam Replaced | Expected Speedup |
|-----------|--------|----------------------|-----------------|
| GPU-1 | Population evaluation | `population_evaluate(...)` | High (embarrassingly parallel) |
| GPU-2 | Offspring generation | `select + crossover` loop | High |
| GPU-3 | Mutation | `mutate_*` loop | Moderate |
| GPU-4 | Elite selection | `extract_elites` | Low–moderate (memory bound) |
| GPU-5 | Optional: 2-opt local search | Post-evaluation improvement | Very high ($O(n^2)$ per individual) |

At each GPU phase, the regression suite from Phase 10 is the correctness oracle.  The GPU implementation is correct if and only if it reproduces the serial baseline outputs within the documented tolerances on the fixed-seed experiment set.

---

*Roadmap · `ga-tsp` · Joel Maldonado · University of Arizona · v1.0*