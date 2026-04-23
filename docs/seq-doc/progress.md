# ga-tsp — Implementation Progress

> Tracks deliverable completion against the roadmap defined in
> [`roadmap.md`](roadmap.md).  
> Updated at the close of each phase.

---

## Phase Status Summary

| Phase | Subsystem | Status | Completed |
|-------|-----------|--------|-----------|
| **0** | Repository & build backbone | ✅ Complete | 2026-04-10 |
| **1** | Instance loading & distance matrix | ✅ Complete | 2026-04-15 |
| **2** | Tour representation & validation | ✅ Complete | 2026-04-15 |
| **3** | Fitness evaluation | ✅ Complete | 2026-04-15 |
| **4** | RNG, initialization & walking skeleton | ✅ Complete | 2026-04-15 |
| **5** | Selection | ✅ Complete | 2026-04-15 |
| **6** | Crossover | ✅ Complete | 2026-04-15 |
| 7 | Mutation | ✅ Complete (backfilled) | 2026-04-15 |
| 8 | Elitism & replacement | ✅ Complete | 2026-04-15 |
| 9 | GA driver & statistics | ✅ Complete | 2026-04-15 |
| **10** | Profiling & regression lock | ✅ Complete | 2026-04-15 |

---

## Phase 0 — Repository and Build Backbone

**Goal:** A reproducible C99 project skeleton with a working build system and
test harness, but no algorithm code.

### Deliverables

- [x] Repository structure initialized
- [x] `Makefile` with targets: `all`, `test`, `clean`, `asan`
- [x] Directory layout established
- [x] Minimal smoke executable compiles and exits cleanly (`src/main.c`)
- [x] `README.md` skeleton committed

### Files Created

| File | Purpose |
|------|---------|
| `sequential/include/ga.h` | Status codes (`GA_OK`, `GA_ERR_*`) and `GA_RETURN_IF_ERR` macro |
| `sequential/src/main.c` | Smoke executable — prints `ga-tsp: build OK`, exits 0 |
| `sequential/tests/test_smoke.c` | Smoke test — verifies status codes compile and are distinct |
| `sequential/Makefile` | `all`, `test`, `clean`, `asan` targets; zero-warning `-Wall -Wextra` |
| `sequential/README.md` | Project overview, quick-start, conventions table |
| `sequential/include/` | Public headers directory (empty until Phase 1) |
| `sequential/src/` | Implementation sources directory |
| `sequential/tests/fixtures/` | `.tsp` test input files (populated from Phase 1) |
| `sequential/scripts/` | Benchmark helpers (populated from Phase 10) |
| `sequential/docs/` | Module-level documentation |

### Conventions Locked

| Convention | Decision |
|------------|----------|
| C standard | C99 (`-std=c99`) |
| Naming | `snake_case` for all identifiers |
| Error handling | Return codes (`ga_status_t`); no global `errno` reliance |
| Memory ownership | Documented in header comment for every allocating function |
| Status codes | `GA_OK = 0`, `GA_ERR_ALLOC`, `GA_ERR_INVALID`, `GA_ERR_IO` |

### Phase Gate Result

```
make clean && make all    → zero warnings with -Wall -Wextra -Wpedantic
make test                 → smoke: PASS  (exit 0)
```

---

## Phase 1 — Instance Loading and Distance Matrix

**Goal:** Parse TSPLIB EUC_2D coordinate files into a contiguous `TSPInstance`
struct and build a flat row-major Euclidean distance matrix.

### Deliverables

- [x] `include/instance.h` — `TSPInstance` struct, `DIST` macro, function declarations
- [x] `src/instance.c` — `tsp_instance_load`, `tsp_instance_build_distance_matrix`, `tsp_instance_free`
- [x] `tests/test_instance.c` — 23 checks: P-01..P-05, F-01, F-02, M-01, M-05
- [x] `tests/test_distance_matrix.c` — 94 checks: D-01..D-05, triangle inequality
- [x] `tests/fixtures/square_4.tsp` — 4-city unit square (hand-computable)
- [x] `tests/fixtures/malformed_bad_coord.tsp` — malformed coordinate line
- [x] `tests/fixtures/malformed_missing_section.tsp` — missing NODE_COORD_SECTION

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Matrix layout | Flat `double *dist` (row-major) | Single malloc, contiguous for GPU copy |
| Distance type | `double` | Natural for `sqrt()`; precision over memory in Phase 1 |
| Symmetry storage | Full n×n | Simplicity; no conditional in index macro |
| Parsing strategy | `fgets` + `sscanf` | Clean error recovery, line-level granularity |
| Allocation | Single-pass via DIMENSION header | No `realloc`, no two-pass |

### Phase Gate Result

```
make clean && make test  →  test_instance: 23/23 passed
                             test_distance_matrix: 94/94 passed
```

---

## Phase 2 — Tour Representation and Validation

**Goal:** Define a `Tour` struct with allocation, cleanup, deep-copy, and an
O(n) permutation validator using no heap allocation.

### Deliverables

- [x] `include/tour.h` — `Tour` struct, function declarations
- [x] `src/tour.c` — `tour_alloc`, `tour_free`, `tour_copy`, `tour_validate`
- [x] `tests/test_tour_validity.c` — 13 checks: V-01..V-08, NULL safety, double-free safety

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation algorithm | C99 VLA `bool seen[n]` | O(n) time, O(n) stack, zero heap |
| Deep copy | `memcpy` on pre-allocated dst | No hidden allocation; caller owns both |
| Free safety | NULL-safe, double-free-safe | Same pattern as `tsp_instance_free` |

### Phase Gate Result

```
make clean && make test  →  test_tour_validity: 13/13 passed
```

TDD and memory-safety constraints strictly followed for both phases.
Codebase ready for verification via `make test` and `make asan`.

---

## Phase 3 — Fitness Evaluation

**Goal:** Compute closed-tour length correctly with uniform `int` return codes
and provide a batch evaluation interface for the entire population.

### Deliverables

- [x] `include/fitness.h` — `tour_evaluate`, `population_evaluate` (both return `int`)
- [x] `src/fitness.c` — closed-loop length + inverse fitness, NULL safety
- [x] `tests/test_fitness.c` — 17 checks: F-01..F-07, batch vs. individual, NULL safety

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| Return type | `int` (not `void`) | Uniform return-code error handling across entire codebase |
| Fitness formula | `1.0 / length` (0.0 when length == 0) | Simple, invertible; n=1 edge case safe |
| Batch function | Loops over `tour_evaluate` | Future CUDA kernel replacement seam; no heap allocation |
| Distance access | `DIST(inst, cities[k], cities[k+1])` | Cache-friendly sequential access |

### Phase Gate Result

```
make clean && make test  →  test_fitness: 17/17 passed
```

---

## Phase 4 — RNG, Initialization, and Walking Skeleton

**Goal:** Generate reproducible random tours, seed the initial population with
per-individual RNG states, and produce the first runnable program.

### Deliverables

- [x] `include/rng.h` — `RNGState` struct, `rng_seed`, `rng_next_int`, `rng_next_double`
- [x] `src/rng.c` — Xorshift64 implementation, seed-0 remap, 53-bit double
- [x] `include/init.h` — `tour_random_init`, `population_init` (takes `RNGState *rng_states` array)
- [x] `src/init.c` — Fisher-Yates shuffle, per-individual RNG mapping
- [x] `src/skeleton_main.c` — Walking skeleton: parse CLI, load, init, eval, print best, exit 0
- [x] `tests/test_rng.c` — 6 checks: R-01..R-06 (determinism, divergence, range, seed-0)
- [x] `tests/test_initialization.c` — 12 checks: I-01..I-05, NULL safety
- [x] `tests/fixtures/smoke_20.tsp` — 20-city instance for skeleton testing

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| RNG algorithm | Xorshift64 (Marsaglia) | Simple, fast, deterministic; period 2^64-1 |
| RNG state | One `RNGState` per individual | Direct `curandState` analog for GPU port |
| `population_init` signature | `RNGState *rng_states` (array) | No shared mutable state; embarrassingly parallel |
| Shuffle | Fisher-Yates (Knuth) | O(n), uniform distribution, well understood |
| Seed 0 handling | Remap to `0x5DEECE66D` | Xorshift requires non-zero state |
| No `rand()`/`srand()` | Explicit | No hidden global state; reproducible across platforms |

### Walking Skeleton Output

```
$ skeleton --instance tests/fixtures/smoke_20.tsp --pop 50 --seed 42
Walking Skeleton Complete. Best length: 147.197789
  (individual 45 of 50, seed = 42, n = 20)
```

### Phase Gate Result

```
make clean && make test  →  test_rng: 6/6 passed
                             test_initialization: 12/12 passed
                             skeleton: exits 0, finite best length
```

All evaluation functions use `int` return codes (not `void`).
RNG layer is entirely self-contained (no `<stdlib.h>` random).
`population_init` takes an array of `RNGState`s — CUDA `curandState` ready.
Codebase ready for `make test` and walking skeleton verification.

---

## Phase 5 — Selection

**Goal:** Implement tournament selection with within-tournament sampling without
replacement and deterministic tie-breaking.  No global synchronization.

### Deliverables

- [x] `include/selection.h` — `tournament_select`, `select_parent_pair` (both return `int`)
- [x] `src/selection.c` — k-tournament without replacement (C99 VLA), deterministic tie-breaking
- [x] `tests/test_selection.c` — 9 checks: S-01..S-06, NULL safety

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| Tournament sampling | Without replacement (C99 VLA `int drawn[k]`) | Prevents degenerate same-individual tournaments |
| Tie-breaking | First drawn candidate wins (strict `>`) | Deterministic, cross-platform reproducible |
| Output style | `int *out_index` return params, `int` status return | Uniform error handling across codebase |
| Pair selection | Redraw second until distinct | Guarantees two different parents; no infinite loop risk with N >= 2 |
| k clamping | `eff_k = min(k, N)` | Gracefully handles k > N without error |

### CUDA-Readiness Notes

- Tournament selection requires zero global information — each contest is independent
- Maps to one GPU thread per selection with zero synchronization overhead
- Per-individual RNGState used throughout (no shared state)

### Phase Gate Result

```
make clean && make test  →  test_selection: 9/9 passed
```

---

## Phase 6 — Crossover

**Goal:** Implement Order Crossover (OX1) that produces valid permutation
offspring with zero heap allocations.

### Deliverables

- [x] `include/crossover.h` — `crossover_ox1`, `apply_crossover` (both return `int`)
- [x] `src/crossover.c` — OX1 with C99 VLA lookup buffer, `apply_crossover` with `p_c` check
- [x] `tests/test_crossover.c` — 9 checks: C-01..C-06, NULL safety
- [x] `tests/stress_crossover.c` — 100,000 offspring stress test executable

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| Lookup buffer | C99 VLA `bool used[n]` | Zero heap; maps to per-thread shared memory on GPU |
| Segment indices | Draw two, swap if l > r | Uniform segment distribution |
| Fill order | Start at `(r+1) % n` in both child and parent B | Standard OX1 wrap-around |
| No-crossover path | `p_c=0.0` copies parent A verbatim | `apply_crossover` consumes one RNG draw regardless |
| Child state | `length = fitness = 0.0` after crossover | Must be re-evaluated; no stale fitness |

### Stress Test Result

```
$ stress_crossover --n 50 --trials 100000 --seed 42
All 100000 children valid. 0 failures.
```

### Phase Gate Result

```
make clean && make test  →  test_crossover: 9/9 passed
                             stress_crossover: 100000/100000 valid
```

Tournament selection avoids global synchronization and implements within-tournament
sampling without replacement.  OX1 uses a C99 VLA to avoid heap allocations.
The `stress_crossover` executable successfully passed 100,000 permutation validations.
All functions return `int` status codes.  Codebase ready for Phase 7 (Mutation).

---

## Phase 7 — Mutation

**Goal:** Add swap and inversion mutation operators that preserve permutation
validity under all probability settings.  Zero heap allocations — all in-place.

### Deliverables

- [x] `include/mutation.h` — `mutate_swap`, `mutate_invert`, `apply_mutation_swap`, `apply_mutation_invert`
- [x] `src/mutation.c` — in-place swap (O(1)) and inversion (O(n)), probability wrappers
- [x] `tests/test_mutation.c` — 20 checks: M-01..M-08, NULL safety
- [x] `tests/stress_mutation.c` — 50,000-trial stress test executable

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| Operator isolation | Separate functions (`mutate_swap`, `mutate_invert`) | No mixed branching; prevents CUDA warp divergence |
| Memory strategy | Zero heap, zero VLAs | Everything operates directly on `t->cities` |
| Swap index selection | Draw i in [0,n), j in [0,n-1), shift j≥i | Guarantees distinct indices in exactly 2 RNG draws |
| Inversion | Two-pointer while loop on `t->cities[l..r]` | O(n) worst case, simple, no allocation |
| State invalidation | `length = 0.0`, `fitness = 0.0` after every mutation | Forces re-evaluation; no stale fitness propagation |
| Probability gate | `rng_next_double(rng) <= p_m` | Consumed one RNG draw regardless of outcome |
| Edge case n < 2 | Returns `GA_ERR_INVALID` | Nothing meaningful to mutate |

### Stress Test Result

```
$ stress_mutation --n 30 --trials 50000 --seed 7
stress_mutation: 0 multiset violations across 50000 mutations.
```

### Phase Gate Result

```
make clean && make test  →  test_mutation: 20/20 passed
                             stress_mutation: 0 violations across 50000 trials
```

Swap and Inversion are strictly isolated functions for CUDA readiness.
All mutations happen 100% in-place without dynamic memory allocation.
`length`/`fitness` fields are properly invalidated upon mutation.
Phase 7 was backfilled after Phase 8 — no functional dependencies were affected.

---

## Phase 8 — Elitism and Replacement

**Goal:** Preserve top individuals across generations via context-safe
EliteRef sort pattern and build aliasing-free next-generation populations.

### Deliverables

- [x] `include/elitism.h` — `extract_elites` declaration
- [x] `src/elitism.c` — EliteRef struct + qsort, descending fitness, lower-index tie-break, deep copy
- [x] `include/replacement.h` — `build_next_generation` declaration
- [x] `src/replacement.c` — deep-copy elites → slots [0..e-1], offspring → slots [e..N-1]
- [x] `tests/test_elitism.c` — 16 checks: E-01..E-06, NULL/invalid-arg safety
- [x] `tests/test_replacement.c` — 11 checks: Re-01..Re-06, NULL safety

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| Sort context | Local `EliteRef` struct (index+fitness) | Avoids qsort global variable trap in strict C99 |
| Sort order | Descending fitness, ascending index tie-break | Deterministic, reproducible across platforms |
| Copy strategy | Deep copy via `tour_copy` | No pointer aliasing between source and destination |
| Elites placement | `next_pop[0..e-1]` | Predictable layout for debugging and GPU port |
| offspring_count enforcement | Must equal `N - e` exactly | Catches off-by-one bugs at the API boundary |
| Temporary allocation | `malloc` for EliteRef array in `extract_elites` | Freed before return; `GA_ERR_ALLOC` on failure |
| GPU replacement marker | `/* GPU: replace with bitonic Top-K kernel */` | Unambiguous replacement site for CUDA port |

### Phase Gate Result

```
make clean && make test  →  test_elitism: 16/16 passed
                             test_replacement: 11/11 passed
```

EliteRef pattern avoids C99 qsort context pointer limitation.
All generational transitions use deep copies — zero pointer aliasing.
Combined with Phase 7 (Mutation), all evolution operators are now complete.
Codebase ready for Phase 9 (GA Driver & Statistics).

---

## Phase 9 — GA Driver and Statistics

**Goal:** Integrate all modules into a working sequential GA that runs to
completion on real TSP instances with proper statistics logging.

### Deliverables

- [x] `include/ga_driver.h` — `GAConfig`, `GAStats`, `ga_run`, `ga_stats_alloc/free/write_csv`
- [x] `src/ga_stats.c` — stats lifecycle (alloc, free, CSV export with header)
- [x] `src/ga_driver.c` — main loop with double-buffering, O(1) pointer swap
- [x] `src/main.c` — full CLI: `--instance`, `--pop`, `--gen`, `--seed`, `--elites`, `--tk`, `--pc`, `--pm`, `--csv`
- [x] `tests/test_ga_smoke.c` — 11 checks: G-01..G-05 (run OK, finite distance, monotonic best, G+1 entries, CSV rows)

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| Header naming | `ga_driver.h` (not `ga.h`) | Avoids collision with existing status code header |
| Stats metric | Distance (not fitness) | Human-readable; convergence plots in natural units |
| Generation 0 | Logged as stats index 0 | G generations → G+1 log entries; true random baseline |
| Double-buffering | O(1) pointer swap (`pop ↔ next_pop`) | No deep copy for generational transition |
| Memory ownership | `ga_run` allocates/frees all internal buffers | Self-contained; caller owns cfg, inst, stats, out_best |
| Offspring RNG | `rng_states[i]` per offspring | Deterministic, reproducible; maps to per-thread curandState |
| Best tracking | Absolute best across all generations | `out_best` deep-copied only when improved |
| Mutation operator | `apply_mutation_swap` in driver loop | Isolated function; switchable to inversion without driver change |
| CSV format | `generation,best,mean,worst` header + data rows | Standard for plotting; G+2 total lines |

### CLI Verification

```
$ ga-tsp --instance tests/fixtures/smoke_20.tsp --pop 100 --gen 200 --seed 42
ga-tsp: instance=tests/fixtures/smoke_20.tsp  n=20  pop=100  gen=200  seed=42
Best distance: 77.492495
Statistics written to results.csv (201 generations)
```

### Phase Gate Result

```
make clean && make test  →  test_ga_smoke: 11/11 passed
                             All 13 suites pass, zero warnings
```

Generation 0 is logged.  O(1) pointer swapping handles generational transitions.
Stats reflect distance (not fitness).  Memory ownership is self-contained.
All internal buffers freed before `ga_run` returns.
The driver contains no raw GA logic — it strictly orchestrates Phase 1-8 APIs.
Ready for Phase 10 (Baseline Benchmarking and Regression Lock).

---

## Phase 10 — Baseline Benchmarking and Regression Lock

**Goal:** Establish an optimized, immutable single-threaded CPU baseline as the
ground-truth oracle for correctness and performance during the CUDA GPU port.

### Deliverables

- [x] `Makefile` — `BUILD=release` (`-O3 -march=native -DNDEBUG`) and `BUILD=profile` (`-O2 -g -fno-omit-frame-pointer`)
- [x] `scripts/benchmark.sh` — 5-seed benchmark runner (compiles release, runs GA, exports CSV with timing)
- [x] `scripts/compare_baseline.sh` — awk-based regression comparator (0.1% relative tolerance)
- [x] `tests/test_regression.c` — 24 checks: Reg-01..Reg-04 (locked value, monotonicity, multi-seed, determinism)
- [x] `results/serial_baseline.csv` — locked 5-seed baseline (pop=500, gen=1000, `-O3 -march=native`)
- [x] `docs/profiling_notes.md` — HPC profiling guide, `perf`/`gprof` commands, >60% hotspot hypothesis

### Build Mode Summary

| Mode | Flags | Purpose |
|------|-------|---------|
| Debug (default) | `-std=c99 -Wall -Wextra -Wpedantic` | Development, full warnings |
| Release (`BUILD=release`) | `-std=c99 -Wall -Wextra -O3 -march=native -DNDEBUG` | Benchmark baseline |
| Profile (`BUILD=profile`) | `-std=c99 -Wall -Wextra -O2 -g -fno-omit-frame-pointer` | `perf`/`gprof` on HPC |

### Locked Baseline Values (smoke_20.tsp, `-O3 -march=native`)

| Seed | Best Distance | Elapsed (s) |
|------|---------------|-----------|
| 42 | 77.419989 | 0.178 |
| 123 | 76.359059 | 0.174 |
| 999 | 75.776906 | 0.179 |
| 5555 | 77.347991 | 0.220 |
| 9876 | 75.776906 | 0.190 |

### Regression Tests

- **Reg-01:** Locked best distance (seed 42, pop 100, gen 200) = 77.492495 ± 1e-6
- **Reg-02:** Best distance non-increasing across generations (monotonicity)
- **Reg-03:** All 5 seeds produce finite positive distances
- **Reg-04:** Two identical runs produce bit-identical results (determinism)

### Design Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|---------|
| `-ffast-math` | Omitted | Strict IEEE-754 compliance for reproducible regression |
| Regression tolerance | 1e-6 absolute | Tighter than 0.1% script tolerance; catches bit-level drift |
| Profiling deferred | UA HPC Linux | MinGW lacks `perf`/`gprof`; `libasan/libubsan` also unavailable |
| Hotspot hypothesis | `population_evaluate` >60% | Validates GPU-1 targeting `population_evaluate` as first kernel |

### Phase Gate Result

```
make clean && make all                →  zero warnings (debug)
make test                              →  14/14 suites pass (24/24 regression checks)
make clean && make all BUILD=release   →  zero warnings (-O3 -march=native -DNDEBUG)
make test BUILD=release                →  14/14 suites pass (regression value stable)
```

The serial implementation of `ga-tsp` is now formally regression-locked.
All 10 phases are complete. The project is ready for the GPU transition.

---

*All 10 phases complete. Serial era of `ga-tsp` concluded 2026-04-15.*
