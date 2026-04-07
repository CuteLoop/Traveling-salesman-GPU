# Test Plan — Sequential GA for TSP
### `ga-tsp` · University of Arizona · Applied Mathematics
**Version:** 1.0 · **Status:** Living Document — update on every module change

---

> **Philosophy.**  A test does not pass when the logic is correct.  
> A test passes when the logic is correct *and* memory is fully accounted for.  
> Every suite from Phase 2 onward must be clean under AddressSanitizer.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Testing Philosophy](#2-testing-philosophy)
3. [Core Invariants](#3-core-invariants)
4. [Required Tooling](#4-required-tooling)
5. [Test Organization](#5-test-organization)
6. [Module-Level Test Suites](#6-module-level-test-suites)
7. [Phase Gates](#7-phase-gates)
8. [Small Test Fixtures](#8-small-test-fixtures)
9. [GPU-Readiness Test Notes](#9-gpu-readiness-test-notes)
10. [Test Maintenance Rules](#10-test-maintenance-rules)
11. [Definition of Done](#11-definition-of-done)
12. [Living Checklist](#12-living-checklist)

---

## 1. Purpose

This document defines the complete test strategy for the sequential C99 implementation of a Genetic Algorithm (GA) for the Traveling Salesman Problem (TSP).

The goals are to:

- verify correctness of every individual module in isolation
- preserve invariants across refactors and operator additions
- support strict test-driven development (TDD) — tests are written *before* implementation
- establish a trustworthy, memory-clean serial baseline before GPU migration
- create regression gates that later CUDA work must preserve exactly

---

## 2. Testing Philosophy

The strategy is layered from smallest scope to largest.  No layer is skipped.

| Layer | Scope | Purpose |
|-------|-------|---------|
| **Unit** | Single function | Verify isolated behavior and edge cases |
| **Property** | Invariant across random inputs | Confirm algebraic / combinatorial correctness |
| **Integration** | Module interactions | Validate cross-boundary contracts |
| **Memory** | Full allocation lifecycle | Zero leaks, zero out-of-bounds under ASan / Valgrind |
| **Smoke** | End-to-end on tiny instance | Confirm the system does not crash |
| **Regression** | Fixed seed + fixed instance | Freeze serial baseline before GPU work begins |

### Compile Flags Required for Every Test Run

```bash
# Debug / correctness build — mandatory for all test runs
-O0 -g -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer

# Regression / benchmark build — used only in Phase 10
-O3 -march=native -ffast-math -DNDEBUG
```

> **Rationale for two builds.**  ASan and `-O3` are incompatible in practice.  
> Correctness is verified at `-O0 + ASan`; performance is measured at `-O3`.  
> Comparing GPU results against an unoptimized CPU baseline is scientifically invalid.

---

## 3. Core Invariants

These must hold at all times.  Any violation is a blocking defect.

| # | Invariant | Checked By |
|---|-----------|------------|
| I-1 | Every valid tour contains each city exactly once | `test_tour_validity.c` |
| I-2 | Distance matrix diagonal is identically zero | `test_distance_matrix.c` |
| I-3 | Distance matrix is symmetric: $d_{ij} = d_{ji}$ | `test_distance_matrix.c` |
| I-4 | Population size is constant across generations | `test_replacement.c` |
| I-5 | Elitism preserves the best-so-far solution | `test_elitism.c` |
| I-6 | Batch evaluation matches individual evaluation entry-wise | `test_fitness.c` |
| I-7 | A fixed seed fully reproduces any run | `test_rng.c`, regression suite |
| I-8 | No genetic operator introduces a memory leak | ASan / Valgrind on every suite |

---

## 4. Required Tooling

| Tool | Purpose | When Required |
|------|---------|---------------|
| `make test` | Build and run all suites | Every commit |
| AddressSanitizer (`-fsanitize=address`) | Heap/stack bounds | Phase 2 onward |
| UndefinedBehaviorSanitizer (`-fsanitize=undefined`) | UB detection | Phase 2 onward |
| `valgrind --leak-check=full` | Leak audit | Phase 9 smoke test |
| `gprof` / `perf` | CPU hotspot profiling | Phase 10 only |

Install on the UA HPC cluster:
```bash
module load gcc/12        # ASan ships with GCC >= 4.8
module load valgrind
```

---

## 5. Test Organization

```
tests/
├── fixtures/
│   ├── square_4.tsp          # 4-city unit square — hand-computable
│   ├── malformed.tsp         # intentionally broken — parser rejection
│   ├── smoke_20.tsp          # 20-city Euclidean — smoke runs
│   └── benchmark_100.tsp    # 100-city — regression baseline only
│
├── test_instance.c
├── test_distance_matrix.c
├── test_tour_validity.c
├── test_fitness.c
├── test_rng.c
├── test_initialization.c
├── test_selection.c
├── test_crossover.c
├── test_mutation.c
├── test_elitism.c
├── test_replacement.c
├── test_memory.c             # ASan / Valgrind integration targets
├── test_ga_smoke.c
└── test_regression.c
```

Each file is independently compilable and linkable against the production library.  There is no shared global test state between files.

---

## 6. Module-Level Test Suites

Each suite lists: objectives, concrete test cases, the phase at which it must first pass, and memory requirements.

---

### 6.1 Instance Parser · `test_instance.c`

**Objectives**
- Confirm files load without error on valid input
- Confirm failure is returned (not a crash) on malformed input
- Verify coordinate parsing fidelity

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| P-01 | Load `square_4.tsp` | Returns success; `n == 4` |
| P-02 | Load `malformed.tsp` | Returns error code, no crash |
| P-03 | City count matches file header | `instance.n == expected` |
| P-04 | First three coordinates match fixture values | Within `1e-9` |
| P-05 | Parser on empty file | Returns error, no segfault |

**Memory requirement:** No leaks after `tsp_instance_free()`.  
**First required:** End of Phase 1.

---

### 6.2 Distance Matrix · `test_distance_matrix.c`

**Objectives**
- Verify geometric correctness and algebraic consistency

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| D-01 | Diagonal entries $d_{ii} = 0$ for all $i$ | Exact zero |
| D-02 | Matrix is symmetric: $d_{ij} = d_{ji}$ | Difference `< 1e-12` |
| D-03 | `square_4` edge lengths match $\sqrt{2}$ or $1.0$ | Within `1e-9` |
| D-04 | Rebuild produces identical values | Bitwise or `< 1e-15` |
| D-05 | Triangle inequality holds for random triples | $d_{ik} \leq d_{ij} + d_{jk}$ |

**Memory requirement:** No leaks after matrix free.  
**First required:** End of Phase 1.

---

### 6.3 Tour Validity · `test_tour_validity.c`

**Objectives**
- Ensure the validator correctly accepts valid permutations and rejects all malformed ones

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| V-01 | Valid permutation `[0,1,2,3]` | Pass |
| V-02 | Duplicate city `[0,1,1,3]` | Fail |
| V-03 | Missing city `[0,1,3,3]` | Fail |
| V-04 | Out-of-range index `[0,1,2,99]` with $n=4$ | Fail |
| V-05 | Deep copy has same values | Pass |
| V-06 | Deep copy does not alias source memory | `cities != src->cities` |
| V-07 | Tour of length 1 | Pass (trivial cycle) |

**Memory requirement:** No leaks from allocation and free of copied tours.  
**First required:** End of Phase 2.

---

### 6.4 Fitness Evaluation · `test_fitness.c`

**Objectives**
- Verify closed-tour length computation is correct and consistent

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| F-01 | `square_4` identity tour has length $4.0$ | Within `1e-9` |
| F-02 | Same tour evaluated twice gives identical result | Exact equality |
| F-03 | Rotationally equivalent tour has same length | Within `1e-9` |
| F-04 | Population batch evaluation matches individual evaluation | Entry-wise `< 1e-12` |
| F-05 | Fitness is inverse of length; shorter tour has higher fitness | Strict inequality |
| F-06 | Tour of length $n=1$ evaluates without crash | Finite result |

**Memory requirement:** Evaluation must not allocate heap memory (all stack).  
**First required:** End of Phase 3.

---

### 6.5 RNG Layer · `test_rng.c`

**Objectives**
- Enforce full reproducibility and statistical divergence between seeds

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| R-01 | Same seed → same integer stream (first 1000 values) | Exact match |
| R-02 | Same seed → same `[0,1)` real stream | Exact match |
| R-03 | Different seeds diverge within 10 draws | At least one difference |
| R-04 | RNG state is fully contained in passed struct (no global) | Verified by two independent instances running in parallel |

**Memory requirement:** RNG state is stack- or caller-allocated; no heap.  
**First required:** End of Phase 4.

---

### 6.6 Initialization · `test_initialization.c`

**Objectives**
- Verify valid random tour generation and deterministic population seeding

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| I-01 | Single generated tour is a valid permutation | Pass V-01 check |
| I-02 | Population has exactly `POP_SIZE` individuals | `pop.size == N` |
| I-03 | All individuals in initial population are valid | All pass V-01 |
| I-04 | Same seed reproduces same initial population | Individual-wise exact match |
| I-05 | Walking skeleton runs: init + evaluate + print best, no crash | Exit code 0, finite output |

> **I-05 is the Walking Skeleton test.**  It must pass by end of Phase 4 even though evolution operators do not yet exist.  This prevents Phase 9 integration hell.

**Memory requirement:** Full population frees without leaks.  
**First required:** End of Phase 4.

---

### 6.7 Selection · `test_selection.c`

**Objectives**
- Confirm selection returns legal parents and statistically favors fitter individuals

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| S-01 | Returned parent index is in `[0, N)` | Bounds check passes |
| S-02 | Returned parent pointer/copy belongs to current population | Identity or value match |
| S-03 | Best individual wins majority of 1000-trial tournaments in controlled fixture | Win rate `> 0.9` for $k=2$ against weakest |
| S-04 | $k=1$ selection is uniform (no bias) | All individuals selected roughly equally |

**Memory requirement:** No heap allocation inside selection function.  
**First required:** End of Phase 5.

---

### 6.8 Crossover · `test_crossover.c`

**Objectives**
- Ensure offspring validity and correct probability control

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| C-01 | Child is a valid permutation | Pass V-01 |
| C-02 | Child city count equals $n$ | Exact |
| C-03 | No duplicates over 10,000 randomized crossover trials | Zero failures |
| C-04 | `p_c = 0.0` → child is a copy of parent A | Exact value match |
| C-05 | `p_c = 1.0` → child is never identical to either parent (probabilistically) | Fails with probability $< (1/n!)^2$ |
| C-06 | Both parents identical → child identical | Exact match |

**Memory requirement:** Offspring memory freed without leak; no internal heap allocation after first call.  
**First required:** End of Phase 6.

---

### 6.9 Mutation · `test_mutation.c`

**Objectives**
- Ensure mutation preserves permutation validity under all probability settings

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| M-01 | Mutated tour is a valid permutation | Pass V-01 |
| M-02 | City multiset is unchanged after mutation | Sorted arrays identical |
| M-03 | `p_m = 0.0` → tour is unchanged | Exact value match |
| M-04 | `p_m = 1.0` → tour changes in at least 1 of 100 trials | Probabilistic |
| M-05 | Inversion mutation: reversed segment is valid | V-01 passes |
| M-06 | Swap mutation: exactly two positions differ from original | Hamming distance = 2 |

**Memory requirement:** Mutation is in-place; no heap allocation.  
**First required:** End of Phase 7.

---

### 6.10 Elitism · `test_elitism.c`

**Objectives**
- Confirm that top-$e$ individuals are correctly identified and preserved

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| E-01 | Elite extraction returns $e$ individuals | Count exact |
| E-02 | Extracted elites are the top-$e$ by fitness | Verified by sorting and comparison |
| E-03 | Best individual in generation $g$ appears in generation $g+1$ | Fitness non-regression |
| E-04 | `e = 0` (no elitism) runs without crash | Exit code 0 |
| E-05 | `e = N` (full elitism) → population unchanged | Value match |

**Memory requirement:** No leaks from elite copy / free cycle.  
**First required:** End of Phase 8.

---

### 6.11 Replacement · `test_replacement.c`

**Objectives**
- Verify next-generation assembly is complete and correctly sized

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| Re-01 | Next-generation population has exactly `N` individuals | Exact |
| Re-02 | No null or uninitialized tour pointers in new population | All pass V-01 |
| Re-03 | Elite slots are filled before offspring slots | Order enforced |
| Re-04 | Replacement does not alias tours across generations | Source and destination memory disjoint |

**Memory requirement:** Previous generation memory is freed completely.  
**First required:** End of Phase 8.

---

### 6.12 Memory and Sanitization · `test_memory.c`

**Objectives**
- Guarantee zero leaks and zero out-of-bounds access across all module boundaries

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| Mem-01 | Smoke test under `valgrind --leak-check=full` | "0 bytes in 0 blocks" leaked |
| Mem-02 | Full suite under `-fsanitize=address` | Zero ASan aborts |
| Mem-03 | Full suite under `-fsanitize=undefined` | Zero UBSan aborts |
| Mem-04 | 500-generation run under ASan | No heap buffer overflow in crossover or mutation |
| Mem-05 | `tsp_instance_free` after partial initialization | No crash, no leak |

**First required:** End of Phase 2 (and continuously thereafter).

---

### 6.13 GA Smoke Test · `test_ga_smoke.c`

**Objectives**
- Confirm end-to-end execution completes correctly on a tiny instance

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| G-01 | GA runs 100 generations on `smoke_20.tsp` without crash | Exit code 0 |
| G-02 | Final best objective is a finite positive double | No NaN, no Inf |
| G-03 | Best-so-far is non-increasing across all generations | Monotone under elitism |
| G-04 | Statistics log has exactly `G` entries | Count exact |
| G-05 | `valgrind` on smoke run reports zero leaks | Clean |

**First required:** End of Phase 9.

---

### 6.14 Regression Baseline · `test_regression.c`

**Objectives**
- Freeze serial behavior as the ground-truth reference before any GPU work

**Test Cases**

| ID | Description | Expected |
|----|-------------|----------|
| Reg-01 | Seed 42 + `benchmark_100.tsp` → best distance within `±0.1%` of recorded baseline | Pass |
| Reg-02 | Convergence history shape (first/last/min) matches stored reference | Within `1e-6` |
| Reg-03 | All suites 6.1–6.13 still pass | Zero regressions |
| Reg-04 | Build under `-O3 -march=native` completes and all regression tests pass | No correctness change under optimization |
| Reg-05 | Profiler confirms fitness evaluation is the dominant hotspot | Validates GPU porting priority |

**First required:** End of Phase 10.  Re-run after every substantial refactor.

---

## 7. Phase Gates

A phase is closed **only** when its gate passes completely.  No partial credit.

| Phase | Name | Required Passing Suites |
|-------|------|------------------------|
| 0 | Build & Skeleton | Build clean; empty smoke exits 0 |
| 1 | Instance & Distance | 6.1, 6.2 |
| 2 | Tour Representation | 6.3, 6.12 (ASan enabled from here) |
| 3 | Fitness | 6.4 |
| 4 | RNG, Init & Walking Skeleton | 6.5, 6.6 (including I-05) |
| 5 | Selection | 6.7 |
| 6 | Crossover | 6.8 |
| 7 | Mutation | 6.9 |
| 8 | Elitism & Replacement | 6.10, 6.11 |
| 9 | GA Driver & Smoke | 6.13, 6.12 (Valgrind full) |
| 10 | Regression Lock | 6.14 + all prior suites + `-O3` build |

---

## 8. Small Test Fixtures

All fixtures live in `tests/fixtures/`.  They must be small enough that every suite runs in under 5 seconds on a laptop.

| File | Cities | Purpose |
|------|--------|---------|
| `square_4.tsp` | 4 | Unit square; all distances hand-computable |
| `malformed.tsp` | — | Intentionally broken; parser must reject, not crash |
| `smoke_20.tsp` | 20 | Small Euclidean instance; end-to-end smoke |
| `benchmark_100.tsp` | 100 | Regression baseline only; not used in unit tests |

The 4-city square has known optimal tour length of $4.0$ and known distance matrix — this gives concrete expected values for tests D-03 and F-01.

---

## 9. GPU-Readiness Test Notes

Several tests are written specifically to support the future CUDA port.

| Test | GPU Migration Role |
|------|--------------------|
| F-04 (batch vs. individual evaluation) | Validates the interface that becomes the evaluation kernel |
| R-04 (no global RNG) | Confirms per-thread `cuRAND` state will be a drop-in replacement |
| V-01–V-07 (tour validity) | Ground-truth checker for kernel output correctness |
| Reg-01–Reg-05 (regression baseline) | Reference against which GPU results are compared for correctness |
| Mem-04 (ASan on long run) | Reveals allocation patterns that must be restructured for `cudaMalloc` |

---

## 10. Test Maintenance Rules

Update this document whenever any of the following occurs:

- [ ] A new genetic operator is added
- [ ] A data structure changes its memory layout
- [ ] A module boundary changes
- [ ] A bug is fixed — a new regression test capturing it must be added immediately
- [ ] A benchmark fixture is added or retired
- [ ] A CUDA kernel replaces a serial module — its correctness tests are added here

---

## 11. Definition of Done

A phase is complete if and only if:

1. The intended feature is implemented and compiles cleanly under both builds
2. The corresponding tests exist and were written before the implementation (TDD)
3. All designated tests pass from a clean `make clean && make test`
4. No previously passing suite has regressed
5. The ASan / Valgrind requirement for the phase is satisfied
6. Any new bug discovered during the phase is captured in a new named test before the fix is committed

---

## 12. Living Checklist

Update this table as phases close.

| Suite | Status | Date Closed | Notes |
|-------|--------|-------------|-------|
| Build system stable | ☐ | — | |
| Parser + distance matrix | ☐ | — | |
| Tour validity | ☐ | — | |
| Fitness evaluation | ☐ | — | |
| RNG reproducibility | ☐ | — | |
| Walking skeleton runs | ☐ | — | I-05 specifically |
| Initialization | ☐ | — | |
| Selection | ☐ | — | |
| Crossover | ☐ | — | |
| Mutation | ☐ | — | |
| Elitism | ☐ | — | |
| Replacement | ☐ | — | |
| Memory / ASan clean | ☐ | — | |
| GA smoke passes | ☐ | — | |
| Regression baseline recorded | ☐ | — | |
| `-O3` build verified | ☐ | — | |

---

*Test Plan · `ga-tsp` · Joel Maldonado · University of Arizona · v1.0*