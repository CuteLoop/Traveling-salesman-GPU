# Correctness and Benchmark Audit

## Scope

This document defines the pre-performance-claim audit path for the TSP implementations in this repository.

Audited implementations:

| Implementation | Source file | Primary binary path | CLI shape |
|---|---|---|---|
| Sequential C GA | `sequential/src/main.c` | `build/Sequential` or `sequential/bin/ga-tsp(.exe)` | `--instance <file> --pop 100 --gen 200 --seed <seed> --elites 2 --tk 3 --pc 0.9 --pm 0.1 --csv <file>` |
| GPU-Naive | `src/cuda/GPU-Naive.cu` | `build/GPU-Naive` | `<file.tsp>` |
| CUDA-GA hybrid | `src/cuda/CUDA-GA.cu` | `build/CUDA-GA` | `<file.tsp> [population=512] [generations=1000] [mutation_rate=0.05] [elite_count=4] [seed]` |
| CUDA-GA-GPU-Pop | `src/cuda/CUDA-GA-GPU-Pop.cu` | `build/CUDA-GA-GPU-Pop` | `<file.tsp> [islands=128] [generations=1000] [mutation_rate=0.05] [elite_count=2] [seed]` |
| CUDA-GA-GPU-Pop-bankconflict | `src/cuda/variants/CUDA-GA-GPU-Pop-bankconflict.cu` | `build/CUDA-GA-GPU-Pop-bankconflict` | same as GPU-Pop |
| CUDA-GA-GPU-Pop-bitset | `src/cuda/variants/CUDA-GA-GPU-Pop-bitset.cu` | `build/CUDA-GA-GPU-Pop-bitset` | same as GPU-Pop |
| CUDA-GA-B1-stride | `src/cuda/variants/CUDA-GA-B1-stride.cu` | `build/CUDA-GA-B1-stride` | same as GPU-Pop |
| CUDA-GA-B2-bitmask | `src/cuda/variants/CUDA-GA-B2-bitmask.cu` | `build/CUDA-GA-B2-bitmask` | same as GPU-Pop |
| CUDA-GA-B3-reduce | `src/cuda/variants/CUDA-GA-B3-reduce.cu` | `build/CUDA-GA-B3-reduce` | same as GPU-Pop |
| CUDA-GA-B3-shuffle | `src/cuda/variants/CUDA-GA-B3-shuffle.cu` | `build/CUDA-GA-B3-shuffle` | same as GPU-Pop |
| CUDA-GA-B4-global | `src/cuda/variants/CUDA-GA-B4-global.cu` | `build/CUDA-GA-B4-global` | same as GPU-Pop |
| CUDA-GA-B4-smem | `src/cuda/variants/CUDA-GA-B4-smem.cu` | `build/CUDA-GA-B4-smem` | same as GPU-Pop |

## Shared Audit Harness

Added deliverables:

- `scripts/tsp_harness.py`: implementation registry, canonical TSPLIB evaluator, output parser, route validator, summary statistics.
- `scripts/audit_tours.py`: verifies solver outputs or raw benchmark CSV rows against the canonical evaluator.
- `scripts/run_seed_benchmark.py`: runs determinism audits or fixed-seed benchmarks and emits raw CSV with validation fields.
- `scripts/aggregate_tsp_results.py`: produces normalized raw CSV, summary CSV, markdown summary, and plot-ready CSV.
- `slurm/run_seed_benchmark.slurm`: generic Slurm entrypoint for determinism and seed-pool benchmarks.
- `Makefile` targets:
  - `make audit_determinism_smoke20`
  - `make audit_determinism_berlin52`
  - `make benchmark_seeds_smoke20`
  - `make benchmark_seeds_berlin52`

Aggregation command:

```bash
python scripts/aggregate_tsp_results.py results/<raw_results>.csv
```

## Correctness Criteria

Each raw benchmark row now records:

- implementation
- dataset
- seed
- repeat id
- command
- runtime seconds
- reported best length
- recomputed best length
- valid tour flag
- missing city count
- duplicate city count
- output text path
- best tour CSV path

The canonical verifier normalizes repeated start-city output before validation and fails the length check if the absolute mismatch exceeds `1e-6`.

## Current Static Findings

### Distance convention audit

- The CUDA family uses the shared C++ TSPLIB parser in `src/cpp/tsplib_parser.cpp` and therefore respects TSPLIB `EDGE_WEIGHT_TYPE` semantics, including `EUC_2D` integer rounding.
- A direct recomputation against `results/cuda_ga_berlin52_5526438.txt` confirmed the reported length `7542` matches TSPLIB `EUC_2D` rounding exactly.
- The sequential C loader in `sequential/src/instance.c` does **not** parse `EDGE_WEIGHT_TYPE`; it reads `NODE_COORD_SECTION` and computes raw Euclidean `sqrt(dx*dx + dy*dy)` distances into a `double` matrix.
- Consequence: sequential results are not yet distance-comparable with the CUDA implementations on TSPLIB instances like `berlin52.tsp` unless the sequential parser is upgraded or results are evaluated externally with the canonical verifier.

### Tour validity and best-length consistency

- The new audit script successfully validated current pulled outputs for `GPU-Naive`, `CUDA-GA`, `CUDA-GA-GPU-Pop`, `CUDA-GA-GPU-Pop-bankconflict`, and `CUDA-GA-GPU-Pop-bitset` with exact recomputation match and valid tours.
- The currently pulled `B1`, `B2`, `B3`, and `B4` text artifacts do not contain a printed best tour, so they are currently marked as `route_present = false` rather than being treated as invalid tours.
- Current source inspection shows those variant binaries do print `Best tour (0-based indices):`, so the missing-route issue appears to be an artifact/version mismatch in the pulled outputs rather than a limitation of the present source tree.
- The harness is designed to audit every produced best-tour CSV or text output before any benchmark summary is trusted.

### RNG audit

| Implementation family | RNG scope | Expected fixed-seed behavior | Notes |
|---|---|---|---|
| Sequential C GA | per-individual host RNG state | deterministic | `ga_driver.c` seeds `rng_states[i]` with `cfg->seed + i` |
| GPU-Naive | none | deterministic | no RNG path |
| CUDA-GA hybrid | single host `std::mt19937` | deterministic | one host RNG drives shuffle, selection, crossover, mutation |
| GPU-Pop control + variants | per-block/per-thread xorshift32 | expected deterministic if block/thread topology is fixed | seed is mixed with island and thread indices; changing topology changes stream assignment |

Additional observations:

- No CUDA implementation uses `curand_init`; the GPU-pop family uses a custom xorshift32 PRNG.
- No inspected kernel reinitializes RNG inside the generation loop.
- Same seed does **not** imply equal tour length across different algorithm families. It only implies reproducibility within the same implementation and launch configuration.

### Race-condition audit

- No obvious global race was found in the inspected GPU-pop kernels for best-tour export: each island maps to one CUDA block, and a single writer thread stores `best_lengths[island]` and the corresponding `best_tours[...]` after synchronization.
- The control, B1, B2, B3, and B4 variants all use block-local synchronization (`__syncthreads()`) around shared-memory phases.
- `B3-reduce` and `B3-shuffle` alter the elite-selection reduction strategy. They should be treated as potential semantic changes until the fixed-seed benchmark confirms equivalent result distributions.

## Fair Comparison Groups

Group A: algorithm family comparison

- Sequential C GA
- CUDA-GA hybrid
- CUDA-GA-GPU-Pop

Group B: incremental optimization story

- CUDA-GA-GPU-Pop
- CUDA-GA-GPU-Pop-bankconflict
- CUDA-GA-GPU-Pop-bitset
- CUDA-GA-B1-stride
- CUDA-GA-B2-bitmask
- CUDA-GA-B3-reduce
- CUDA-GA-B3-shuffle
- CUDA-GA-B4-global
- CUDA-GA-B4-smem

Group B must keep the same seed pool and the same island / generation / mutation / elitism budget. If any variant changes the search process rather than only the cost of the same operations, document it as an algorithmic change.

## Reproducibility Workflow

Determinism audit:

```bash
make audit_determinism_smoke20
make audit_determinism_berlin52
```

Seed-pool benchmark:

```bash
make benchmark_seeds_smoke20
make benchmark_seeds_berlin52
```

Raw benchmark CSV rows are then aggregated into summary artifacts with mean, median, standard deviation, and success-rate fields.

## Statistical Benchmark Tables

This section should be filled from the generated `*_summary.csv` outputs after the Slurm runs complete.

Reference lengths:

- `berlin52`: `7542` (known optimum)
- `smoke_20`: `73` (best known from current repository experiments; replace if an exact optimum is confirmed)

Required reporting fields:

- minimum best length
- maximum best length
- mean best length
- median best length
- standard deviation of best length
- mean runtime
- median runtime
- standard deviation of runtime
- success rate within 0%, 1%, 5%, and 10% of the reference length

## Pass / Fail Checklist

Before making any optimization claim, confirm all four conditions:

1. tours are valid,
2. lengths are recomputed correctly,
3. fixed-seed repeatability is understood,
4. mean and standard deviation over the fixed seed pool support the claim.

## Current Warnings

- Do not compare sequential Berlin52 lengths directly to CUDA Berlin52 lengths until the sequential parser is brought in line with TSPLIB `EUC_2D` semantics.
- Do not treat same-seed differences across algorithm families as evidence of incorrectness.
- Do not describe B3/B4 variants as pure optimizations until the seed-pool benchmark shows they preserve the intended search semantics.