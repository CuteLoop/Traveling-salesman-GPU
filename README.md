# Traveling Salesman Problem - GPU Approaches

This repository tracks a staged TSP implementation arc:
- Python baseline
- sequential C GA
- naive CUDA baseline
- hybrid CUDA-GA and GPU-resident island-GA variants

The repo is organized around experiment workflows rather than a single executable. Core implementation notes live in `docs/`, and the experiment runbooks live in `docs/EXPERIMENTS/`.

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Build and Run](#build-and-run)
- [HPC Workflow](#hpc-workflow)
- [Headline Results](#headline-results)
- [Optimization Story](#optimization-story)
- [Documentation Map](#documentation-map)
- [Current Caveats](#current-caveats)

## Overview

Current status:

| Stage | Status | Main entry points |
|---|---|---|
| Python baseline | Done | `baselines/py_combinatorial_ga_example_berlin52.py`, `baselines/pycombinatorial_latlong_compare.py` |
| Sequential C GA | Done | `sequential/`, `build/Sequential` |
| CUDA baseline | Done | `src/cuda/GPU-Naive.cu` |
| CUDA GA variants | In progress | `src/cuda/CUDA-GA.cu`, `src/cuda/CUDA-GA-GPU-Pop.cu`, `src/cuda/variants/` |

Primary references:

- [docs/EXPERIMENTS/README.md](docs/EXPERIMENTS/README.md)
- [docs/optimization-roadmap.md](docs/optimization-roadmap.md)
- [docs/report/report.md](docs/report/report.md)

## Repository Layout

```text
.
|-- baselines/                  Python reference implementations and comparison scripts
|-- sequential/                 Sequential C GA implementation
|-- src/cuda/                   CUDA implementations
|-- src/cpp/                    Shared TSPLIB parser
|-- slurm/                      HPC batch scripts
|-- scripts/                    Local/HPC helper scripts
|-- docs/                       Implementation notes, experiments, and report material
|-- data/                       TSPLIB and Madeira datasets
|-- results/                    Result artifacts and plots
`-- README.md
```

## Quick Start

### Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Tests

```bash
python -m pytest tests/ -v
```

### Build everything from the root Makefile

```bash
make all
```

## Build and Run

### Python baselines

Berlin52 GA baseline:

```bash
python baselines/py_combinatorial_ga_example_berlin52.py
```

Madeira comparison:

```bash
python baselines/pycombinatorial_latlong_compare.py
```

### Sequential C GA

Build from `sequential/`:

```bash
cd sequential
make clean
make all
make test
```

Example run:

```bash
./bin/ga-tsp --instance tests/fixtures/smoke_20.tsp --pop 100 --gen 200 --seed 42 --elites 2 --tk 3 --pc 0.9 --pm 0.1 --csv results.csv
```

### CUDA builds

Build all wired targets:

```bash
make all
```

Build only the CUDA matrix targets used by the Slurm runners:

```bash
make all_cuda_versions
```

Print the standardized CUDA matrix config:

```bash
make print_cuda_matrix_config
```

Current root Makefile targets include:

- `build/Sequential`
- `build/GPU-Naive`
- `build/CUDA-GA`
- `build/CUDA-GA-no-greedy`
- `build/CUDA-GA-GPU-Pop`
- `build/CUDA-GA-GPU-Pop-bankconflict`
- `build/CUDA-GA-GPU-Pop-bitset`
- `build/GA-GPU-POP-ParallelSort`
- `build/GA-GPU-POP-AoS`
- `build/GA-GPU-POP-GlobalDist`
- `build/GA-GPU-POP-VerboseComments`
- `build/CUDA-GA-B1-stride`
- `build/CUDA-GA-B2-bitmask`
- `build/CUDA-GA-B3-reduce`
- `build/CUDA-GA-B3-shuffle`
- `build/CUDA-GA-B4-global`
- `build/CUDA-GA-B4-smem`
- `build/CUDA-GA-B5-bigpop`
- `build/CUDA-GA-C1-stride`
- `build/CUDA-GA-C2-bitmask`
- `build/CUDA-GA-C3-reduce`
- `build/CUDA-GA-C4-global`
- `build/CUDA-GA-C5-bigpop`

Direct compile example for a single variant from repo root:

```bash
nvcc -O3 -std=c++11 -Xcompiler -std=gnu++11 -arch=sm_60 -lineinfo -Isrc/cpp \
  -o /tmp/t src/cuda/variants/CUDA-GA-B1-stride.cu src/cpp/tsplib_parser.cpp
```

## HPC Workflow

Standard CUDA matrix config used for the all-variants Slurm sweeps:

| Setting | Standard value |
|---|---:|
| `RUNS` | `20` |
| `ISLANDS` | `256` |
| `GENERATIONS` | `2000` |
| `MUTATION` | `0.03` |
| `ELITE_POP` | `2` |
| `ELITE_HYBRID` | `4` |
| `POP_HYBRID` | `512` |
| `SEED_BASE` | `100` |

Datasets used with that config:

| Sweep | Dataset |
|---|---|
| smoke_20 | `sequential/tests/fixtures/smoke_20.tsp` |
| berlin52 | `data/berlin52.tsp` |

Preferred entry points from the repo root on HPC:

```bash
make sbatch_cuda_smoke20
make sbatch_cuda_berlin52
```

Those Make targets submit `slurm/run_cuda_all_variants_csv.slurm` with the standard config above.

After a completed sweep writes its CSVs, regenerate the README result tables with:

```bash
make refresh_readme_cuda_results
```

If you need one-off overrides, pass them through `make`:

```bash
make sbatch_cuda_berlin52 CUDA_MATRIX_RUNS=5
make sbatch_cuda_smoke20 CUDA_MATRIX_GENERATIONS=1000 CUDA_MATRIX_SEED_BASE=200
```

The most useful all-variants runner is:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,RUNS=20,ISLANDS=256,GENERATIONS=2000,MUTATION=0.03,ELITE_POP=2,ELITE_HYBRID=4,POP_HYBRID=512,SEED_BASE=100 slurm/run_cuda_all_variants_csv.slurm
```

That script:
- always invokes `make all_cuda_versions` so source edits propagate (an earlier short-circuit on `[ -x build/CUDA-GA-C5-bigpop ]` could leave stale binaries in place after print/format changes; that guard has been removed)
- runs the current CUDA matrix, including B1 through B5 and C1 through C5
- writes one per-run CSV (`results/cuda_all_variants_runs_<dataset>_<jobid>.csv`) and one aggregated avg CSV (`results/cuda_all_variants_avg_<dataset>_<jobid>.csv`)

If you suspect the cached binaries are out of sync with the sources for any reason, force a clean rebuild before submitting:

```bash
make clean
make all_cuda_versions
```

Then resubmit the sweep. This is rarely necessary now that the SLURM script always defers to `make`'s dependency tracking, but it's a useful one-shot reset.

Outputs for the example above are:

- `results/cuda_all_variants_runs_berlin52_<jobid>.csv`
- `results/cuda_all_variants_avg_berlin52_<jobid>.csv`

One-shot runner, one execution per variant:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,ISLANDS=256,GENERATIONS=2000,MUTATION=0.03,ELITE_POP=2,SEED=100 slurm/run_cuda_all_variants.slurm
```

Other experiment runners:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,RUNS=10 slurm/run_cuda_all_variants_csv_randomseed.slurm
sbatch --export=ALL,DATASET=data/berlin52.tsp,TARGET_LENGTH=7542,RUNS=10 slurm/run_cuda_modified_target_avg.slurm
```

For smoke_20 with the same standardized matrix config:

```bash
sbatch --export=ALL,DATASET=sequential/tests/fixtures/smoke_20.tsp,RUNS=20,ISLANDS=256,GENERATIONS=2000,MUTATION=0.03,ELITE_POP=2,ELITE_HYBRID=4,POP_HYBRID=512,SEED_BASE=100 slurm/run_cuda_all_variants_csv.slurm
```

Experiment references:

- [docs/EXPERIMENTS/Experiment2_cuda_matrix_seed_schedule.md](docs/EXPERIMENTS/Experiment2_cuda_matrix_seed_schedule.md)
- [docs/EXPERIMENTS/Experiment3_cuda_matrix_random_seed.md](docs/EXPERIMENTS/Experiment3_cuda_matrix_random_seed.md)
- [docs/EXPERIMENTS/Experiment4_target_length_runtime.md](docs/EXPERIMENTS/Experiment4_target_length_runtime.md)

## Headline Results

Detailed methodology lives in [docs/EXPERIMENTS/README.md](docs/EXPERIMENTS/README.md).

### smoke_20

Latest completed standardized CUDA sweep:

- dataset: `sequential/tests/fixtures/smoke_20.tsp`
- runner: `slurm/run_cuda_all_variants_csv.slurm`
- settings: `RUNS=20, ISLANDS=256, GENERATIONS=2000, MUTATION=0.03, ELITE_POP=2, ELITE_HYBRID=4, POP_HYBRID=512, SEED_BASE=100`
- latest completed artifacts used here:
  - `results/cuda_all_variants_runs_smoke_20_5532235.csv`
  - `results/cuda_all_variants_avg_smoke_20_5532235.csv`

Per-implementation results from that sweep:

| Variant | Best length | Avg length | Length stddev | Best time (ms) | Avg time (ms) | Time stddev (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `gpu_naive` | 80 | 80 | 0 | 0.131155 | 0.135038 | 0.006551 |
| `cuda_ga` | 75 | 75 | 0 | 207.458 | 208.87 | 2.15816 |
| `cuda_ga_gpu_pop` | 73 | 73 | 0 | 29.4514 | 30.4586 | 1.72076 |
| `cuda_ga_gpu_pop_bankconflict` | 73 | 73 | 0 | 29.2197 | 30.2686 | 1.75376 |
| `cuda_ga_gpu_pop_bitset` | 73 | 73 | 0 | 27.4659 | 28.4726 | 1.65234 |
| `cuda_ga_gpu_pop_parallel_sort` | 73 | 73 | 0 | 94.5684 | 95.2308 | 0.832296 |
| `cuda_ga_gpu_pop_aos` | 73 | 73 | 0 | 166.686 | 169.545 | 4.45991 |
| `cuda_ga_gpu_pop_global_dist` | 73 | 73 | 0 | 34.6396 | 34.6408 | 0.001097 |
| `cuda_ga_gpu_pop_verbose_comments` | 73 | 73 | 0 | 114.175 | 114.732 | 0.539419 |
| `cuda_ga_b1_stride` | 73 | 73 | 0 | 94.3831 | 95.2463 | 0.829161 |
| `cuda_ga_b2_bitmask` | 73 | 73 | 0 | 93.1425 | 93.773 | 0.547147 |
| `cuda_ga_b3_reduce` | 73 | 73 | 0 | 92.845 | 93.3016 | 0.412932 |
| `cuda_ga_b3_shuffle` | 73 | 73 | 0 | 93.4792 | 93.6598 | 0.188824 |
| `cuda_ga_b4_global` | 73 | 73 | 0 | 92.7258 | 93.1482 | 0.401541 |
| `cuda_ga_b5_bigpop` | 73 | 73 | 0 | 246.325 | 246.701 | 0.332447 |
| `cuda_ga_b4_smem` | 73 | 73 | 0 | 93.1285 | 93.6578 | 0.510209 |

Notes:

- Many variants tie at the smoke_20 optimum length `73`, so the timing columns are what separate them under this configuration.
- On smoke_20, `cuda_ga_gpu_pop_bitset` is the fastest timed variant among the implementations that also hit the best observed length.
- For the optimization story, smoke_20 should be treated as a correctness and regression check, not as the main performance-comparison dataset; the small problem size does not reflect the bottlenecks the B/C family is targeting.

### berlin52

Latest completed standardized CUDA sweep:

- dataset: `data/berlin52.tsp`
- runner: `slurm/run_cuda_all_variants_csv.slurm`
- settings: `RUNS=20, ISLANDS=256, GENERATIONS=2000, MUTATION=0.03, ELITE_POP=2, ELITE_HYBRID=4, POP_HYBRID=512, SEED_BASE=100`
- latest completed artifacts used here:
  - `results/cuda_all_variants_runs_berlin52_5532233.csv`
  - `results/cuda_all_variants_avg_berlin52_5532233.csv`

Per-implementation results from that sweep:

| Variant | Best length | Avg length | Length stddev | Best time (ms) | Avg time (ms) | Time stddev (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `gpu_naive` | 8181 | 8181 | 0 | NA | NA | NA |
| `cuda_ga` | 7604 | 7795.45 | 208.35 | NA | NA | NA |
| `cuda_ga_gpu_pop` | 8929 | 9125.4 | 120.514 | 330.932 | 334.398 | 4.3233 |
| `cuda_ga_gpu_pop_bankconflict` | 8929 | 9125.4 | 120.514 | 327.868 | 330.908 | 1.60382 |
| `cuda_ga_gpu_pop_bitset` | 8929 | 9125.4 | 120.514 | 283.419 | 284.177 | 0.341427 |
| `cuda_ga_gpu_pop_parallel_sort` | 8836 | 9089.1 | 126.029 | NA | NA | NA |
| `cuda_ga_gpu_pop_aos` | 8929 | 9125.4 | 120.514 | NA | NA | NA |
| `cuda_ga_gpu_pop_global_dist` | 8929 | 9125.4 | 120.514 | 279.225 | 279.731 | 0.255912 |
| `cuda_ga_gpu_pop_verbose_comments` | 8929 | 9125.4 | 120.514 | NA | NA | NA |
| `cuda_ga_b1_stride` | 8777 | 9234.55 | 195.84 | NA | NA | NA |
| `cuda_ga_b2_bitmask` | 8777 | 9234.55 | 195.84 | NA | NA | NA |
| `cuda_ga_b3_reduce` | 8777 | 9234.55 | 195.84 | NA | NA | NA |
| `cuda_ga_b3_shuffle` | 8777 | 9234.55 | 195.84 | NA | NA | NA |
| `cuda_ga_b4_global` | 8777 | 9234.55 | 195.84 | NA | NA | NA |
| `cuda_ga_b5_bigpop` | 7542 | 7542 | 0 | NA | NA | NA |
| `cuda_ga_b4_smem` | 8777 | 9234.55 | 195.84 | NA | NA | NA |

Interpretation:

- `cuda_ga_b5_bigpop` is the best quality variant in the latest completed Berlin52 sweep and reaches the known Berlin52 optimum `7542` consistently across runs.
- `cuda_ga_gpu_pop_global_dist` is the fastest timed CUDA variant on Berlin52 under the standardized settings, but it does not match the solution quality of `cuda_ga_b5_bigpop`.
- The current Berlin52 tradeoff is still quality versus speed, not one variant dominating both.

Timing caveat:

- `NA` timing cells in the Berlin52 table above come from the 5532233 sweep, which ran before the SLURM build-skip guard was removed. That guard short-circuited `make all_cuda_versions` whenever `build/CUDA-GA-C5-bigpop` already existed on disk, so source edits that added or changed timing prints (the chrono-based `CUDA kernel elapsed ms:` line in the AoS / ParallelSort / VerboseComments / B-series variants) never propagated into the binaries that were actually executed. The runner's awk extractor then found nothing to parse and recorded NA. With the guard gone, a fresh sweep populates these cells.
- The Slurm parser still grabs `CUDA kernel elapsed ms` from stdout. Source-side per-variant CSV writing (`src/cpp/result_writer.h`, included by the B and C variants) is wired in but currently inert — the runner does not yet set `RESULT_CSV` / `RUN_ID` env vars to activate it. That is the next planned hardening step; it does not block reruns.
- `cuda_ga_no_greedy` and `cuda_ga_c1` through `cuda_ga_c5` will appear automatically once a fresh sweep completes.

To regenerate this table after a fresh sweep, point the script at the new job IDs:

```bash
make refresh_readme_cuda_results
```

## Optimization Story

The repo currently has one implemented optimization story family:

- B1: stride padding in shared memory
- B2: bitmask-based OX bookkeeping
- B3-reduce / B3-shuffle: faster elite selection
- B4-global / B4-smem: distance-memory layout experiments
- B5-bigpop: large population layout plus greedy nearest-neighbor seeding

A useful way to think about the current result split is:

- B1-B4 mostly change how the same search is executed.
- B5 changes how the search starts.

Current hypothesis for why B5 is much better on Berlin52:

1. B5 computes nearest-neighbor tours on the GPU with `greedy_nn_kernel`, chooses the best start city, and seeds each island from that strong tour with only small perturbations.
2. B1-B4 start each island from a fully random permutation, so they spend many generations climbing out of much worse initial tours.
3. On Berlin52 with `2000` generations and `256` islands, initialization quality appears to dominate the kernel-level improvements.
4. The identical or near-identical B1-B4 aggregate results suggest those variants are still exploring essentially the same search basin, just with different implementation details.

So the current evidence does not say B1-B4 are useless. It says their optimizations are mostly throughput optimizations, while B5 introduced a search-quality optimization.

Planned next framing for the optimization story:

- B1-B5: kernel and memory optimization family
- C1-C5: same progression, but with the C5 endpoint explicitly adding greedy nearest-neighbor population seeding

That split would make the comparison cleaner:
- B-family answers: how much do kernel optimizations help if search behavior stays roughly the same?
- C-family answers: what changes when the search starts from a stronger population?

## Documentation Map

Implementation and architecture:

- [docs/cuda-ga-implementation.md](docs/cuda-ga-implementation.md)
- [docs/cuda-ga-gpu-pop-implementation.md](docs/cuda-ga-gpu-pop-implementation.md)
- [docs/gpu-naive-cuda-implementation.md](docs/gpu-naive-cuda-implementation.md)

Optimization notes:

- [docs/optimization-roadmap.md](docs/optimization-roadmap.md)
- [docs/optimization-story.md](docs/optimization-story.md)
- [docs/optimizations/EXPLAIN-B1.md](docs/optimizations/EXPLAIN-B1.md)
- [docs/optimizations/EXPLAIN-B2.md](docs/optimizations/EXPLAIN-B2.md)
- [docs/optimizations/EXPLAIN-B3.md](docs/optimizations/EXPLAIN-B3.md)
- [docs/optimizations/EXPLAIN-B4.md](docs/optimizations/EXPLAIN-B4.md)

Experiments and report:

- [docs/EXPERIMENTS/README.md](docs/EXPERIMENTS/README.md)
- [docs/report/report.md](docs/report/report.md)

## Current Caveats

- The Berlin52 timing table contains NA cells inherited from a sweep that ran while the SLURM build-skip guard was still active. Removing that guard (May 2026) means the next sweep on `data/berlin52.tsp` will populate the missing columns. See the [Timing caveat](#berlin52) note for full context.
- The B1–B4 family ties on Berlin52 quality (best=8777, avg=9234.55, std=195.84) because each B-step is a strict superset of the previous one's optimizations applied to the same GA, with the same RNG seed sequence. Identical statistics across them is the *intended* correctness invariant of the optimization story (changing how the kernel runs without changing what it computes), not evidence that the binaries are colliding. B5 differs because `BLOCK_POP_SIZE = 512` (vs `32` for B1–B4), giving a 16× larger total population.
- `gpu_naive` reports the CPU greedy nearest-neighbor tour length; it does not run a GA and is not directly comparable to the GA variants on solution quality.
- `result_writer.h` is wired into B1–B5 and C1–C5 source code so that future sweeps can bypass stdout-grep extraction. It activates only when the runner sets `RESULT_CSV` and `RUN_ID`; that runner integration is not yet done, so the helper is currently inert and the legacy stdout-grep path is what's in use.
- The sequential C parser and the CUDA parser do not yet use identical TSPLIB distance semantics for every TSPLIB instance, so direct sequential-vs-CUDA comparisons on `berlin52.tsp` should be treated carefully.
- The root Makefile and the Slurm scripts are much closer now, but the experiment notes remain the most reliable source of truth for reruns.
