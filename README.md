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
- builds with `make all_cuda_versions` if needed
- runs the current CUDA matrix, including B1 through B5
- writes one per-run CSV and one averaged CSV

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

Headline committed results:

| Implementation | Environment / artifact | Runtime (s) | Best tour length |
|---|---|---:|---:|
| Python baseline | `results/python_vs_sequential_compare.txt` | 41.956912 | 75.776906 |
| Sequential C | `results/python_vs_sequential_compare.txt` | 0.055615 | 77.492495 |
| Sequential HPC | `results/sequential_5487207.txt` | 0.01 | 77.492495 |
| GPU-Naive | `results/gpu_naive_5487208.txt` | 0.13 | 80 |
| CUDA-GA hybrid | `results/cuda_ga_5487209.txt` | 0.39 | 75 |
| CUDA-GA GPU-pop | `results/cuda_ga_gpu_pop_5487210.txt` | 0.20 | 73 |

Best committed smoke_20 tour:

- implementation: `CUDA-GA-GPU-Pop`
- length: `73`

```text
0 -> 2 -> 1 -> 16 -> 17 -> 3 -> 7 -> 12 -> 11 -> 13 -> 14 -> 6 -> 5 -> 19 -> 4 -> 10 -> 8 -> 9 -> 18 -> 15 -> 0
```

### berlin52

Latest all-variants Berlin52 sweep checked in this repo:

- dataset: `data/berlin52.tsp`
- runner: `slurm/run_cuda_all_variants_csv.slurm`
- settings: `RUNS=20, ISLANDS=256, GENERATIONS=2000, MUTATION=0.03, ELITE_POP=2, SEED_BASE=100`
- artifacts:
  - `results/cuda_all_variants_runs_berlin52_5532233.csv`
  - `results/cuda_all_variants_avg_berlin52_5532233.csv`

Summary from that sweep:

Measured timing results from variants that currently export `CUDA kernel elapsed ms`:

| Variant | Mean CUDA time (ms) | Stddev (ms) | Mean reported length |
|---|---:|---:|---:|
| `cuda_ga_gpu_pop_global_dist` | 279.731 | 0.256 | 9125.40 |
| `cuda_ga_gpu_pop_bitset` | 284.177 | 0.341 | 9125.40 |
| `cuda_ga_gpu_pop_bankconflict` | 330.908 | 1.604 | 9125.40 |
| `cuda_ga_gpu_pop` | 334.398 | 4.323 | 9125.40 |

Quality summary across the full sweep:

| Variant family | Mean reported length | Timing status |
|---|---:|---|
| `cuda_ga_b5_bigpop` | 7542.00 | timing not yet exported |
| `cuda_ga` | 7795.45 | timing not yet exported |
| `gpu_naive` | 8181.00 | timing not yet exported in this CSV |
| `cuda_ga_gpu_pop_parallel_sort` | 9089.10 | timing not yet exported |
| GPU-pop control / bitset / bankconflict / AoS / GlobalDist / verbose | 9125.40 | partial timing coverage |
| B1 / B2 / B3 / B4 family | 9234.55 | timing not yet exported |

Interpretation:

- Among the variants with timing instrumentation, `cuda_ga_gpu_pop_global_dist` is fastest, followed closely by `cuda_ga_gpu_pop_bitset`.
- The speedups currently visible in the README are within the GPU-pop control family; they do not yet correspond to better Berlin52 tour quality.
- `cuda_ga_b5_bigpop` is the only variant in this sweep that consistently reached the Berlin52 optimum `7542`.
- The older GPU-pop optimization family improved runtime more than quality under these settings.
- The B1-B4 family currently looks like a kernel-optimization story, not a solution-quality story.

Timing caveat:

- This README should not be read as a complete runtime ranking across all variants yet. Several binaries still do not export `CUDA kernel elapsed ms`, so the timing table above is currently a measured subset, not the whole matrix.

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

- Not all variants export the same timing fields yet, so quality comparisons are more complete than runtime comparisons in the latest Berlin52 matrix.
- The B1-B4 family currently shares very similar quality outcomes on Berlin52; before claiming performance wins, verify that the intended search behavior actually differs.
- The sequential C parser and the CUDA parser do not yet use identical TSPLIB distance semantics for every TSPLIB instance, so direct sequential-vs-CUDA comparisons on `berlin52.tsp` should be treated carefully.
- The root Makefile and the Slurm scripts are much closer now, but the experiment notes remain the most reliable source of truth for reruns.
