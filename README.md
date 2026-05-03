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
  - `results/cuda_all_variants_runs_smoke_20_5532258.csv`
  - `results/cuda_all_variants_avg_smoke_20_5532258.csv`

Per-implementation results from that sweep:

| Variant | Best length | Avg length | Length stddev | Best time (ms) | Avg time (ms) | Time stddev (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `gpu_naive (eval only)` | 80 | 80 | 0 | 0.123908 | 0.13027 | 0.005707 |
| `cuda_ga` | 74 | 74.65 | 0.48936 | 559.206 | 561.685 | 2.18129 |
| `cuda_ga_no_greedy` | 73 | 73.7 | 1.17429 | 559.359 | 564.083 | 6.09395 |
| `cuda_ga_gpu_pop` | 73 | 73 | 0 | 121.253 | 122.043 | 2.8705 |
| `cuda_ga_gpu_pop_bankconflict` | 73 | 73 | 0 | 120.223 | 120.467 | 0.315725 |
| `cuda_ga_gpu_pop_bitset` | 73 | 73 | 0 | 114.945 | 115.121 | 0.091226 |
| `cuda_ga_gpu_pop_parallel_sort` | 73 | 73 | 0 | 124.759 | 125.496 | 0.468964 |
| `cuda_ga_gpu_pop_aos` | 73 | 73 | 0 | 625.337 | 626.005 | 0.382793 |
| `cuda_ga_gpu_pop_global_dist` | 73 | 73 | 0 | 148.799 | 149.132 | 0.215895 |
| `cuda_ga_gpu_pop_verbose_comments` | 73 | 73 | 0 | 203.256 | 204.368 | 0.72499 |
| `cuda_ga_b1_stride` | 73 | 73 | 0 | 124.328 | 125.272 | 0.658154 |
| `cuda_ga_b2_bitmask` | 73 | 73 | 0 | 116.473 | 117.41 | 0.581083 |
| `cuda_ga_b3_reduce` | 73 | 73 | 0 | 115.818 | 117.327 | 0.642565 |
| `cuda_ga_b3_shuffle` | 73 | 73 | 0 | 116.367 | 117.497 | 0.673584 |
| `cuda_ga_b4_global` | 73 | 73 | 0 | 116.516 | 117.231 | 0.520973 |
| `cuda_ga_b5_bigpop` | 73 | 73 | 0 | 1681.68 | 1683.5 | 1.52303 |
| `cuda_ga_c1_stride` | 73 | 73 | 0 | 125.343 | 126.115 | 0.737526 |
| `cuda_ga_c2_bitmask` | 73 | 73 | 0 | 116.27 | 117.256 | 0.503559 |
| `cuda_ga_c3_reduce` | 73 | 73 | 0 | 116.239 | 117.369 | 0.587855 |
| `cuda_ga_c4_global` | 73 | 73 | 0 | 117.019 | 117.825 | 0.604088 |
| `cuda_ga_c5_bigpop` | 73 | 73 | 0 | 1679.8 | 1683.7 | 1.4726 |
| `cuda_ga_b4_smem` | 73 | 73 | 0 | 115.99 | 117.644 | 1.43144 |

Notes:

- Many variants tie at the smoke_20 optimum length `73`, so the timing columns are what separate them under this configuration.
- `gpu_naive` is labeled `eval only` because its CUDA timing measures a tiny verification kernel, not a GA optimization run.
- On smoke_20, `cuda_ga_gpu_pop_bitset` is the fastest timed variant among the implementations that also hit the best observed length.

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
| `gpu_naive (eval only)` | 8181 | 8181 | 0 | NA | NA | NA |
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

- `cuda_ga_b5_bigpop` is the best quality variant in the latest completed Berlin52 sweep and reaches the optimum `7542` consistently across runs.
- `cuda_ga_gpu_pop_global_dist` is the fastest timed CUDA variant on Berlin52 under the standardized settings, but it does not match the solution quality of `cuda_ga_b5_bigpop`.
- The current Berlin52 tradeoff is still quality versus speed, not one variant dominating both.

Timing caveat:

- `NA` timing cells in these tables come from historical completed result files that were generated before those binaries were rerun with the now-standard timing export.
- `gpu_naive (eval only)` remains in the table as a correctness reference, but its timing is not comparable to the GA variant timings.
- The newest `cuda_ga_no_greedy` and `cuda_ga_c1` through `cuda_ga_c5` variants will appear automatically once a completed standardized sweep writes non-empty run and average CSVs for them.

## Optimization Story

The repo currently has one implemented optimization story family:

- B1: stride padding in shared memory
- B2: bitmask-based OX bookkeeping
- B3-reduce / B3-shuffle: faster elite selection
- B4-global / B4-smem: distance-memory layout experiments
- B5-bigpop: large 512-individual island layout in global memory

A useful way to think about the current result split is:

- B1-B4 mostly change how the same search is executed.
- B5 changes how the search starts.

Current hypothesis for why B5 is much better on Berlin52:

1. B5 does not use greedy seeding; it still starts from random permutations like the rest of the B-family.
2. The main search-behavior change in B5 is population scale: each island grows to `512` individuals, which increases within-island diversity and gives tournament selection and elitism a much stronger pool to work with.
3. On Berlin52 with `2000` generations and `256` islands, that larger search budget appears to matter more than the earlier kernel-level optimizations.
4. The near-identical B1-B4 aggregate results still suggest those variants are mostly changing throughput on the same search process rather than changing the search trajectory itself.

So the current evidence does not say B1-B4 are useless. It says their optimizations are mostly throughput optimizations, while B5 is the first B-family variant that materially changes search capacity.

Planned next framing for the optimization story:

- B1-B5: kernel and memory optimization family
- C1-C5: same progression, but each island injects one greedy nearest-neighbor seeded elite and fills the rest of the population randomly

That split would make the comparison cleaner:
- B-family answers: how much do kernel optimizations help if search behavior stays roughly the same?
- C-family answers: what changes when each island gets one strong greedy seed without otherwise abandoning random population diversity?

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
