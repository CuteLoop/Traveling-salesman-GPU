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
  - `results/cuda_all_variants_runs_smoke_20_5532375.csv`
  - `results/cuda_all_variants_avg_smoke_20_5532375.csv`

Per-implementation results from that sweep:

| Variant | Best length | Avg length | Length stddev | Best time (ms) | Avg time (ms) | Time stddev (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `gpu_naive (eval only)` | 80 | 80 | 0 | 0.126852 | 0.131514 | 0.006421 |
| `cuda_ga` | 74 | 74.65 | 0.48936 | 561.745 | 564.046 | 2.30094 |
| `cuda_ga_no_greedy` | 73 | 73.7 | 1.17429 | 561.436 | 564.856 | 2.27152 |
| `cuda_ga_gpu_pop` | 73 | 73 | 0 | 121.215 | 122.087 | 2.91467 |
| `cuda_ga_gpu_pop_bankconflict` | 73 | 73 | 0 | 120.112 | 120.466 | 0.300516 |
| `cuda_ga_gpu_pop_bitset` | 73 | 73 | 0 | 114.927 | 115.151 | 0.189846 |
| `cuda_ga_gpu_pop_parallel_sort` | 73 | 73 | 0 | 124.725 | 126.407 | 1.33541 |
| `cuda_ga_gpu_pop_aos` | 73 | 73 | 0 | 625.803 | 626.783 | 0.797835 |
| `cuda_ga_gpu_pop_global_dist` | 73 | 73 | 0 | 148.638 | 149.062 | 0.237839 |
| `cuda_ga_gpu_pop_verbose_comments` | 73 | 73 | 0 | 203.189 | 204.955 | 1.11084 |
| `cuda_ga_b1_stride` | 73 | 73 | 0 | 124.232 | 125.3 | 0.746125 |
| `cuda_ga_b2_bitmask` | 73 | 73 | 0 | 116.671 | 117.691 | 1.23082 |
| `cuda_ga_b3_reduce` | 73 | 73 | 0 | 116.535 | 117.871 | 0.975001 |
| `cuda_ga_b3_shuffle` | 73 | 73 | 0 | 116.133 | 117.557 | 0.99224 |
| `cuda_ga_b4_global` | 73 | 73 | 0 | 116.472 | 117.559 | 0.971294 |
| `cuda_ga_b5_bigpop` | 73 | 73 | 0 | 1683.7 | 1685.99 | 1.48049 |
| `cuda_ga_c1_stride` | 73 | 73 | 0 | 124.96 | 126.937 | 1.23231 |
| `cuda_ga_c2_bitmask` | 73 | 73 | 0 | 116.585 | 117.898 | 1.0843 |
| `cuda_ga_c3_reduce` | 73 | 73 | 0 | 116.731 | 117.974 | 0.954398 |
| `cuda_ga_c4_global` | 73 | 73 | 0 | 117.136 | 118.253 | 0.811 |
| `cuda_ga_c5_bigpop` | 73 | 73 | 0 | 1681.84 | 1686.39 | 1.9697 |
| `cuda_ga_b4_smem` | 73 | 73 | 0 | 116.749 | 118.626 | 1.20893 |

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
  - `results/cuda_all_variants_runs_berlin52_5532274.csv`
  - `results/cuda_all_variants_avg_berlin52_5532274.csv`

Per-implementation results from that sweep:

| Variant | Best length | Avg length | Length stddev | Best time (ms) | Avg time (ms) | Time stddev (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `gpu_naive (eval only)` | 8181 | 8181 | 0 | 0.117045 | 0.132069 | 0.005729 |
| `cuda_ga` | 7604 | 7795.45 | 208.35 | 873.534 | 878.081 | 3.93289 |
| `cuda_ga_no_greedy` | 7718 | 8275.75 | 226.052 | 888.723 | 893.529 | 2.90515 |
| `cuda_ga_gpu_pop` | 8929 | 9125.4 | 120.514 | 332.4 | 334.779 | 3.68133 |
| `cuda_ga_gpu_pop_bankconflict` | 8929 | 9125.4 | 120.514 | 328.022 | 330.536 | 1.09669 |
| `cuda_ga_gpu_pop_bitset` | 8929 | 9125.4 | 120.514 | 283.716 | 283.998 | 0.215688 |
| `cuda_ga_gpu_pop_parallel_sort` | 8836 | 9089.1 | 126.029 | 262.054 | 264.198 | 1.43638 |
| `cuda_ga_gpu_pop_aos` | 8929 | 9125.4 | 120.514 | 823.995 | 825.036 | 0.975611 |
| `cuda_ga_gpu_pop_global_dist` | 8929 | 9125.4 | 120.514 | 279.251 | 279.633 | 0.244034 |
| `cuda_ga_gpu_pop_verbose_comments` | 8929 | 9125.4 | 120.514 | 416.183 | 418.509 | 1.34174 |
| `cuda_ga_b1_stride` | 8777 | 9234.55 | 195.84 | 260.978 | 262.716 | 0.847132 |
| `cuda_ga_b2_bitmask` | 8777 | 9234.55 | 195.84 | 246.046 | 247.056 | 0.748003 |
| `cuda_ga_b3_reduce` | 8777 | 9234.55 | 195.84 | 246.055 | 247.162 | 0.773716 |
| `cuda_ga_b3_shuffle` | 8777 | 9234.55 | 195.84 | 245.797 | 247.297 | 0.888541 |
| `cuda_ga_b4_global` | 8777 | 9234.55 | 195.84 | 245.997 | 247.04 | 0.692264 |
| `cuda_ga_b5_bigpop` | 7542 | 7590.1 | 66.7288 | 7484.76 | 7513.55 | 25.3839 |
| `cuda_ga_c1_stride` | 7909 | 8079.4 | 112.122 | 261.281 | 306.977 | 161.502 |
| `cuda_ga_c2_bitmask` | 7909 | 8079.4 | 112.122 | 245.709 | 246.705 | 0.891915 |
| `cuda_ga_c3_reduce` | 7909 | 8079.4 | 112.122 | 245.692 | 246.991 | 1.55146 |
| `cuda_ga_c4_global` | 7909 | 8079.4 | 112.122 | 247.521 | 248.357 | 0.746895 |
| `cuda_ga_c5_bigpop` | 7542 | 7542 | 0 | 7490.05 | 7521.45 | 21.2939 |
| `cuda_ga_b4_smem` | 8777 | 9234.55 | 195.84 | 315.985 | 318.503 | 1.46895 |

Interpretation:

- `cuda_ga_b5_bigpop` is the best quality variant in the latest completed Berlin52 sweep and reaches the optimum `7542` consistently across runs.
- `cuda_ga_gpu_pop_global_dist` is the fastest timed CUDA variant on Berlin52 under the standardized settings, but it does not match the solution quality of `cuda_ga_b5_bigpop`.
- The current Berlin52 tradeoff is still quality versus speed, not one variant dominating both.

Timing caveat:

- The Berlin52 table above is populated end-to-end after the SLURM build-skip guard was removed and the sweep was rerun (job 5532274). Earlier completed CSVs (e.g. 5532233) had `NA` timing for many variants; those came from stale binaries that pre-dated the timing prints in the source — see [Current Caveats](#current-caveats).
- `gpu_naive (eval only)` remains in the table as a correctness reference, but its timing measures only a tiny verification kernel and is not comparable to the GA variants.
- The newest `cuda_ga_no_greedy` and `cuda_ga_c1` through `cuda_ga_c5` variants now appear in both tables.

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

- The B1–B4 family ties on Berlin52 quality (best=8777, avg=9234.55, std=195.84) because each B-step is a strict superset of the previous one's optimizations applied to the same GA with the same RNG seed sequence. Identical quality statistics across them is the *intended* correctness invariant of the optimization story — the kernel changes how it runs without changing what it computes — not evidence that the binaries are colliding. Their timing values do diverge run-to-run as the optimizations bite: `b1_stride` lands around `260 ms`, `b4_global` around `246 ms`. B5 differs in both quality and timing because `BLOCK_POP_SIZE = 512` (vs `32` for B1–B4), giving a 16× larger per-island population that converges to optimum 7542 every run, at the cost of much longer per-island compute.
- The C-family "NN seeding" is precise: each island has its `tid == 0` thread copy the global-best NN tour into `current[0]` and apply 2–9 random swaps; all other `tid > 0` threads use random Fisher-Yates. So C1–C4 inject 1 NN-seeded individual out of 32 (≈3.1%) and C5 injects 1 out of 512 (≈0.2%). C5's quality on Berlin52 (avg=7542, std=0) shows the seed is enough to dominate via elitism, but the percentage is small.
- `gpu_naive` reports the CPU greedy nearest-neighbor tour length and is labeled `gpu_naive (eval only)` in the result tables; its CUDA timing measures only a verification kernel and is not directly comparable to the GA variants.
- `result_writer.h` is wired into B1–B5 and C1–C5 source so future sweeps can bypass stdout-grep extraction. It activates only when the runner sets `RESULT_CSV` and `RUN_ID`. That runner integration is not yet done — the helper is currently inert and the legacy stdout-grep path is what's in use. The legacy path is reliable now that all SLURM runners always invoke `make` (no more stale-binary skips).
- B5 ships a `greedy_nn_kernel` definition in `src/cuda/variants/CUDA-GA-B5-bigpop.cu` that is never invoked. It is dead code — B5 actually uses pure random Fisher-Yates init across all threads. The kernel was scaffolding inherited from the C5 line and has been left in place pending a decision to delete or wire.
- The sequential C parser and the CUDA parser do not yet use identical TSPLIB distance semantics for every TSPLIB instance, so direct sequential-vs-CUDA comparisons on `berlin52.tsp` should be treated carefully.
- The root Makefile and the Slurm scripts are much closer now, but the experiment notes remain the most reliable source of truth for reruns.

### SLURM build-skip guards (fixed)

Every SLURM script that invokes a built binary now defers to `make` rather than short-circuiting on binary existence:

- `slurm/run_cuda_all_variants_csv.slurm` — `build_cuda_variants` (with retry on transient NFS failures) followed by `verify_required_binaries` (pre-flight check that all 21 expected binaries exist).
- `slurm/run_cuda_all_variants.slurm` — unconditional `make all_cuda_versions`.
- `slurm/run_cuda_ga.slurm`, `slurm/run_cuda_ga_gpu_pop.slurm`, `slurm/run_cuda_ga_gpu_pop_bankconflict.slurm`, `slurm/run_cuda_ga_gpu_pop_bitset.slurm`, `slurm/run_gpu_naive.slurm`, `slurm/run_sequential.slurm` — unconditional `make build/<target>`.

The previous `if [ ! -x ./build/<target> ]; then make` pattern caused stale binaries to silently survive source edits — for example, when a `CUDA kernel elapsed ms:` print was added to a variant's source after its binary had been cached, the SLURM sweep kept running the old binary and the stdout-grep extractor produced NA in the resulting CSV. With the guard gone, `make`'s own dependency tracking incrementally rebuilds only the variants whose source has changed.
