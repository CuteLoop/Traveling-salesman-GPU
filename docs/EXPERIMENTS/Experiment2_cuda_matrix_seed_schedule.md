# Experiment2: CUDA Matrix Sweep with Controlled Seed Schedule

## Motivation

This experiment compares many CUDA implementations in one batch while keeping the random-seed schedule controlled.

The main goal is to answer:

- which implementation families are fastest,
- which ones produce the shortest tours,
- and which optimization variants appear to preserve or change search quality.

Unlike a single fixed-seed run, this workflow steps through a known seed sequence:

- run 1 uses `SEED_BASE`,
- run 2 uses `SEED_BASE + 1`,
- and so on.

That makes it easier to compare implementations over the same seed pool.

## What Was Run

Primary Slurm script:

- `slurm/run_cuda_all_variants_csv.slurm`

Committed result artifacts:

- `results/cuda_all_variants_smoke_20_5527277.csv`
- `results/cuda_all_variants_berlin52_5527278.csv`

The script covers:

- `GPU-Naive`
- `CUDA-GA`
- `CUDA-GA-GPU-Pop`
- bank-conflict, bitset, parallel-sort, AoS, global-dist, verbose-comments variants
- B1, B2, B3, and B4 variants

## Has It Been Run?

Yes.

Two committed matrix CSVs show this experiment was run on:

- `smoke_20.tsp`
- `berlin52.tsp`

## Current Findings

### smoke_20

From the committed matrix CSV:

- `GPU-Naive` reports length `80`
- `CUDA-GA` reports length `75`
- the GPU-pop control and several early variants report length `73`

This suggests the GPU-pop family is already competitive on the small smoke20 instance.

### berlin52

From the committed berlin52 matrix CSV:

- `GPU-Naive` reports `8181`
- `CUDA-GA` reaches `7542` on at least one run
- GPU-pop control and nearby variants cluster around `9412`
- faster parallel-sort/B3/B4 style variants are faster in time, but committed result capture for some B-variants is incomplete

This supports the current interpretation that:

- `CUDA-GA` and the GPU-pop family are different search behaviors,
- and some later GPU-pop optimizations may also change search semantics, not only speed.

## Caveats

- The committed CSV format still shows an exporter issue in the timing columns for these older runs: `elapsed_seconds` and `max_rss_kb` are partially merged in the output.
- In the committed matrix CSVs, some B1/B2/B3/B4 rows also have missing `reported_length` values even though timings were recorded.
- Treat these artifacts as useful but not final-quality benchmark tables.

## How To Run

Smoke20 example:

```bash
sbatch --export=ALL,DATASET=sequential/tests/fixtures/smoke_20.tsp,RUNS=10 slurm/run_cuda_all_variants_csv.slurm
```

Berlin52 example:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,RUNS=10 slurm/run_cuda_all_variants_csv.slurm
```

Useful overrides:

- `RUNS=<n>` to change repeat count
- `SEED_BASE=<seed>` to shift the deterministic seed schedule
- `ISLANDS=<n>` to change the GPU-pop island count
- `GENERATIONS=<n>` to change generation budget

## Recommended Use

Use this experiment when you want:

- a broad implementation sweep,
- controlled cross-implementation seed comparison,
- or a first-pass performance and quality table for many CUDA variants at once.