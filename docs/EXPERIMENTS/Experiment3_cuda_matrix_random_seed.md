# Experiment3: CUDA Matrix Sweep with Fresh Random Seeds

## Motivation

The controlled-seed matrix sweep is good for fair comparison, but it does not show how the implementations behave under fresh random initializations.

This experiment exists to answer:

- how much best-tour quality varies across random seeds,
- whether faster variants remain competitive under seed variation,
- and whether an apparent regression is stable or just seed-sensitive.

## What Was Run

Primary Slurm script:

- `slurm/run_cuda_all_variants_csv_randomseed.slurm`

Committed result artifact:

- `results/cuda_all_variants_randomseed_berlin52_5529311.csv`

This script draws a new seed for every run, using `/dev/urandom` when available.

## Has It Been Run?

Yes.

The committed berlin52 random-seed CSV contains `10` runs per implementation.

## Current Findings

Summarizing the committed berlin52 random-seed CSV:

- `CUDA-GA` ranged from `7640` to `8106`, with average best length about `7815.8`
- `GPU-Naive` stayed fixed at `8181`
- GPU-pop control, bank-conflict, bitset, AoS, global-dist, verbose-comments, B1, and B2 all ranged from `9097` to `9713`, with average best length about `9414.4`
- `parallel_sort`, `B3-reduce`, `B3-shuffle`, `B4-global`, and `B4-smem` improved on that cluster, ranging from `8883` to `9706`, with average best length about `9334.6`

Timing from the same committed CSV shows:

- `CUDA-GA` averages about `0.507 s`
- `GPU-Naive` averages about `0.102 s`
- GPU-pop control averages about `0.191 s`
- `B3-shuffle` averages about `0.140 s`
- `B4-global` averages about `0.141 s`

Interpretation:

- the hybrid `CUDA-GA` remains clearly strongest on tour quality for berlin52,
- `B3`/`B4` appear faster than the GPU-pop control,
- and the random-seed sweep suggests they also improve average berlin52 quality relative to the control family, though they still remain far behind `CUDA-GA`.

## Caveats

- This committed random-seed artifact exists only for berlin52, not smoke20.
- Same-seed equivalence across algorithm families is not expected, so the meaningful comparison is distribution over runs, not path-by-path equality.
- This experiment measures variation under changing seeds, not deterministic repeatability.

## How To Run

Berlin52 example:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,RUNS=10 slurm/run_cuda_all_variants_csv_randomseed.slurm
```

Smoke20 example:

```bash
sbatch --export=ALL,DATASET=sequential/tests/fixtures/smoke_20.tsp,RUNS=10 slurm/run_cuda_all_variants_csv_randomseed.slurm
```

Useful overrides:

- `RUNS=<n>` to change repeat count
- `ISLANDS=<n>` to change island count
- `GENERATIONS=<n>` to change generation budget

## Recommended Use

Use this experiment when you want:

- a seed-variability view of the CUDA variants,
- a better sense of average behavior than a single deterministic run,
- or evidence for whether a quality difference is robust across random restarts.