# Experiment4: Runtime to Reach a Target Tour Length

## Motivation

Average runtime for a fixed generation budget is useful, but it does not answer a more practical question:

- how long does a variant need to reach a target quality threshold?

This experiment changes the framing from fixed-budget optimization to target-driven stopping.

That makes it useful for comparing variants when the real question is:

- which implementation reaches an acceptable tour length first,
- and how many generations it takes to get there.

## What Was Run

Primary Slurm script:

- `slurm/run_cuda_modified_target_avg.slurm`

Committed result artifact:

- `results/cuda_modified_target_runs_berlin52_5532193.csv`

The script targets the currently modified GPU-pop family only:

- `CUDA-GA-GPU-Pop`
- `CUDA-GA-GPU-Pop-bankconflict`
- `CUDA-GA-GPU-Pop-bitset`
- `GA-GPU-POP-GlobalDist`

## Has It Been Run?

Partially, yes.

The committed per-run CSV shows berlin52 runs with target length `8500`.

However, the script also writes an average CSV:

- `results/cuda_modified_target_avg_<dataset>_<jobid>.csv`

and that average CSV is not currently committed in the repository.

## Current Findings

From the committed per-run berlin52 CSV:

- all committed runs marked `target_reached = yes`
- `CUDA-GA-GPU-Pop` average reported length is about `8427.33`
- `CUDA-GA-GPU-Pop-bankconflict` average reported length is also about `8427.33`
- `GA-GPU-POP-GlobalDist` averages about `8479.5`
- `CUDA-GA-GPU-Pop-bitset` averages about `8304.5` in the committed rows

Timing/generation behavior in the committed artifact is notable:

- control and bank-conflict variants show kernel times on the order of a few thousand milliseconds
- `GlobalDist` is slower than control in the committed target-stop runs
- `bitset` shows extremely large elapsed and generation counts in the committed rows, which looks unusual enough to warrant a follow-up correctness and instrumentation check

Interpretation:

- this experiment is already useful for target-quality comparisons,
- but the committed bitset behavior should be treated as suspicious until rerun and verified.

## Caveats

- Only the per-run CSV is currently committed.
- The averaging output expected from the script is missing from the repository.
- The bitset target-stop numbers are large enough that they may indicate either a real slowdown in this mode or an instrumentation/reporting problem.

## How To Run

Berlin52 target-quality example:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,TARGET_LENGTH=8500,RUNS=10 slurm/run_cuda_modified_target_avg.slurm
```

To aim for the known berlin52 optimum instead:

```bash
sbatch --export=ALL,DATASET=data/berlin52.tsp,TARGET_LENGTH=7542,RUNS=10 slurm/run_cuda_modified_target_avg.slurm
```

Useful overrides:

- `RUNS=<n>` to change repeat count
- `SEED_BASE=<seed>` to shift the deterministic seed schedule
- `ISLANDS=<n>` to change island count
- `GENERATIONS=<n>` to change the per-launch generation chunk

## Recommended Use

Use this experiment when you want:

- quality-threshold-oriented benchmarking,
- average time-to-target comparisons,
- or a better answer to practical stopping-performance questions than fixed-budget timing alone.