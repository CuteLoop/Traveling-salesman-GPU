# Challenges

## Incomplete Best-Tour Artifacts In Variant Runs

### Problem

The main CUDA matrix workflow was easy to run, but some pulled optimization-variant result files were missing the printed best tour even when they contained:

- best tour length
- runtime
- exit status

That meant correctness auditing could not fully validate those runs, because the route itself was missing.

### Why This Mattered

Without the best tour, the audit harness could not:

- verify every city appeared exactly once,
- recompute the canonical tour length,
- or distinguish a valid run from an incomplete artifact.

This made the main experiment pipeline and the correctness workflow diverge more than necessary.

### Fix

Instead of replacing the main pipeline, the fix was folded back into it.

`slurm/run_cuda_all_variants.slurm` now remains the primary workflow and automatically produces:

- one `.txt` file per implementation repeat,
- one `_best_tour.csv` per implementation repeat when a tour is printed,
- one raw summary CSV with one row per implementation repeat,
- one aggregate CSV with per-implementation:
  - mean runtime,
  - runtime standard deviation,
  - mean best length,
  - best-length standard deviation,
  - minimum best length,
  - maximum best length,
  - best-run repeat id,
  - best-run output path,
  - best-run tour CSV path.

### Workflow Decision

The workflows are now intentionally separate:

1. Main pipeline:
   `slurm/run_cuda_all_variants.slurm`
   This is the default experiment path and should stay simple.

2. Determinism / validation pipeline:
   `scripts/run_seed_benchmark.py` and `slurm/run_seed_benchmark.slurm`
   This is for fixed-seed repeatability and audit experiments. It is not the default benchmark path.

### Result

The normal matrix run stays simple, while still always producing:

- best tour when available,
- best length,
- runtime,
- average over runs,
- standard deviation over runs.