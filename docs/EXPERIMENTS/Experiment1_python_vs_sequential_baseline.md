# Experiment1: Python vs Sequential Baseline

## Motivation

Before comparing CUDA variants, the repository needed a simple baseline comparison between:

- the Python baseline GA,
- and the sequential C GA.

The purpose of this experiment is not GPU benchmarking. It is a local sanity check for:

- baseline tour quality,
- baseline runtime gap,
- and whether both implementations solve the same small instance reasonably.

## What Was Run

Primary script:

- `baselines/compare_python_vs_sequential_ga.py`

Committed result artifacts:

- `results/python_vs_sequential_compare.csv`
- `results/python_vs_sequential_compare.txt`
- `results/sequential_stats_5487207.csv`

## Has It Been Run?

Yes.

The committed `results/python_vs_sequential_compare.txt` file shows an executed comparison on `smoke_20.tsp` using:

- population `100`
- generations `200`
- mutation `0.1`
- elite `2`
- seed `42`

## Current Findings

Observed from `results/python_vs_sequential_compare.txt`:

- Python baseline distance: `75.776906`
- Python runtime: `41.956912 s`
- Sequential C distance: `77.492495`
- Sequential C runtime: `0.055615 s`

Interpretation:

- the sequential C implementation is dramatically faster on this local smoke20 run,
- but the Python baseline found a slightly shorter route,
- so this experiment is useful as a baseline quality vs runtime tradeoff reference.

## Caveats

- This is a local-PC experiment, not an HPC GPU result.
- It should not be mixed directly into CUDA timing claims.
- It is useful mainly as a baseline and sanity check.

## How To Run

From repository root:

```bash
python baselines/compare_python_vs_sequential_ga.py --tsp sequential/tests/fixtures/smoke_20.tsp --pop 100 --gen 200 --mutation 0.1 --elite 2 --seed 42
```

Optional sequential-only budgeted run:

```powershell
.\sequential\bin\ga-tsp.exe --instance .\sequential\tests\fixtures\smoke_20.tsp --pop 100 --gen 150000 --seed 42 --elites 2 --tk 3 --pc 0.9 --pm 0.1 --csv .\sequential\results_budget_1tenth.csv
```

## Recommended Use

Use this experiment when you want:

- a quick non-GPU baseline,
- a sanity check that the sequential implementation still runs,
- or a quality/runtime reference before moving to CUDA experiments.