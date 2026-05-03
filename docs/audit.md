**Phase 1: Makefile sbatch target****Phase 2: SLURM runner per-variant invocations****Phase 3: Binary argv contracts and what the C-series actually does****Phase 4: Stdout extractor patterns vs. what each variant prints**Let me look at `build_cuda_variants` and check the SLURM-vs-Makefile defaults divergence:# End-to-end audit: `make sbatch_cuda_berlin52` → variant binaries → CSV

I traced the flow through Makefile → SLURM script → each `.cu` argv contract → stdout printing → awk extractor → CSV aggregator. Findings ranked by severity.

## What's correct (verified)

- `make sbatch_cuda_berlin52` exports all `CUDA_MATRIX_*` env vars correctly to the SLURM script.
- All 21 binary invocations in `run_cuda_all_variants_csv.slurm` match each binary's `parse_config` argv contract: `cuda_ga`/`cuda_ga_no_greedy` get `population_size` at argv[2]; B/C/gpu_pop family get `islands` at argv[2]; both groups share `generations`, `mutation_rate`, `elite_count`, `seed` at argv[3..6].
- CSV header has 18 fields; `run_and_record` writes 18 fields per row; the aggregator's awk indices ($1, $3, $4, $12, $13, $14, $15, $16, $17) correspond correctly to header columns.
- `build_cuda_variants` retries with `rm -rf build` on failure — handles transient NFS "Stale file handle" errors I saw in the `.error` logs.
- `awk` length-extraction patterns now cover all four output formats: `Best GPU-population GA tour length` (gpu_pop family), `Best GA tour length` (cuda_ga), `Best tour length` (B/C series), `^Tour length:` (gpu_naive greedy).
- The B4-smem skip-on-n>99 logic correctly writes a `skipped` CSV row instead of running a binary that would throw on `n > MAX_CITIES`.
- Dimension extraction returns `20` for `smoke_20.tsp`, `52` for `berlin52.tsp` — B4-smem runs in both.

## Real bugs (will affect a run)

**A. SLURM script defaults diverge from Makefile defaults.**

`slurm/run_cuda_all_variants_csv.slurm` has its own fallback values that *don't match* the standardized config:

| Param | Makefile | SLURM fallback |
|-------|---------:|---------------:|
| `RUNS` | 20 | 10 |
| `ISLANDS` | 256 | 128 |
| `GENERATIONS` | 2000 | 1000 |
| `MUTATION` | 0.05 | 0.05 |
| `ELITE_POP` | 2 | 2 |
| `SEED_BASE` | 100 | 42 |

If anyone calls `sbatch slurm/run_cuda_all_variants_csv.slurm` directly without env vars (e.g. for a quick repro), they get a smaller, differently-seeded sweep than what `make sbatch_*` produces. Result CSVs from those two paths cannot be compared. Fix: align the SLURM defaults to the Makefile's standardized values (or hard-fail if the env vars aren't set).

**B. `generations_completed` column is always NA for chrono-timing variants.**

The runner extracts via `awk '/^Generations completed:/'`. Only the cudaEvent-timing variants (`gpu_pop`, `bankconflict`, `bitset`, `global_dist`) actually print that line. The chrono-timing variants — `cuda_ga`, `cuda_ga_no_greedy`, `AoS`, `ParallelSort`, `VerboseComments`, all of B1–B5, all of C1–C5 — don't print it, so the column is NA for them. They actually run to completion (loop is `for (gen = 0; gen < cfg.generations; ++gen)` with no early exit), so the value is *known* to equal the requested generations count. Fix: extractor should fall back to the requested-generations column when the printed line is missing, or the chrono variants should also emit it.

**C. `gpu_naive` timing column is meaningless.**

`gpu_naive` doesn't run a GA — it computes the CPU greedy NN tour, then runs a tiny CUDA kernel that evaluates a 4-tour batch for cross-checking. The "CUDA kernel elapsed ms" it reports times only that test eval (microseconds), not any optimization. Including it in the same timing column as the GA variants suggests it's a fast GA implementation, which it isn't. Fix: either exclude `gpu_naive` timing from the result tables, or annotate it as "eval only" in the README.

## Quality issues (not breaking, worth hardening)

**D. B5 has dead `greedy_nn_kernel` code.**

`src/cuda/variants/CUDA-GA-B5-bigpop.cu` defines `greedy_nn_kernel` (lines 177–225) but `run_gpu_population_ga` never invokes it. B5 actually uses pure random Fisher-Yates init. Either delete the dead kernel or wire it (which would change the optimization story for B5).

**E. C-family "NN seeding" is single-thread per island.**

In every C variant (C1–C5), only `if (tid == 0)` copies the NN tour into `current[0]` with 2–9 swaps. The other `tid > 0` threads use random Fisher-Yates. So:

| Variant | Total individuals/island | NN-seeded |
|---------|------------------------:|----------:|
| C1–C4 | 32 | 1 (≈3.1%) |
| C5 | 512 | 1 (≈0.2%) |

Calling C "NN seeded" in the optimization story is technically true but overstates the effect — it's "1 strong elite injected into an otherwise random population" rather than "population seeded near the NN tour." For C5 specifically, that 0.2% NN concentration is so dilute it may not differentiate from B5. Worth verifying experimentally before claiming C5 > B5 in the report.

**F. No pre-flight binary existence check.**

After `build_cuda_variants` returns, the run loop doesn't verify all 21 binaries are present. A partial build failure (e.g., one variant fails to compile but the rest succeed and `make` returns 0 because of dependency parallelism quirks) would silently produce `failed_127` rows in the CSV. Adding a simple existence check before the run loop:

```bash
required=( build/GPU-Naive build/CUDA-GA build/CUDA-GA-no-greedy
           build/CUDA-GA-GPU-Pop build/CUDA-GA-GPU-Pop-bankconflict ... )
for b in "${required[@]}"; do
  [ -x "$b" ] || { echo "Missing binary: $b"; exit 1; }
done
```

would catch this at the source.

**G. CSV cells could be broken by future stdout changes.**

`version` and `notes` columns come unsanitized from the binary's stdout (modulo `tr ',' ';'` for notes on failure). If a future variant prints a comma in its `VERSION:` line, the CSV becomes ambiguous. Currently safe, fragile.

**H. Single-run scripts (`run_cuda_ga.slurm` etc.) still have build-skip guards.**

The fix only addressed the all-variants runners. Single-variant runners like `slurm/run_cuda_ga.slurm` still have `if [ ! -x ./build/CUDA-GA ]; then ...`. If you `git pull` source changes and run one of those, you get the stale binary. Lower priority because they only build one binary anyway, but the same anti-pattern is present.

## Things this audit did NOT verify

- I didn't run any binary; I didn't inspect actual nvcc compile output. If a binary fails to compile due to source mistakes added by my earlier edits, this audit can't see that.
- I didn't run `awk` against synthetic fixtures to verify pattern matching empirically. The assertions about pattern matches are based on reading source and the awk regex.
- I didn't audit the Sequential C side (the user's question was about the CUDA flow specifically).
- I didn't audit the `scripts/generate_cuda_results_readme.py` aggregation (the script that `make refresh_readme_cuda_results` runs). That's a candidate for the next audit pass — it consumes the avg CSV and produces the README tables, so any field-index bug there would propagate to the report.

## Recommended next actions, in priority order

1. **Align SLURM defaults to the Makefile's standardized values** (Bug A) — five-line fix in `run_cuda_all_variants_csv.slurm`.
2. **Make the `generations_completed` extractor fall back to the requested count** when the print is missing (Bug B) — one extra awk fallback in `run_and_record`.
3. **Annotate `gpu_naive` in result tables as "evaluation kernel only"** (Bug C) — README change, no code.
4. **Add pre-flight binary existence check** (Quality F) — short bash snippet between `build_cuda_variants` and the run loop.
5. **Audit `scripts/generate_cuda_results_readme.py`** before trusting another `make refresh_readme_cuda_results` run.

Want me to push fixes for 1, 2, and 4 directly? They're all small and bounded.

Sources:
- [Makefile](computer://C%3A%5CUsers%5Cual-laptop%5CDocuments%5CTraveling-salesman-GPU%5CMakefile)
- [slurm/run_cuda_all_variants_csv.slurm](computer://C%3A%5CUsers%5Cual-laptop%5CDocuments%5CTraveling-salesman-GPU%5Cslurm%5Crun_cuda_all_variants_csv.slurm)
- [src/cuda/CUDA-GA.cu](computer://C%3A%5CUsers%5Cual-laptop%5CDocuments%5CTraveling-salesman-GPU%5Csrc%5Ccuda%5CCUDA-GA.cu)
- [src/cuda/variants/CUDA-GA-B1-stride.cu](computer://C%3A%5CUsers%5Cual-laptop%5CDocuments%5CTraveling-salesman-GPU%5Csrc%5Ccuda%5Cvariants%5CCUDA-GA-B1-stride.cu)
- [src/cuda/variants/CUDA-GA-B5-bigpop.cu](computer://C%3A%5CUsers%5Cual-laptop%5CDocuments%5CTraveling-salesman-GPU%5Csrc%5Ccuda%5Cvariants%5CCUDA-GA-B5-bigpop.cu)
- [optimizations/C5/CUDA-GA-C5-bigpop.cu](computer://C%3A%5CUsers%5Cual-laptop%5CDocuments%5CTraveling-salesman-GPU%5Coptimizations%5CC5%5CCUDA-GA-C5-bigpop.cu)