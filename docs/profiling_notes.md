# Profiling Notes — `ga-tsp` Sequential Baseline

**Phase 10 · Baseline Benchmarking and Regression Lock**

---

## Windows / MinGW Limitations

The local development environment (GCC 15.2.0, WinLibs POSIX UCRT, Windows)
does not support `gprof` or `perf`.  MinGW also lacks `libasan`/`libubsan`
for AddressSanitizer.  Therefore:

- **Numeric regression locking** is performed locally (deterministic RNG, fixed seeds).
- **Final hotspot validation** must occur on a Linux/HPC environment (UA HPC cluster).

---

## Profiling Build

The Makefile provides a `profile` build mode:

```bash
make clean && make all BUILD=profile
```

This compiles with:

```
CFLAGS = -std=c99 -Wall -Wextra -O2 -g -fno-omit-frame-pointer
```

The `-fno-omit-frame-pointer` flag preserves the frame pointer for accurate
`perf` call stacks without sacrificing significant optimization.

---

## HPC Profiling Commands

Run these on the UA HPC cluster (or any Linux system with `perf`):

```bash
# 1. Build in profile mode
make clean && make all BUILD=profile

# 2. Record with perf (sampling profiler)
perf record -g ./bin/ga-tsp \
    --instance tests/fixtures/benchmark_100.tsp \
    --pop 500 --gen 1000 --seed 42 \
    --elites 5 --tk 2 --pc 0.9 --pm 0.1

# 3. Analyze
perf report

# Alternative: flat profile with gprof
gcc -O2 -pg -std=c99 -Wall -Wextra -Iinclude -lm -o ga_profile \
    src/main.c src/instance.c src/tour.c src/fitness.c src/rng.c \
    src/init.c src/selection.c src/crossover.c src/elitism.c \
    src/replacement.c src/mutation.c src/ga_stats.c src/ga_driver.c
./ga_profile --instance tests/fixtures/benchmark_100.tsp \
    --pop 500 --gen 1000 --seed 42
gprof ga_profile gmon.out | head -40
```

---

## Hypothesis

> **`population_evaluate(...)` and the inner distance-lookup loop should
> account for >60% of the total runtime.**

### Rationale

- `population_evaluate` iterates $N \times n$ distance lookups per generation.
- With $N = 500$, $n = 100$, $G = 1000$: that is $5 \times 10^7$ distance
  lookups just for evaluation.
- Selection, crossover, and mutation are comparatively lightweight ($O(N
  \times k)$, $O(N \times n)$, $O(N)$ respectively).

### If the hypothesis fails

If `population_evaluate` accounts for **less than 60%** of runtime:

1. **Investigate crossover** — OX1's boolean lookup may be cache-hostile for
   large $n$.
2. **Investigate memory allocation** — if `alloc_pop`/`free_pop` appear in
   the profile, the double-buffer strategy may need revision.
3. **Reassess GPU porting priorities** — the kernel targeted first should be
   the actual bottleneck, not the assumed one.

---

## Release Build

For the final optimized baseline (benchmarking, not profiling):

```bash
make clean && make all BUILD=release
```

Compiles with:

```
CFLAGS = -std=c99 -Wall -Wextra -O3 -march=native -DNDEBUG
```

Note: `-ffast-math` is intentionally omitted to ensure strict IEEE-754
compliance during baseline locking.  If future benchmarks need it, verify
that regression outputs do not deviate by more than `1e-6` before enabling.

---

*Phase 10 · `ga-tsp` · Joel Maldonado · University of Arizona*
