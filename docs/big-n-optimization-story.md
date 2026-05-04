## Optimizing for Unbounded n — Full Design Analysis

Let me work through each dimension systematically.

---

## The Fundamental Scaling Problem

For the GA kernel, three quantities scale with n:

```
Tour storage (population):   BLOCK_POP_SIZE × n × 4 bytes per buffer × 2 buffers
Distance matrix:             n² × 4 bytes
Working scratch:             fixed (lengths, order, s_red — independent of n)

At n=128:  pop = 2×32×128×4 = 32KB,  dist = 64KB   → 96KB total (exceeds 64KB smem)
At n=256:  pop = 2×32×256×4 = 64KB,  dist = 256KB  → 320KB (impossible in smem)
At n=1000: pop = 2×32×1000×4 = 250KB, dist = 3.9MB → nowhere near smem
```

**Conclusion**: for large n, neither the population nor the distance matrix fits in shared memory. Both must live in global memory. The optimization story becomes about making global memory access patterns as efficient as possible.

---

## Memory Architecture for Large n

The only viable layout:

```
Global memory (16GB on P100):
  d_pop_a[islands × BLOCK_POP_SIZE × n]    ← current gen population
  d_pop_b[islands × BLOCK_POP_SIZE × n]    ← next gen population  
  d_dist[n × n]                            ← distance matrix

Shared memory (64KB):
  lengths[BLOCK_POP_SIZE]                  ← fitness values
  order[BLOCK_POP_SIZE]                    ← top-k indices
  s_red[BLOCK_POP_SIZE]                    ← reduction scratch
  tile_dist[TILE × TILE]                   ← distance tile cache (see below)
  tile_tour[BLOCK_POP_SIZE × TILE]         ← tour segment cache (see below)

Registers (per thread):
  rng state                                ← 1 register
  local computation variables              ← ~10-20 registers
  possibly: hot tour segment               ← TILE registers (see below)
```

---

## Population Tiling

The key insight: you don't need the entire tour in memory simultaneously. Tour evaluation, crossover, and mutation all process tours **sequentially in segments**. This enables tiling.

### Tour Evaluation Tiling

Tour length evaluation accesses `tour[k]` and `tour[k+1]` for `k = 0..n-1`. This is a sequential scan — perfect for tiling.

```cpp
// Instead of loading full tour, process TILE cities at a time
// Each thread evaluates its own tour but only holds TILE cities at once

constexpr int TILE = 32;  // fits in registers

__device__ int tour_length_tiled(
    const int* d_pop,      // global memory population
    int individual_idx,
    int n,
    const int* __restrict__ d_dist)
{
    const int base = individual_idx * n;
    int total = 0;
    int prev_city = d_pop[base + n - 1];  // last city (for wrap-around)

    for (int tile_start = 0; tile_start < n; tile_start += TILE) {
        // Load TILE cities into registers
        int tile[TILE];
        int tile_len = min(TILE, n - tile_start);
        for (int i = 0; i < tile_len; ++i)
            tile[i] = d_pop[base + tile_start + i];

        // Evaluate edges within this tile
        // First edge: prev_city → tile[0]
        total += d_dist[prev_city * n + tile[0]];
        for (int i = 1; i < tile_len; ++i)
            total += d_dist[tile[i-1] * n + tile[i]];

        prev_city = tile[tile_len - 1];
    }
    return total;
}
```

This keeps only `TILE` cities in registers at once. The global memory reads for the tour are sequential and **coalesced** if consecutive threads read consecutive individuals — which they do when the population is laid out as `[individual][city]` (SoA layout discussed below).

### Crossover Tiling

OX crossover is harder to tile because it has a random cut point. However, the **used-city tracking** (currently `int used[n]` causing lmem spill for large n) becomes a register bitmask problem that doesn't scale beyond n=128 with 4 uint32s.

For large n, the bitmask approach requires:
```
n=128:  4 × uint32   = 4 registers    ← current B2 approach
n=256:  8 × uint32   = 8 registers    ← still fine
n=512:  16 × uint32  = 16 registers   ← acceptable
n=1024: 32 × uint32  = 32 registers   ← getting heavy but works
n=2048: 64 × uint32  = 64 registers   ← borderline
```

The bitmask scales linearly with n in register usage, which is much better than the `int used[n]` array that scales linearly in **local memory** (DRAM). The bitmask approach remains the right answer for all practical n.

---

## Distance Matrix Access Patterns and L2 Optimization

For large n, the full distance matrix doesn't fit in smem. But L2 (4MB on P100) can cache a useful fraction:

```
n=128:   dist = 64KB    →  1.6% of L2   (fits entirely, always warm)
n=256:   dist = 256KB   →  6.4% of L2   (fits entirely)
n=512:   dist = 1MB     →  25% of L2    (fits, but competing with pop)
n=1024:  dist = 4MB     →  100% of L2   (exactly fills L2 — eviction likely)
n=2048:  dist = 16MB    →  400% of L2   (thrashing)
```

**L2 is only effective up to approximately n=700** before eviction becomes problematic. Beyond that you need a different strategy.

### Smem Distance Tile Cache

For any n, you can cache a **tile of the distance matrix** in shared memory to exploit locality in tour evaluation:

```
Observation: when evaluating a tour segment [c0, c1, c2, ..., c_TILE],
the distance lookups are dist[c_i][c_{i+1}].
If consecutive cities in a tour tend to be geographically nearby
(true for good solutions), their rows in the distance matrix are
spatially clustered.

Strategy: for each tile of the tour, cooperatively load the relevant
rows of the distance matrix into smem before evaluating that tile.
```

```cpp
constexpr int DIST_TILE = 16;  // load 16 rows of dist matrix into smem

// In smem layout:
__shared__ int s_dist_tile[DIST_TILE][DIST_TILE + 1];  // +1 avoids bank conflicts
// s_dist_tile[i][j] = dist[city_set[i]][city_set[j]]

// Before evaluating a tour segment:
// 1. Identify which cities appear in this segment (city_set)
// 2. Cooperatively load their pairwise distances into s_dist_tile
// 3. Evaluate the segment using s_dist_tile lookups
// 4. Fall through to d_dist for distances not in the tile
```

The effectiveness depends on the **edge reuse histogram** from Experiment Set 5 — exactly what that experiment was designed to measure.

### `__ldg()` Read-Only Cache

For global dist with `__restrict__`, explicitly using `__ldg()` (load via texture cache) bypasses L1 and uses a separate 48KB read-only cache:

```cpp
// Instead of:
total += d_dist[a * n + b];

// Use:
total += __ldg(&d_dist[a * n + b]);
```

On P100, the read-only cache and L2 are separate paths. For scatter access patterns where L1 thrashes, `__ldg()` routing through the read-only cache can reduce L2 pressure. Worth measuring but not always a win.

---

## Population Layout: AoS vs SoA

This is a major optimization opportunity for large n that has not been fully explored.

### Current layout: Array of Structures (AoS) — individual-major

```
Memory: [ind0_city0][ind0_city1]...[ind0_city_n][ind1_city0][ind1_city1]...

Access in tour evaluation (thread t reads individual t):
  Thread 0: reads addresses 0, 1, 2, ..., n-1
  Thread 1: reads addresses n, n+1, ..., 2n-1
  Thread 2: reads addresses 2n, 2n+1, ..., 3n-1

Coalescing: POOR for warp-level reads
  When 32 threads simultaneously read city k of their respective tours,
  they access addresses k, k+n, k+2n, ..., k+31n
  These are stride-n apart — NOT coalesced (32 separate cache lines)
```

### Alternative layout: Structure of Arrays (SoA) — city-major

```
Memory: [ind0_city0][ind1_city0]...[ind31_city0][ind0_city1][ind1_city1]...

Access in tour evaluation (thread t reads individual t, city k):
  address = k * BLOCK_POP_SIZE + t

When 32 threads simultaneously read city k:
  Thread 0: address k*32 + 0
  Thread 1: address k*32 + 1
  ...
  Thread 31: address k*32 + 31
  → 32 consecutive addresses → PERFECTLY COALESCED (1 cache line)
```

**SoA wins decisively for tour evaluation at large n.** The cost is that crossover becomes more complex — reading a parent's full tour requires strided access (every 32nd element instead of every 1st). But crossover is called once per individual per generation, while tour evaluation iterates over n edges — so the evaluation pattern dominates at large n.

```cpp
// SoA access macro
#define POP_SoA(pop_base, island, individual, city, n, block_pop_size) \
    ((pop_base)[(island) * (block_pop_size) * (n) + (city) * (block_pop_size) + (individual)])

// Tour evaluation with SoA — fully coalesced warp reads
__device__ int tour_length_soa(
    const int* d_pop_soa,
    int island, int individual, int n,
    const int* __restrict__ d_dist)
{
    int total = 0;
    int prev = POP_SoA(d_pop_soa, island, individual, n-1, n, BLOCK_POP_SIZE);
    for (int k = 0; k < n; ++k) {
        int city = POP_SoA(d_pop_soa, island, individual, k, n, BLOCK_POP_SIZE);
        total += __ldg(&d_dist[prev * n + city]);
        prev = city;
    }
    return total;
}
```

---

## Warp-Level Distance Prefetching

For large n, distance lookups stall while waiting for global memory. You can hide this latency by **prefetching the next city's distance row** while computing the current edge:

```cpp
// Prefetch pattern for tour evaluation
__device__ int tour_length_prefetch(
    const int* d_pop, int base, int n,
    const int* __restrict__ d_dist)
{
    int total = 0;
    int city_curr = d_pop[base];
    int city_next = d_pop[base + 1];

    // Issue prefetch for first row before loop
    // (compiler hint — actual hardware prefetch via __ldg pipeline)
    const int* row_next = d_dist + city_next * n;

    for (int k = 0; k < n - 1; ++k) {
        int city_after = d_pop[base + k + 2];  // load k+2 while computing k
        const int* row_curr = d_dist + city_curr * n;

        total += row_curr[city_next];           // compute current edge

        city_curr = city_next;
        city_next = city_after;
        row_next  = d_dist + city_after * n;   // pointer ready for next iter
    }
    // Handle last edge separately
    total += d_dist[city_curr * n + d_pop[base]];
    return total;
}
```

The GPU's memory pipeline will overlap the `d_pop[base + k + 2]` load with the `row_curr[city_next]` compute, hiding some latency.

---

## Multi-Individual Parallelism for Large n

At large n, a single thread evaluating a tour of n=1000 cities takes 1000 sequential distance lookups — potentially 1000 × 200 cycles = 200,000 cycles per evaluation. With 32 threads per island all doing this simultaneously but independently, you have good parallelism.

An alternative: **assign multiple threads per individual**. Split the tour into segments and have `W` threads cooperatively evaluate one tour:

```
W=1:   1 thread evaluates all n edges (current approach)
W=2:   2 threads split n/2 edges each, reduce with 1 shuffle
W=4:   4 threads split n/4 edges each, reduce with 2 shuffles
W=32:  32 threads split n/32 edges each — full warp per individual
```

For W=32 (full warp per individual):
```
Block structure: BLOCK_POP_SIZE individuals × 32 threads = 32×32 = 1024 threads
Smem: same scratch arrays but indexed by individual, not thread
Tour eval: each warp of 32 threads evaluates one tour cooperatively
           → n/32 sequential lookups per thread (vs n for W=1)
           → 32× fewer sequential stalls, at cost of 1 warp-level reduction per individual
```

For n=1000, W=32 reduces sequential distance lookups per thread from 1000 to ~31, with one `__shfl_down_sync` tree reduction to sum the partial lengths. This is a significant win when n is large and memory latency dominates.

```cpp
// W=32: full warp evaluates one tour
__device__ int tour_length_warp_parallel(
    const int* tour,   // tour for this individual (in global or smem)
    int n,
    const int* __restrict__ d_dist,
    int lane)          // threadIdx.x % 32
{
    int partial = 0;
    for (int k = lane; k < n; k += 32) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        partial += __ldg(&d_dist[a * n + b]);
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    return partial;  // valid only in lane 0
}
```

---

## 2-opt at Large n

For large n, 2-opt has O(n²) candidate pairs. With n=1000 that is ~500,000 pairs per pass. This maps beautifully to GPU parallelism:

```
Strategy: assign one thread per candidate pair (i, j)
Block: 1024 threads per tour
Pairs per block per pass: 1024

Passes needed to cover all n*(n-3)/2 pairs: ceil(500,000 / 1024) ≈ 488 kernel launches
OR: each thread iterates over multiple pairs (strided loop within one kernel)
```

For large n, 2-opt becomes more valuable because the GA alone struggles to find fine-grained improvements — the search space is too large. 2-opt post-processing consistently closes 5–15% of the gap to optimal for n > 200.

**Or-opt** (moving a single city or a chain of 2–3 cities to a better position) is often faster than 2-opt for large n because it only needs O(n) candidates per move instead of O(n²).

---

## The Full Large-n Optimization Story

```
global_0   Baseline: global dist, global pop (SoA? AoS?), thread-0 sort, no lmem fix
           BLOCK_POP_SIZE = 32 or 64 (register pressure allows more)

global_1   + lmem fix (bitmask — scales to n=2048 with 64 uint32 registers)

global_2   + SoA population layout (coalesced warp reads during evaluation)

global_3   + top-k reduction (thread-0 sort → O(k log P) reduction)

global_4   + warp-parallel tour evaluation (W=32, 1 warp per individual)
             changes BLOCK_POP_SIZE × 32 threads per block

global_5   + __ldg() prefetching for distance reads

global_6   + smem distance tile cache
             (uses histogram from profiling to pick tile size)

global_7   + 2-opt post-processing (parallel, one block per island best)

global_8   + greedy NN init (quality variant at any step)

── scaling knobs (Experiment Set 3 equivalent for large n) ──────────────
global_sweep   vary islands × pop_per_island with total fixed
               measure convergence vs n for n = 128, 256, 512, 1000
```

---

## Summary: What Changes at Each Scale

```
n ≤ 97     shared memory wins for dist AND pop fits
           → shared story is optimal
           → constant story documents the scatter penalty

97 < n ≤ 700  global dist + smem pop partially viable
               L2 caches full dist matrix
               smem holds pop only (32 individuals)
               → global_0 through global_5 applicable

700 < n ≤ 1024  L2 starts thrashing on dist
                 dist tile cache (global_6) becomes necessary
                 warp-parallel eval (global_4) critical
                 2-opt (global_7) important — GA quality degrades at large n

n > 1024    dist matrix > 4MB — L2 cannot hold it
             tile cache mandatory
             consider: only cache distances for the k nearest neighbors
             (k-nearest neighbor dist matrix — sparse, O(kn) instead of O(n²))
             2-opt essential
             greedy init essential — random init is hopeless at n=1000
```

### The k-nearest neighbor distance matrix

For large n, most tour edges connect nearby cities. You can replace the full n×n matrix with a sparse `n × k` matrix of the k nearest neighbors for each city (k ≈ 10–20):

```
Storage: n × k × 4 bytes
n=1000, k=20: 80KB — fits in L2 with room to spare
n=10000, k=20: 800KB — still fits in L2

Tradeoff: crossover and 2-opt can only consider edges to k-nearest neighbors
          Quality loss is small for large n (optimal tours rarely use long edges)
          Runtime gain is enormous (dist lookup table shrinks by n/k × )
```

This is the single highest-leverage optimization for n > 500 and deserves its own variant (`global_knn`).