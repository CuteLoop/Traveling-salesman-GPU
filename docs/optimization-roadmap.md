# CUDA TSP–GA Optimization Playbook
## From Naive Kernel to GPU-Resident Island GA with 2-opt Elite Refinement

> **How to use this document.** Every bottleneck is tied to an exact line in your code, a mathematical claim about P100 hardware, a profiling command that can confirm or refute that claim, and a code fix. The final section is an experimental roadmap with benchmark tables you can fill in. Numbers use: P100 — 732 GB/s DRAM, 4 MB L2, 8 KB constant-cache/SM, 64 KB shared/SM, 32 shared-memory banks × 4 B, 56 SMs, 2560 cores, 64 warps/SM, PCIe 3.0 × 16 ≈ 12 GB/s practical.

---

## Part 0 — The Three-Implementation Arc (orientation)

| Version | Evolution | Fitness | Transfers/gen | Parallelism model |
|---|---|---|---|---|
| `GPU-Naive.cu` | none (evaluation only) | GPU | 2 | 1 thread → 1 tour |
| `CUDA-GA.cu` | **CPU** | GPU | **2 × every gen** | 1 thread → 1 tour |
| `CUDA-GA-GPU-Pop.cu` | **GPU, island model** | GPU | 1 (outputs only) | 1 thread → 1 individual |

The GPU-Pop kernel is the focal implementation. All bottleneck analysis below targets it unless noted.

---

## Part 1 — Bottleneck Priority Matrix

Ratings are independent: **perf-impact** = expected end-to-end speedup potential; **GPU-friendliness** = ease of implementation without a major kernel rewrite.

| # | Bottleneck | Code location | Perf impact | GPU-friendly | Stage |
|---|---|---|---|---|---|
| B1 | 32-way shared memory bank conflict | `pop_a[tid * n + k]` | 🔴 10/10 | 🟢 10/10 | 1 – now |
| B2 | `used[MAX_CITIES]` local memory spill | `order_crossover_device` | 🔴 9/10 | 🟢 8/10 | 1 – now |
| B3 | Thread-0 serialized O(P²) sort | `if (tid == 0)` sort loop | 🟠 8/10 | 🟢 9/10 | 1 – now |
| B4 | Constant memory scatter penalty | `tour_length_const`, `c_dist` | 🟠 7/10 | 🟡 6/10 | 1 – benchmark |
| B5 | Low occupancy (1.56%) | 32-thread block, 33 KB smem | 🟠 7/10 | 🟡 5/10 | 2 – after B1–B3 |
| B6 | Serial per-thread tour evaluation | `tour_length_const` inner loop | 🟠 8/10 | 🟡 4/10 | 2 – experimental |
| B7 | OX branch divergence | `if (used[gene]) continue` | 🟡 5/10 | 🟡 6/10 | 2 – profile first |
| B8 | No intra-tour local search | missing 2-opt pass | 🔴 10/10 (quality) | 🟡 5/10 | 3 – new feature |
| B9 | PCIe memcpy every generation | `evaluate_population_cuda` | 🔴 10/10 | 🟢 fixed in GPU-Pop | already fixed |

---

## Part 2 — Detailed Bottleneck Analysis

### B1 — Shared Memory Bank Conflicts

**Code:**
```cpp
// ga_island_kernel — current layout
int* pop_a = shared;
int* pop_b = pop_a + BLOCK_POP_SIZE * n;       // stride = n = 128
// ...
init_random_tour(pop_a + tid * n, n, rng);     // access: pop_a[tid*128 + k]
```

**Math:**  
P100 shared memory has 32 banks, each 4 bytes wide. Bank for element at address `e`:
```
bank(e) = e % 32
```
For `pop_a[t * n + k]` with `n = 128`:
```
bank(t, k) = (t × 128 + k) % 32
           = (0 + k) % 32              ← because 128 % 32 = 0
           = k % 32
```
The bank index is **independent of thread index t**. Every thread in the warp hits the same bank at the same instruction — a 32-way bank conflict. The hardware serializes these 32 accesses into 32 sequential transactions.




**Throughput consequence:**  
Ideal shared memory: 1 transaction/cycle. With 32-way conflict: 32 transactions/cycle-slot → **32× slower** for every population read/write.

**Shared memory transactions per generation (current):**
```
Per generation, each thread does:
  - 1× tour_length_const:     128 reads from pop (scatter, but pop is smem)
  - 1× order_crossover:       reads from parent_a[k] and parent_b[(r+off)%n]
  - Elite copy (2 threads):   128 reads + 128 writes

Lower bound smem transactions/gen:
  32 threads × 128 reads × 32-way conflict = 131,072 serialized transactions
  vs ideal (no conflict):                    4,096 transactions

Per-SM overhead/gen: 127,0 wasted transaction cycles
```

**The Fix — Pad stride to n+1:**
```cpp
// FIXED: stride = n + 1 = 129
const int stride = n + 1;
int* pop_a = shared;
int* pop_b = pop_a + BLOCK_POP_SIZE * stride;
int* lengths = pop_b + BLOCK_POP_SIZE * stride;
int* order   = lengths + BLOCK_POP_SIZE;

// All accesses: pop_a[tid * stride + k]
```
With stride = 129:
```
bank(t, k) = (t × 129 + k) % 32
           = (t × 1 + k) % 32      ← because 129 % 32 = 1
           = (t + k) % 32
```
Thread 0: bank k, Thread 1: bank (k+1), ..., Thread 31: bank (k+31) — all 32 distinct banks.  
**Zero bank conflicts.**

**Shared memory overhead of padding:**
```
Before: (2 × 32 × 128 + 2 × 32) × 4 = 33,024 bytes
After:  (2 × 32 × 129 + 2 × 32) × 4 = 33,280 bytes  (+256 bytes, negligible)
```

**Profiling verification:**
```bash
ncu --metrics \
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --kernel-name ga_island_kernel \
  ./CUDA-GA-GPU-Pop benchmark.tsp 128 100

# Legacy nvprof equivalent:
nvprof --metrics \
  shared_load_transactions_per_request,\
  shared_store_transactions_per_request \
  ./CUDA-GA-GPU-Pop benchmark.tsp 128 100
```
**Expected:** `shared_load_transactions_per_request` drops from ~32 to ~1.

---

### B2 — Local Memory Spill from `used[MAX_CITIES]`

**Code:**
```cpp
__device__ void order_crossover_device(const int* parent_a,
                                        const int* parent_b,
                                        int* child, int n,
                                        unsigned int& rng) {
    int used[MAX_CITIES];              // ← 128 ints, runtime-indexed → SPILLS
    for (int i = 0; i < n; ++i) { child[i] = -1; used[i] = 0; }

    for (int i = left; i <= right; ++i) {
        child[i] = parent_a[i];
        used[parent_a[i]] = 1;        // ← runtime index: compiler cannot keep in regs
    }
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (used[gene]) continue;     // ← runtime index
        child[out] = gene;
        used[gene] = 1;
    }
}
```

**Why it spills:** The CUDA compiler can allocate small arrays to registers only when all index expressions are compile-time constants or loop-invariant. `used[parent_a[i]]` and `used[gene]` are indexed by runtime-dependent values. The compiler must place `used[]` in **local memory** — which is physically DRAM, per-thread addressed.

**Compile-time confirmation:**
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O3 -lineinfo \
     -o gpu_pop CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep -A5 "ga_island_kernel"

# Look for:
# ptxas info: Function properties for ga_island_kernel
#   lmem = 512    ← 128 × 4 = 512 bytes of local memory per thread
#   smem = 33024
#   registers = 40
```
`lmem > 0` confirms the spill.

**Traffic calculation:**
```
Per crossover call:
  Writes to used[]:  n = 128 writes × 4 bytes = 512 bytes
  Reads from used[]: n = 128 reads  × 4 bytes = 512 bytes
  Total: 1024 bytes per child

Per full run (30 non-elite threads, 128 islands, 1000 gens):
  30 × 128 × 1000 × 1024 = 3,932,160,000 bytes ≈ 3.93 GB
```
This competes directly with L2 bandwidth and is completely invisible in standard bandwidth metrics.

**The Fix — 4-register bitmask:**
```cpp
__device__ void order_crossover_device(const int* parent_a,
                                        const int* parent_b,
                                        int* child, int n,
                                        unsigned int& rng) {
    // Replace: int used[MAX_CITIES]
    // With:    4 × uint32 held in registers (128 bits covers 128 cities)
    uint32_t used0 = 0, used1 = 0, used2 = 0, used3 = 0;

    // Inline helpers: set city c, test city c
    #define MARK(c)   do { \
        uint32_t bit = 1u << ((c) & 31); \
        if ((c) < 32)       used0 |= bit; \
        else if ((c) < 64)  used1 |= bit; \
        else if ((c) < 96)  used2 |= bit; \
        else                used3 |= bit; \
    } while(0)

    #define ISSET(c) ( \
        (c) < 32  ? (used0 >> ((c)     )) & 1 : \
        (c) < 64  ? (used1 >> ((c)-32  )) & 1 : \
        (c) < 96  ? (used2 >> ((c)-64  )) & 1 : \
                    (used3 >> ((c)-96  )) & 1   )

    // ... rest of OX logic using MARK() and ISSET()
    for (int i = 0; i < n; ++i) child[i] = -1;

    for (int i = left; i <= right; ++i) {
        child[i] = parent_a[i];
        MARK(parent_a[i]);
    }
    int out = (right + 1) % n;
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (ISSET(gene)) continue;
        child[out] = gene;
        MARK(gene);
        out = (out + 1) % n;
    }
    #undef MARK
    #undef ISSET
}
```
After this fix, `lmem` in ptxas output should drop to 0. The 4 uint32 variables are held in registers — zero DRAM traffic.

**Profiling verification:**
```bash
# Step 1: compile-time
nvcc -arch=sm_60 --ptxas-options=-v -O3 -o gpu_pop_v2 CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep lmem
# Goal: lmem = 0

# Step 2: runtime confirmation
nvprof --metrics local_load_transactions,local_store_transactions \
       ./gpu_pop_v2 benchmark.tsp 128 1000
# Goal: both = 0
```

---

### B3 — Thread-0 Serialized Sort

**Code:**
```cpp
if (tid == 0) {
    for (int i = 0; i < BLOCK_POP_SIZE; ++i) order[i] = i;
    // O(P²) selection sort — 496 comparisons for P=32
    for (int i = 0; i < BLOCK_POP_SIZE - 1; ++i) {
        int best = i;
        for (int j = i + 1; j < BLOCK_POP_SIZE; ++j) {
            if (lengths[order[j]] < lengths[order[best]]) best = j;
        }
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
    }
}
__syncthreads();
```

**Math:**  
Selection sort comparisons for P = 32: `(P-1) + (P-2) + ... + 1 = P(P-1)/2 = 496`.  
All 31 other threads idle during these 496 steps.

```
Serial ops across full run: 496 × 128 islands × 1000 gens = 63,488,000 comparisons
All single-threaded.
```

The sort is actually unnecessary for this use case — we only need the indices of the top `elite_count = 2` individuals. Finding the 2 minimum elements from 32 requires at most 31 + 30 = 61 comparisons serially, or 5 warp-shuffle steps in parallel.

**The Fix — Warp shuffle min-reduction for top-k elites:**
```cpp
// Find top-2 elite indices using warp shuffle (all 32 threads participate)
// Works perfectly when BLOCK_POP_SIZE == 32 (one warp)

__device__ void find_top2_warp(const int* lengths,
                                int* elite0_idx,
                                int* elite1_idx) {
    int my_len = lengths[threadIdx.x];
    int my_idx = threadIdx.x;

    // Pass 1: find global minimum (5 shuffle steps)
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_len = __shfl_xor_sync(0xffffffff, my_len, mask);
        int o_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
        if (o_len < my_len || (o_len == my_len && o_idx < my_idx)) {
            my_len = o_len; my_idx = o_idx;
        }
    }
    int best_idx = my_idx;  // lane 0 holds; broadcast
    best_idx = __shfl_sync(0xffffffff, best_idx, 0);

    // Pass 2: find second minimum (exclude best_idx)
    my_len = (threadIdx.x == best_idx) ? INT_MAX : lengths[threadIdx.x];
    my_idx = threadIdx.x;
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_len = __shfl_xor_sync(0xffffffff, my_len, mask);
        int o_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
        if (o_len < my_len || (o_len == my_len && o_idx < my_idx)) {
            my_len = o_len; my_idx = o_idx;
        }
    }
    int second_idx = __shfl_sync(0xffffffff, my_idx, 0);

    if (threadIdx.x == 0) {
        *elite0_idx = best_idx;
        *elite1_idx = second_idx;
    }
}
```
**Steps: 2 × 5 shuffle iterations = 10 warp-synchronous steps.**  
No `__syncthreads()` needed (shuffle is warp-synchronous). No shared memory used.

> **Important:** `__shfl_*` operates within one warp. This works correctly as long as `BLOCK_POP_SIZE ≤ 32`. If you ever scale `BLOCK_POP_SIZE` beyond 32, you need a two-level reduction (warp-local shuffles → shared memory → shuffle of warp results).

**Profiling verification:**
```bash
# Warp stall analysis — thread-0 bottleneck shows as "stall_sync" in other threads
ncu --metrics \
  smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
  smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct \
  --kernel-name ga_island_kernel \
  ./CUDA-GA-GPU-Pop benchmark.tsp 128 1000

# Before fix: high barrier stall % (31 threads at __syncthreads after tid==0 sort)
# After fix:  barrier stall % drops significantly
```

---

### B4 — Constant Memory Scatter Penalty

**Code:**
```cpp
__constant__ int c_dist[MAX_CITIES * MAX_CITIES];  // 64 KB

__device__ int tour_length_const(const int* tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        total += c_dist[a * n + b];   // ← scatter: a,b differ per thread
    }
    return total;
}
```

**Analysis:**  
Constant memory is backed by an 8 KB/SM cache with **hardware broadcast**: if all warp threads read the *same* address, one fetch serves all 32 lanes. If threads read *different* addresses (scatter), the hardware **serializes**: 32 requests → 32 sequential fetches.

With 32 threads each following a different tour, `a` and `b` differ per thread at every step. This is maximum scatter — constant memory serializes every single dist lookup.

**P100 L2 argument:**  
Matrix size: `128 × 128 × 4 = 65,536 bytes = 64 KB`.  
P100 L2 = 4 MB. Matrix occupies 1.6% of L2. After the first ~10 tours warm L2, all subsequent dist reads are L2 hits. L2 serves scatter reads in **parallel** (multiple requests in flight). Constant memory does not.

```
Constant memory scatter (worst case):
  32 threads × 1 serialized fetch each = 32 × ~10 cycles = ~320 cycles/warp/step

L2-cached global memory scatter:
  32 threads, requests issued in parallel, L2 hit latency ~30–80 cycles
  With memory-level parallelism: effective throughput significantly higher
```

**The Fix — Switch to `const int* __restrict__`:**
```cpp
// Host: pass dist as a regular global memory pointer
// In kernel signature:
__global__ void ga_island_kernel(int n, int generations,
                                  float mutation_rate, int elite_count,
                                  unsigned int seed,
                                  const int* __restrict__ dist,  // ← new
                                  int* best_tours, int* best_lengths)

// In tour_length:
__device__ int tour_length_global(const int* __restrict__ dist,
                                   const int* tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        total += dist[a * n + b];     // L2-cached, scatter-parallel
    }
    return total;
}
```

> **This must be benchmarked, not assumed.** For N = 128, L2 warming should win. The decision gate is kernel elapsed time, not theoretical argument.

**A/B benchmark command:**
```bash
# Build V_A: constant memory (current)
nvcc -arch=sm_60 -O3 -DUSE_CONSTANT_MEM -o gpu_pop_A CUDA-GA-GPU-Pop.cu tsplib_parser.cpp

# Build V_B: global memory + restrict
nvcc -arch=sm_60 -O3 -DUSE_GLOBAL_MEM -o gpu_pop_B CUDA-GA-GPU-Pop.cu tsplib_parser.cpp

# Same seed, same config
for i in $(seq 1 10); do
  ./gpu_pop_A benchmark.tsp 128 1000 0.05 2 42
  ./gpu_pop_B benchmark.tsp 128 1000 0.05 2 42
done

# Profiler metrics to compare:
ncu --metrics \
  l2_read_hit_rate,\
  sm__cycles_elapsed.avg,\
  gpu__time_duration.sum \
  ./gpu_pop_A benchmark.tsp 128 100
```

---

### B5 — Low Occupancy

**Math:**
```
Shared memory per block: 33,024 bytes (before padding) / 33,280 bytes (after)
P100 shared memory per SM: 64 KB = 65,536 bytes

Max blocks per SM from shared memory:
  floor(65,536 / 33,280) = 1 block per SM

Warps per block: BLOCK_POP_SIZE / 32 = 32/32 = 1 warp

Active warps per SM: 1 × 1 = 1 warp
P100 max warps per SM: 64

Occupancy: 1/64 = 1.56%
```

With 1 active warp per SM, there are zero backup warps to hide memory latency. Every L2 miss (≈80 cycles) stalls the entire SM. This is the root cause of much of the observed slowness.

**Check with CUDA API:**
```cpp
// Add this to run_gpu_population_ga before the kernel launch:
int max_active_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, ga_island_kernel,
    BLOCK_POP_SIZE, shared_bytes);
printf("Max active blocks/SM: %d  (warps/SM: %d, occupancy: %.1f%%)\n",
       max_active_blocks,
       max_active_blocks * (BLOCK_POP_SIZE / 32),
       100.0f * max_active_blocks * BLOCK_POP_SIZE / 32 / 64);
```

**Options to improve occupancy:**

| Option | Change | Predicted occupancy | Risk |
|---|---|---|---|
| A (current) | 32 threads, 33 KB smem | 1.56% | — |
| B | Reduce to BLOCK_POP_SIZE=16, 16.5 KB smem | 6.25% (4 blocks/SM) | Smaller island population |
| C | Pack 2 islands per block (64 threads) | 3.1% (but 2× throughput/SM) | Requires intra-block sync |
| D | Warp-per-individual redesign (1024 threads) | ~50% | Major kernel rewrite |

---

### B6 — Serial Tour Evaluation (Experimental: 1 Warp per Individual)

**Current:** Thread `t` loops through all N = 128 edges serially.

```cpp
lengths[tid] = tour_length_const(current + tid * n, n);
// 128 serial iterations per thread
```

**Proposed: 1 warp = 1 individual.**  
Each of 32 lanes computes a subset of edges, then warp-reduces:

```cpp
// 32 threads, each handling 4 edges of a 128-city tour
__device__ int tour_length_warp(const int* __restrict__ dist,
                                 const int* tour, int n) {
    int lane = threadIdx.x & 31;
    int partial = 0;
    for (int k = lane; k < n; k += 32) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        partial += dist[a * n + b];
    }
    // Warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, mask);
    return partial;  // valid in lane 0
}
```

**Steps: 4 serial dist lookups + 5 shuffle steps vs 128 serial lookups.**  
Theoretical compute speedup: ~128/4 = **32×** for the evaluation step (memory-bound; real speedup less).

**Block size consequence:**  
32 individuals × 32 threads/individual = **1024 threads/block**.  
Register count and shared memory per block must be re-evaluated for this config.

---

### B7 — OX Branch Divergence

**Code:**
```cpp
for (int offset = 1; offset <= n; ++offset) {
    int gene = parent_b[(right + offset) % n];
    if (used[gene]) continue;     // ← data-dependent branch: diverges across warp lanes
    child[out] = gene;
    used[gene] = 1;
    out = (out + 1) % n;
}
```

When 30 crossover threads execute this simultaneously, each thread's `used[gene]` check depends on its own tour data. Thread 0 may hit `continue`, threads 1–29 may not. The warp hardware must execute both paths in separate passes — the divergent threads are masked off but still consume cycles.

**Expected branch efficiency:** 50–70% (profiler-dependent).

**Profiling command:**
```bash
ncu --metrics \
  smsp__sass_branch_targets_threads_divergent.sum,\
  smsp__sass_branch_targets_threads_uniform.sum \
  --kernel-name ga_island_kernel \
  ./CUDA-GA-GPU-Pop benchmark.tsp 128 1000

# branch_efficiency = uniform / (uniform + divergent)
```

**Fix: predicated write (low-risk):**
```cpp
// No early exit — all threads execute the same number of iterations
for (int offset = 1; offset <= n; ++offset) {
    int gene = parent_b[(right + offset) % n];
    bool not_used = !ISSET(gene);         // predicate
    if (not_used) child[out] = gene;      // predicated store (usually single PTX @p.st)
    if (not_used) { MARK(gene); out = (out + 1) % n; }
}
```

**Fix: PMX crossover (high-reward, changes genetic semantics):**  
Partially Mapped Crossover has a fixed-length loop with no data-dependent `continue`. All lanes execute identical instruction counts → 100% branch efficiency in the crossover kernel.

---

## Part 3 — New Feature: Parallel 2-opt on Elite Individuals

### 3.1 Why 2-opt matters

The GA generates diverse populations but convergence is slow because crossover and mutation are random. **2-opt local search** is a deterministic improvement operator: for each pair of edges (i, i+1) and (j, j+1), test if swapping them (reversing the segment i+1..j) decreases tour length. Accept if `delta < 0`.

Applying 2-opt to the elite individuals after each generation (or every K generations) can dramatically accelerate convergence without changing the island model structure.

**Paper evidence (Ermiş & Çatay, 2017):** Their parallel GPU 2-opt on a 500-node instance ran **181 ms** vs 253 ms sequential, and matched or exceeded solution quality. Their occupancy table (reproduced in Part 5) directly informs the block-size experiments you should run.

### 3.2 Mathematical foundation

For a tour `T` of length N, a 2-opt move swaps edges `(T[i], T[i+1])` and `(T[j], T[j+1])` to produce edges `(T[i], T[j])` and `(T[i+1], T[j+1])`. The **delta** (change in tour length) is:

```
delta(i, j) = dist(T[i], T[j]) + dist(T[i+1], T[j+1])
            - dist(T[i], T[i+1]) - dist(T[j], T[j+1])
```

If `delta < 0`, the swap improves the tour.

Total number of 2-opt candidate pairs for N cities: `N(N-1)/2`.  
For N = 128: `128 × 127 / 2 = 8,128 pairs`.

### 3.3 GPU-parallel 2-opt kernel design

**Thread mapping:** 1 thread per candidate pair `(i, j)`.

```cpp
__global__ void twoopt_pass_kernel(
        const int* __restrict__ dist,
        int* tour,              // elite tour (in shared memory or global)
        int* improved_flag,     // global flag: did any swap improve?
        int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n * (n - 1) / 2;
    if (tid >= total_pairs) return;

    // Map linear tid to (i, j) pair with i < j
    // Closed-form inversion of triangular number mapping:
    int i = (int)((-1.0 + sqrt(1.0 + 8.0 * tid)) / 2.0);
    int j = tid - i * (i + 1) / 2 + i + 1;
    // Note: adjust for your 0-based, non-adjacent-only convention

    // Read 4 cities
    int a = tour[i];
    int b = tour[(i + 1) % n];
    int c = tour[j];
    int d = tour[(j + 1) % n];

    // Compute delta
    int delta = dist[a * n + c] + dist[b * n + d]
              - dist[a * n + b] - dist[c * n + d];

    if (delta < 0) {
        // Mark improvement found; actual reversal done in a second pass
        atomicExch(improved_flag, 1);
        // To avoid race conditions on tour[], write the (i,j) improvement
        // to a candidate buffer and apply in sorted order in a second pass
    }
}
```

> **Race condition note:** Multiple threads may find improving moves that conflict (overlapping segments). The standard fix is to detect improvements only in this pass, then apply the best non-conflicting move, or use a **two-pass strategy**: detect all deltas, then apply only the globally best move per iteration.

### 3.4 Two-pass 2-opt implementation (race-safe)

```
Pass 1: compute all delta(i,j) → store in delta_array[]
Pass 2: parallel reduction to find min delta and its (i,j)
Pass 3: if min_delta < 0, reverse segment tour[i+1..j]
Repeat until no improvement (convergence) or max_iterations reached
```

**Pass 1 kernel — delta computation:**
```cpp
__global__ void twoopt_delta_kernel(
        const int* __restrict__ dist,
        const int* __restrict__ tour,
        int* deltas,    // output: delta for each pair, size = n*(n-1)/2
        int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * (n - 1) / 2) return;

    // Decode (i, j) from linear index tid
    // Efficient method: row = floor((sqrt(8*tid+1)-1)/2)
    int i = (int)floorf((-1.0f + sqrtf(1.0f + 8.0f * tid)) * 0.5f);
    int j = tid - i * (i + 1) / 2 + i + 1;

    int a = tour[i],         b = tour[(i+1) % n];
    int c = tour[j],         d = tour[(j+1) % n];

    deltas[tid] = dist[a*n + c] + dist[b*n + d]
                - dist[a*n + b] - dist[c*n + d];
}
```

**Pass 2 — parallel min-reduction to find best move:**
```cpp
// Use thrust or CUB for this:
// thrust::min_element(thrust::device, deltas, deltas + n*(n-1)/2)
// Or custom kernel with shared memory reduction

__global__ void find_best_move(const int* deltas, int total_pairs,
                                int* best_delta, int* best_tid) {
    extern __shared__ int s_delta[];
    extern __shared__ int s_idx[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s_delta[threadIdx.x] = (tid < total_pairs) ? deltas[tid] : INT_MAX;
    s_idx[threadIdx.x]   = tid;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            s_delta[threadIdx.x + stride] < s_delta[threadIdx.x]) {
            s_delta[threadIdx.x] = s_delta[threadIdx.x + stride];
            s_idx[threadIdx.x]   = s_idx[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMin(best_delta, s_delta[0]);
        // write best_tid if this block has the global minimum
    }
}
```

**Pass 3 — reverse segment:**
```cpp
__global__ void reverse_segment(int* tour, int i, int j, int n) {
    // Reverse tour[i+1 .. j] in-place using (j-i)/2 parallel threads
    int len = j - i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len / 2) return;
    int left  = (i + 1 + tid) % n;
    int right = (j - tid + n) % n;
    int tmp = tour[left]; tour[left] = tour[right]; tour[right] = tmp;
}
```

### 3.5 Integration into the island kernel — elite 2-opt

The cleanest integration is to run 2-opt **within the island kernel** on the elite individual after each generation (or every K generations):

```cpp
// Inside ga_island_kernel generation loop, after sorting/elite selection:
if (generation % TWOOPT_INTERVAL == 0) {
    // Only thread 0..BLOCK_POP_SIZE-1 participate
    // Elite individual is at current[order[0] * stride]

    // Shared memory 2-opt: all 32 threads compute their own (i,j) deltas
    // tid maps to pair (i, j) in the elite tour's N*(N-1)/2 space
    // For N=128: 8128 pairs, 32 threads each do 8128/32 = 254 pairs

    __shared__ int s_best_delta;
    __shared__ int s_best_i, s_best_j;

    if (tid == 0) { s_best_delta = 0; }  // only improve if delta < 0
    __syncthreads();

    int* elite_tour = current + order[0] * stride;
    int my_best_delta = 0, my_best_i = -1, my_best_j = -1;

    // Each thread scans its subset of pairs
    for (int pair = tid; pair < n*(n-1)/2; pair += BLOCK_POP_SIZE) {
        int pi = (int)floorf((-1.0f + sqrtf(1.0f + 8.0f*pair)) * 0.5f);
        int pj = pair - pi*(pi+1)/2 + pi + 1;

        int a = elite_tour[pi],       b = elite_tour[(pi+1)%n];
        int c = elite_tour[pj],       d = elite_tour[(pj+1)%n];
        int delta = c_dist[a*n+c] + c_dist[b*n+d]
                  - c_dist[a*n+b] - c_dist[c*n+d];

        if (delta < my_best_delta) {
            my_best_delta = delta; my_best_i = pi; my_best_j = pj;
        }
    }

    // Warp reduce to find globally best move (32 threads = 1 warp)
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_delta = __shfl_xor_sync(0xffffffff, my_best_delta, mask);
        int o_i     = __shfl_xor_sync(0xffffffff, my_best_i,     mask);
        int o_j     = __shfl_xor_sync(0xffffffff, my_best_j,     mask);
        if (o_delta < my_best_delta) {
            my_best_delta = o_delta; my_best_i = o_i; my_best_j = o_j;
        }
    }
    // Lane 0 has the best move
    if (tid == 0 && my_best_delta < 0) {
        // Apply reversal: reverse elite_tour[my_best_i+1 .. my_best_j]
        int lo = my_best_i + 1, hi = my_best_j;
        while (lo < hi) {
            int tmp = elite_tour[lo]; elite_tour[lo] = elite_tour[hi];
            elite_tour[hi] = tmp;
            ++lo; --hi;
        }
    }
    __syncthreads();
}
```

**Cost of one 2-opt pass inside the kernel:**
```
Pairs to scan per island: N(N-1)/2 = 8,128
Distributed over 32 threads: 254 pairs/thread
Each pair: 4 dist lookups + 3 arithmetic ops = 7 ops
Total per pass: 32 threads × 254 × 7 = ~56,896 ops
```

This runs concurrently across all 128 islands (128 SMs), so the wall time is ≈ 1 island's worth of work.

**When to run 2-opt:**
- Every K generations (K = 10–50 is typical; tune experimentally)
- Only on elite individuals (indices `order[0]` and `order[1]`)
- Use `TWOOPT_INTERVAL = 0` to disable for baseline benchmarks

---

## Part 4 — Parallelization Pattern Experiments

Three architecturally distinct mappings are worth comparing. These are not incremental fixes — they are fundamentally different ways to assign work to threads.

### Pattern A — 1 Thread : 1 Individual (current)
```
Block = BLOCK_POP_SIZE = 32 threads
Thread t → individual t
Tour eval: 128 serial steps per thread
Shared memory: 33 KB/block → 1 block/SM → 1.56% occupancy
```

### Pattern B — 1 Warp : 1 Individual
```
Block = 32 warps × 32 threads = 1024 threads
Warp w → individual w (32 individuals per block)
Tour eval: 4 parallel steps + 5 shuffle reduce
Shared memory: same population size, but split across 32 warps
Occupancy: 1 warp active → potentially much higher
```

**Block layout for Pattern B:**
```cpp
// warp = threadIdx.x / 32  (which individual)
// lane = threadIdx.x % 32  (position within individual's computation)

int warp = threadIdx.x >> 5;
int lane = threadIdx.x & 31;

// Tour evaluation using warp-level parallelism
int partial = 0;
for (int k = lane; k < n; k += 32) {
    int a = pop_a[warp * stride + k];
    int b = pop_a[warp * stride + (k + 1) % n];
    partial += c_dist[a * n + b];
}
partial = __reduce_add_sync(0xffffffff, partial);
if (lane == 0) lengths[warp] = partial;
```

### Pattern C — 2 Islands per Block (occupancy packing)
```
Block = 64 threads (2 × 32)
First 32 threads → island A (first half of shared memory)
Next 32 threads → island B (second half of shared memory)
Shared memory: 2 × 16.5 KB = 33 KB (same total, but 2 islands)
Blocks needed: islands/2
Occupancy: 2 blocks/SM = 3.1%  (2× improvement over current)
```

### Pattern D — All-Pairs Evaluation with Scan (research-grade)
```
Each thread independently evaluates a different candidate modification
(2-opt, 3-opt) rather than managing an individual.
Requires a different GA loop structure.
```

### Pattern comparison table (to be filled in experimentally)

| Pattern | Block size | Islands/SM | Occupancy | Tour eval time | Generation time | Best tour (n=128) |
|---|---|---|---|---|---|---|
| A — current | 32 | 1 | ~1.56% | (measure) | (measure) | (measure) |
| A+B1+B2+B3 | 32 | 1 | ~1.56% | (measure) | (measure) | (measure) |
| B — warp/indiv | 1024 | TBD | (measure) | (measure) | (measure) | (measure) |
| C — 2 islands/block | 64 | 2 | ~3.1% | (measure) | (measure) | (measure) |
| A + 2-opt (K=10) | 32 | 1 | ~1.56% | (measure) | (measure) | (measure) |
| A + 2-opt (K=1) | 32 | 1 | ~1.56% | (measure) | (measure) | (measure) |

---

## Part 5 — Experimental Design and Benchmark Tables

### 5.1 Block and grid size sweep (after Ermiş & Çatay 2017 methodology)

The referenced paper found that occupancy determines performance, with 100% occupancy achievable for block dims 1024, 512, and 256 for a 500-node TSP. For your island kernel, shared memory dominates the residency constraint. Run this matrix:

**Experiment 1 — Block size vs. performance (fixed islands = 128, gens = 1000, n = 128, seed = 42)**

| Block size | Shared mem/block | Max blocks/SM (smem limit) | Theoretical occupancy | Kernel time (ms) | Best length | Std dev (10 runs) |
|---|---|---|---|---|---|---|
| 32 | 33,280 B | 1 | 1.56% | | | |
| 64 | ~66,048 B | 0 (doesn't fit!) | N/A | | | |
| 32 + padding | 33,280 B | 1 | 1.56% | | | |
| 16 (P=16) | ~16,640 B | 3 | 4.69% | | | |

Note: Increasing block size beyond 32 requires splitting islands across blocks or using Pattern B/C.

**Experiment 2 — Island count sweep (fixed gens = 1000, n = 128, seed = 42)**

| Islands | Blocks launched | SMs used (of 56) | Kernel time (ms) | Best length | Gens/sec |
|---|---|---|---|---|---|
| 28 | 28 | 28 | | | |
| 56 | 56 | 56 | | | |
| 112 | 112 | 56 (2/SM) | | | |
| 224 | 224 | 56 (4/SM) | | | |
| 512 | 512 | 56 (9/SM) | | | |

**Expected:** performance plateaus after 56 islands (one per SM). Beyond that you are time-sharing SMs and not gaining wall-clock performance, but *solution quality* may still improve due to more diverse search.

**Experiment 3 — Problem size sweep (fixed islands = 128, gens = 1000, seed = 42)**

| n (cities) | Shared mem/block | Matrix size | L2 fit? | Kernel time (ms) | Best length |
|---|---|---|---|---|---|
| 32 | ~8,448 B | 4 KB | Yes | | |
| 64 | ~16,768 B | 16 KB | Yes | | |
| 96 | ~25,088 B | 36 KB | Yes | | |
| 128 | ~33,280 B | 64 KB | Yes | | |

**Experiment 4 — 2-opt interval sweep (fixed islands = 128, gens = 1000, n = 128)**

| 2-opt interval K | Kernel time (ms) | Best length | Improvement over no-2opt |
|---|---|---|---|
| ∞ (disabled) | | | baseline |
| 100 | | | |
| 50 | | | |
| 20 | | | |
| 10 | | | |
| 5 | | | |
| 1 (every gen) | | | |

**Experiment 5 — Optimization version comparison (the full story)**

| Version | Key fix | Bank conflicts | Local mem | Sort | Kernel time (ms) | Best tour (n=128) |
|---|---|---|---|---|---|---|
| V0 — baseline | none | 32-way | 3.93 GB | O(P²) | | |
| V1 — +stride | B1 fixed | none | 3.93 GB | O(P²) | | |
| V2 — +bitmask | B2 fixed | none | 0 | O(P²) | | |
| V3 — +reduction | B3 fixed | none | 0 | O(log P) | | |
| V4 — +global dist | B4 fixed | none | 0 | O(log P) | | |
| V5 — +2-opt K=10 | B8 added | none | 0 | O(log P) | | |

### 5.2 How to run the benchmark matrix

**Timing wrapper (add to your main or a shell script):**
```cpp
// Inside run_gpu_population_ga, wrap the kernel launch:
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("kernel_time_ms=%.3f\n", ms);
```

**Shell script for Experiment 5:**
```bash
#!/bin/bash
# benchmark_versions.sh
TSP=benchmark.tsp
GENS=1000
ISLANDS=128
N_REPS=10

run_timed() {
    local bin=$1; local seed=$2
    for i in $(seq 1 $N_REPS); do
        ./$bin $TSP $ISLANDS $GENS 0.05 2 $seed
    done
}

echo "=== V0 baseline ==="
run_timed ./gpu_pop_v0 42

echo "=== V1 stride padded ==="
run_timed ./gpu_pop_v1 42

echo "=== V2 bitmask ==="
run_timed ./gpu_pop_v2 42

echo "=== V3 warp reduction ==="
run_timed ./gpu_pop_v3 42

echo "=== V4 global dist ==="
run_timed ./gpu_pop_v4 42

echo "=== V5 2-opt K=10 ==="
run_timed ./gpu_pop_v5 42
```

---

## Part 6 — Complete Profiling Protocol

### 6.1 Pre-profiling: compile-time static analysis

Always run this first. It costs nothing and catches local-memory spills immediately.

```bash
nvcc -arch=sm_60 --ptxas-options=-v -O3 -lineinfo \
     -o gpu_pop CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 \
     | grep -E "ga_island_kernel|lmem|smem|registers"

# Interpret output:
# "lmem = 512"   → used[] is spilling (512 bytes = 128 ints)
# "lmem = 0"     → no spills (after bitmask fix)
# "smem = 33024" → current layout
# "smem = 33280" → after stride padding
# "registers = N"→ use with occupancy calculator
```

### 6.2 Runtime profiling — per bottleneck

**B1 — Shared memory bank conflicts:**
```bash
# Modern ncu (P100 / sm_60)
ncu --kernel-name ga_island_kernel \
    --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./gpu_pop benchmark.tsp 128 100

# Legacy nvprof
nvprof --metrics \
  shared_load_transactions_per_request,\
  shared_store_transactions_per_request \
  ./gpu_pop benchmark.tsp 128 100

# Goal: drops from ~32 to ~1
```

**B2 — Local memory spill:**
```bash
nvprof --metrics \
  local_load_transactions,\
  local_store_transactions \
  ./gpu_pop benchmark.tsp 128 1000

# Goal: both = 0 after bitmask fix
```

**B3 — Thread-0 serialization:**
```bash
ncu --kernel-name ga_island_kernel \
    --metrics \
    smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
    smsp__sass_average_data_bytes_per_sector_mem_shared_op_ld.ratio \
    ./gpu_pop benchmark.tsp 128 1000

# High barrier stall % before fix → drops after warp-shuffle reduction
```

**B4 — Constant vs global memory:**
```bash
ncu --kernel-name ga_island_kernel \
    --metrics \
    lts__t_sectors_srcunit_tex_op_read.sum,\
    lts__t_requests_srcunit_tex_op_read.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./gpu_pop_constmem benchmark.tsp 128 100

ncu --kernel-name ga_island_kernel \
    --metrics \
    lts__t_sectors_srcunit_tex_op_read.sum,\
    lts__t_requests_srcunit_tex_op_read.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./gpu_pop_globalmem benchmark.tsp 128 100

# Compare L2 sector counts and elapsed time
```

**B5 — Occupancy:**
```bash
ncu --kernel-name ga_island_kernel \
    --section Occupancy \
    ./gpu_pop benchmark.tsp 128 1000

# Or with nvprof:
nvprof --metrics \
  achieved_occupancy,\
  warp_execution_efficiency,\
  eligible_warps_per_cycle \
  ./gpu_pop benchmark.tsp 128 1000

# Goal: document the ~1.56% baseline; revisit after pattern B/C experiments
```

**B7 — Branch divergence:**
```bash
nvprof --metrics \
  branch_efficiency,\
  warp_nonpred_exec_efficiency \
  ./gpu_pop benchmark.tsp 128 1000

ncu --kernel-name ga_island_kernel \
    --metrics \
    smsp__sass_branch_targets_threads_divergent.sum,\
    smsp__sass_branch_targets_threads_uniform.sum \
    ./gpu_pop benchmark.tsp 128 1000
```

**2-opt validation:**
```bash
# Verify 2-opt actually improves tour length:
for seed in 42 123 456 789 1000; do
  echo -n "No 2-opt: "
  ./gpu_pop_v3 benchmark.tsp 128 1000 0.05 2 $seed | grep "tour length"
  echo -n "With 2-opt K=10: "
  ./gpu_pop_v5 benchmark.tsp 128 1000 0.05 2 $seed | grep "tour length"
done
```

### 6.3 Full diagnostic run script (one-shot)

```bash
#!/bin/bash
# full_profile.sh — run on P100 via SLURM
#SBATCH --gres=gpu:1 --time=01:00:00

module load cuda/11.8

BIN=./CUDA-GA-GPU-Pop
TSP=./benchmark.tsp

echo "======= STATIC ANALYSIS ======="
nvcc -arch=sm_60 --ptxas-options=-v -O3 -lineinfo \
     -o $BIN CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep -E "lmem|smem|registers"

echo "======= B1: SHARED BANK CONFLICTS ======="
nvprof --metrics shared_load_transactions_per_request,\
shared_store_transactions_per_request \
$BIN $TSP 128 100 2>&1 | tail -10

echo "======= B2: LOCAL MEMORY ======="
nvprof --metrics local_load_transactions,local_store_transactions \
$BIN $TSP 128 100 2>&1 | tail -10

echo "======= B3: WARP STALLS ======="
nvprof --metrics warp_execution_efficiency,branch_efficiency \
$BIN $TSP 128 1000 2>&1 | tail -10

echo "======= B5: OCCUPANCY ======="
nvprof --metrics achieved_occupancy,eligible_warps_per_cycle \
$BIN $TSP 128 1000 2>&1 | tail -10

echo "======= KERNEL TIMING (10 runs) ======="
for i in $(seq 1 10); do
  nvprof --print-gpu-trace $BIN $TSP 128 1000 2>&1 | grep "ga_island_kernel" | awk '{print $2}'
done

echo "Done."
```

---

## Part 7 — Implementation Roadmap

### Stage 1: High-ROI Fixes (minimal restructuring, measurable today)

**Commit 1: Stride padding for B1**
- File: `CUDA-GA-GPU-Pop.cu`
- Change: add `const int stride = n + 1;`, update all `tid * n + k` → `tid * stride + k`
- Shared memory arg: `(2 * BLOCK_POP_SIZE * (n+1) + 2 * BLOCK_POP_SIZE) * sizeof(int)`
- Profile gate: `shared_load_transactions_per_request` → 1

**Commit 2: Bitmask for B2**
- File: `CUDA-GA-GPU-Pop.cu`, function `order_crossover_device`
- Change: replace `int used[MAX_CITIES]` with `uint32_t used0,used1,used2,used3`
- Profile gate: `lmem = 0` in ptxas, `local_load_transactions = 0` at runtime

**Commit 3: Warp shuffle reduction for B3**
- File: `CUDA-GA-GPU-Pop.cu`, inside `ga_island_kernel`
- Change: replace `if (tid == 0)` sort with `find_top2_warp()` function
- Profile gate: barrier stall % decreases; kernel time decreases

**Commit 4: Benchmark constant vs global dist for B4**
- Add compile flag `#ifdef USE_GLOBAL_DIST` and build both versions
- Run A/B timing comparison with fixed seed
- Decision: adopt whichever is faster

### Stage 2: Structural Improvements (require kernel restructuring)

**Commit 5: Parallel 2-opt on elite individuals**
- Add `TWOOPT_INTERVAL` compile constant (default 10)
- Add 2-opt pass inside generation loop (as shown in Part 3)
- New experiment table: interval sweep (Experiment 4)

**Commit 6: Warp shuffle for all-threads fitness evaluation**
- Restructure `tour_length_const` into `tour_length_warp` using `lane = tid & 31`
- Requires keeping BLOCK_POP_SIZE = 32 (1 warp per individual)

### Stage 3: Architectural Experiments (new kernel variants)

**Commit 7: Pattern B — 1 warp per individual**
- New kernel `ga_island_kernel_warp_per_individual`
- 1024 threads/block, 32 individuals (32 warps)
- Compare with Pattern A on Experiment table

**Commit 8: Pattern C — 2 islands per block**
- New kernel `ga_island_kernel_packed`
- 64 threads/block, interleaved island logic
- Expected: 2× block count reduction, same solution quality

### Stage 4: Scaling Experiments

**Commit 9: Island migration**
- Double-buffered ring: islands write best tour to a migration ring in global memory every K generations
- No atomics needed if ring is pre-indexed by island

**Commit 10: N > 128 support**
- Remove `MAX_CITIES = 128` hard cap
- Switch dist from constant memory to global `__restrict__`
- Dynamic shared memory sizing
- Shared-memory tiling for dist when N > 1024

---

## Part 8 — Results Interpretation Matrix

| Measured metric | Value range | Diagnosis | Action |
|---|---|---|---|
| `shared_load_transactions_per_request` | ~32 | B1 confirmed — 32-way bank conflict | Apply stride = n+1 |
| | ~1 | B1 resolved | Move on |
| `lmem` (ptxas) | > 0 | B2 confirmed — `used[]` spilling | Apply bitmask |
| | = 0 | B2 resolved | Move on |
| `warp_execution_efficiency` | ~100% AND `achieved_occupancy` < 5% | Latency hiding impossible — correct inefficiency | Investigate occupancy limits |
| `branch_efficiency` | < 70% | B7 significant — OX divergence hurts | Consider PMX or predicated write |
| | > 90% | B7 negligible | Deprioritize |
| `achieved_occupancy` | < 5% | B5 confirmed — smem-limited | Profile register count too; try Pattern B or C |
| | > 40% | Good — latency hiding active | Move to arithmetic bottlenecks |
| Tour length (2-opt vs no 2-opt) | Measurably lower | 2-opt integration working | Tune interval K |
| | No difference | 2-opt not improving elites | Check delta calculation; check TWOOPT_INTERVAL |
| Kernel time V0 → V3 | Significant drop | Bank conflict + local mem + sort fixes confirmed | Document and continue |
| | Minimal change | Bottleneck is elsewhere | Profile stall reasons more carefully |

---

*End of playbook. Fill in the gray cells in the experiment tables with measured values on your P100. Every cell that deviates from prediction is a finding worth reporting.*