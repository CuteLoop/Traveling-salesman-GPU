# CUDA TSP Optimization Story
## An Inquiry-Based Guide from Naive GPU to Island GA

> **How to use this guide.** Each section opens with a question about code you have already written. We calculate what the hardware is *actually doing*, name the bottleneck, and then derive the fix. Numbers are grounded in P100 specs (memory bandwidth 732 GB/s, L2 4 MB, constant-cache 8 KB/SM, 64 shared-memory banks of 4 bytes, 48–64 KB shared/SM, 2560 CUDA cores, PCIe 3.0 × 16 ≈ 12 GB/s practical).

---

## The Three-Implementation Arc

Before diving in, orient yourself to what you have built:

| Version | Where does evolution live? | Where does fitness live? | Host↔Device transfers/gen |
|---|---|---|---|
| `GPU-Naive.cu` | CPU (none — just evaluation) | GPU | 2 (tours H→D, lengths D→H) |
| `CUDA-GA.cu` | **CPU** | GPU | **2 per generation** |
| `CUDA-GA-GPU-Pop.cu` | **GPU** (island model) | GPU | 1 total (outputs only) |

Each version fixes a bottleneck introduced by the previous one. This guide follows that arc, but pauses at each transition to *measure* before we conclude anything.

---

## Chapter 1 — The Naive Kernel: "How many threads are actually working?"

```cpp
// GPU-Naive.cu — the kernel
__global__ void eval_tour_lengths_kernel(const int* tours,
                                         const int* dist,
                                         int* lengths,
                                         int num_tours, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tours) return;          // guard

    int base = tid * N;
    int sum  = 0;
    for (int k = 0; k < N; ++k) {
        int a = tours[base + k];
        int b = tours[base + ((k+1) % N)];
        sum += dist[a * N + b];            // random scatter into dist
    }
    lengths[tid] = sum;
}
```

**Question 1.1 — Occupancy.**  
The launch config is:
```cpp
const int num_tours = 4;
int block_size = 256;
int grid_size   = (num_tours + block_size - 1) / block_size; // = 1
```

One block of 256 threads is launched. 252 of those threads immediately hit `if (tid >= 4) return` and do nothing. 

> **Warp-level view:** A warp is 32 threads. Of the 8 warps in this block, the first warp has 4 active lanes (12.5% efficiency) and the remaining 7 warps execute zero useful work.

**Arithmetic intensity of one active thread:**  
Each thread performs exactly N multiply-adds to compute a tour length. For N = 128:
- Useful FLOPs: 128 additions + 128 multiplications (for index arithmetic) ≈ 256 integer ops
- Memory traffic: 128 reads from `tours[]` (sequential, stride-1 within the thread) + 128 reads from `dist[]` (random)

Peak P100 integer throughput is ~5.3 TOPS. But you have 4 threads doing work. The bottleneck is not compute — it is that you are spending PCIe budget and kernel launch overhead on 4 threads of useful work.

**Question 1.2 — Is this embarrassingly parallel at all?**  
Yes, *if you scale num_tours*. The pattern "one thread computes one tour length" is correct in concept. The naive version is a *plumbing test*, not a production kernel. The real question is: once you scale to 512 or 4096 tours, does the memory access pattern become efficient? We will answer that in Chapter 2.

---

## Chapter 2 — Memory Access Patterns: "Are the threads cooperating on memory?"

### 2.1 Tour reads — are they coalesced?

Memory coalescing rule: threads in a warp should access consecutive 32-bit words in a single cache line (128 bytes = 32 ints on P100).

Thread `tid` accesses `tours[tid * N + k]` for each step `k`. Consider what happens in a warp (threads 0–31) at step `k = 0`:

```
thread  0: tours[0 * 128 + 0]  = tours[0]
thread  1: tours[1 * 128 + 0]  = tours[128]
thread  2: tours[2 * 128 + 0]  = tours[256]
...
thread 31: tours[31 * 128 + 0] = tours[3968]
```

The stride between consecutive threads is **N = 128 elements = 512 bytes**. A cache line is 128 bytes. Each thread lands in a *completely different* cache line. This is the **worst-case access pattern**: 32 cache line loads for 32 integers, zero sharing.

> **Formal term:** This is an **AoS (Array of Structures) layout** being accessed in column order. The fix is an **SoA (Structure of Arrays)** transpose so that at step `k`, all threads in the warp access city index `k` from different tours simultaneously: `tours_T[k * P + tid]`. Then the warp reads 32 consecutive ints from one cache line.

**Bandwidth waste calculation:**

| Layout | Cache lines loaded per warp per step | Useful data per cache line | Efficiency |
|---|---|---|---|
| AoS (current) | 32 | 4 bytes / 128 bytes | **3.1%** |
| SoA (transposed) | 1 | 32 × 4 / 128 = 128 bytes | **100%** |

For a population of P = 512 tours, N = 128, and 1000 generations:
- AoS total cache-line fetches for tour reads: 512 threads × 128 steps × 1 cache line = 65,536 lines × 128 bytes = **8 MB per generation**
- SoA total: 128 steps × ⌈512/32⌉ warps × 1 cache line = 128 × 16 × 128 = **262 KB per generation**

That is a **~31× reduction in L2 pressure** from this layout change alone.

### 2.2 Dist reads — the scatter problem

```cpp
sum += dist[a * N + b];
```

`a` and `b` are city indices from the tour — data-dependent values that differ per thread. At any step k, 32 threads in a warp issue 32 reads to `dist[]` at 32 *unpredictable* addresses. Each read lands in a potentially different cache line.

The full distance matrix is N × N ints:
- N = 128: 128 × 128 × 4 = **65,536 bytes = 64 KB**

P100's L2 cache is 4 MB, so the entire matrix fits easily. After the first few tours warm the cache, subsequent tours mostly L2-hit. However, the first pass is expensive, and *scatter reads still serialize*: each warp issues 32 separate cache lookups per step.

> **Question to ask:** Could we move `dist[]` into *constant memory* and get better behavior? See Chapter 4.

---

## Chapter 3 — The Hybrid Bottleneck: "How much time are we spending on the bus?"

`CUDA-GA.cu` uses the GPU only for fitness evaluation. Every generation, it does:

```cpp
// Inside the generation loop
evaluate_population_cuda(population, d_dist, d_population,
                         d_lengths, lengths, population_size, n);
// ... then CPU does selection, crossover, mutation ...
population.swap(next_population);
```

Inside `evaluate_population_cuda`:
```cpp
CUDA_CHECK(cudaMemcpy(d_population, population.data(),
                      population_bytes, cudaMemcpyHostToDevice));
// ... kernel ...
CUDA_CHECK(cudaMemcpy(lengths.data(), d_lengths,
                      lengths_bytes, cudaMemcpyDeviceToHost));
```

### 3.1 PCIe transfer budget

With `population_size = 512`, `n = 128`:
```
population upload:   512 × 128 × 4 bytes = 262,144 bytes   per generation
lengths download:    512 × 4 bytes        =   2,048 bytes   per generation
total per gen:                             ≈ 264 KB
total over 1000 gens:                      ≈ 264 MB
```

PCIe 3.0 × 16 practical bandwidth ≈ 12 GB/s.

```
264 MB / 12 GB/s ≈ 22 ms of pure transfer time
```

That is 22 ms out of a 1-second run just paying the PCIe toll — not counting kernel launch overhead (~5 µs each × 2 × 1000 = 10 ms), `cudaDeviceSynchronize` blocking, and the fully serialized CPU evolution loop.

### 3.2 The serialized CPU evolution phase

After GPU fitness evaluation, the CPU does selection, OX crossover, and swap mutation for all `population_size - elite_count` individuals. For 512 individuals with `n = 128`:

- `order_crossover`: O(N) per child × 508 children = 65,024 operations  
- `mutate_swap`: O(1) per individual  
- `tournament_select`: 3 random picks × 2 × 508 = 3,048 operations  
- `std::sort` on 512 elements: ≈ 512 × log₂(512) = 4,608 comparisons

Total CPU evolution work: ~68K operations, single-threaded, with data-dependent memory access inside `order_crossover`'s `used[]` vector.

> **The core problem of CUDA-GA.cu:** The GPU is idle during CPU evolution, and the CPU is idle during GPU fitness evaluation. You have two workers who can never work at the same time — a classic **producer-consumer deadlock pattern**.

### 3.3 Quantifying GPU idle time

With 512 tours and N = 128, the kernel does 512 × 128 = 65,536 additions and 65,536 scattered dist lookups. At ~5 TOPS and under L2 cache hits, this kernel runs in **under 100 µs**. The CPU then spends milliseconds doing evolution. The GPU utilization is perhaps **1–5%** of wall time.

> **Fix:** Move the entire evolution loop onto the GPU. This is exactly what `CUDA-GA-GPU-Pop.cu` does with its island model.

---

## Chapter 4 — Constant Memory: "Is the broadcast worth it?"

`CUDA-GA-GPU-Pop.cu` promotes `dist` from global memory to the constant memory space:

```cpp
__constant__ int c_dist[MAX_CITIES * MAX_CITIES]; // = 128×128×4 = 64 KB

// ... in tour_length_const:
for (int k = 0; k < n; ++k) {
    int a = tour[k];
    int b = tour[(k+1) % n];
    sum += c_dist[a * n + b];               // constant memory access
}
```

### 4.1 How constant memory works

Constant memory is backed by a dedicated **8 KB per-SM cache** with a **hardware broadcast** mechanism: if all threads in a warp access *the same address*, the value is fetched once and broadcast to all 32 lanes in a single cycle. This is the optimal case.

If threads access *different addresses*, the hardware **serializes** the accesses — one at a time — making it worse than L2 for scatter patterns.

### 4.2 When does it help here?

In `tour_length_const`, each thread computes its own tour's length independently. At step k, thread `tid` accesses `c_dist[tour[tid*n + k] * n + tour[tid*n + k+1]]`. Since each thread's tour is different, `a` and `b` differ across threads — this is a **scatter access**. Constant memory serializes this: **32 serialized reads** instead of 32 parallel ones.

> **The paradox:** Constant memory actually *hurts* for the core dist lookup in `tour_length_const`. Its real benefit comes from parameters broadcast to all threads: `n`, `mutation_rate`, `elite_count`. For a proper dist lookup, L2-cached global memory with a coalesced SoA layout (Chapter 2) would outperform constant memory.

### 4.3 When constant memory is the right call

Consider `tournament_select_device`:
```cpp
__device__ int tournament_select_device(const int* lengths, unsigned int& rng) {
    int best = rand_bounded(rng, BLOCK_POP_SIZE);
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        int candidate = rand_bounded(rng, BLOCK_POP_SIZE);
        if (lengths[candidate] < lengths[best])
            best = candidate;
    }
    return best;
}
```

Here `lengths[]` is in **shared memory**, not constant memory. Within a block, multiple threads read from `lengths[]` at data-dependent indices. This is a valid use of shared memory as a per-block software-managed cache — the correct tool for this pattern.

> **Rule of thumb:** Constant memory is best for truly read-only data that is accessed with a **uniform index** across all threads in a warp (parameters, lookup tables with predictable access). It is the wrong tool for data-dependent scatter.

---

## Chapter 5 — Shared Memory and Bank Conflicts: "Is the population layout causing hidden serialization?"

### 5.1 The shared memory layout

In `ga_island_kernel`, the population lives in shared memory:

```cpp
int* pop_a    = shared;                        // BLOCK_POP_SIZE × n ints
int* pop_b    = pop_a + BLOCK_POP_SIZE * n;    // BLOCK_POP_SIZE × n ints
int* lengths  = pop_b + BLOCK_POP_SIZE * n;    // BLOCK_POP_SIZE ints
int* order    = lengths + BLOCK_POP_SIZE;      // BLOCK_POP_SIZE ints
```

Total shared memory per block:
```
(2 × 32 × 128 + 2 × 32) × 4 bytes
= (8192 + 64) × 4
= 33,024 bytes ≈ 32.25 KB
```

The P100 supports up to 48 KB of shared memory per block (configurable up to 64 KB). This fits — but *barely*, and leaves little room for padding.

### 5.2 The bank conflict problem

Shared memory is organized into **32 banks**, each 4 bytes wide. Two accesses to the **same bank in the same instruction** serialize — this is a **bank conflict**.

Bank assignment formula:
```
bank(element_index) = element_index % 32
```

Now examine what happens when BLOCK_POP_SIZE = 32 threads each access column `k` of the population matrix (i.e., city position `k` in their assigned individual):

```
thread t accesses: pop_a[t * n + k]
→ bank = (t * n + k) % 32
```

With **n = 128**:
```
thread  0: bank = (0   × 128 + k) % 32 = k % 32
thread  1: bank = (1   × 128 + k) % 32 = (128 + k) % 32 = k % 32  [since 128 % 32 = 0]
thread  2: bank = (256 + k) % 32 = k % 32
...
thread 31: bank = (3968 + k) % 32 = k % 32
```

**All 32 threads hit the exact same bank.** This is a **32-way bank conflict** — the hardware serializes all 32 accesses. Every single shared memory access to the population in this layout is serialized.

> **This applies to any n that is a multiple of 32.** With MAX_CITIES = 128, every legal n that is a power of 2 (32, 64, 128) produces the worst possible conflict.

### 5.3 The padding fix

The standard cure is **stride padding**: instead of stride = n, use stride = n + 1.

```cpp
// Padded access: pop_a[t * (n + 1) + k]
bank = (t * (n + 1) + k) % 32
```

With n = 128, stride = 129:
```
thread  0: (0   × 129 + k) % 32 = k % 32
thread  1: (1   × 129 + k) % 32 = (129 + k) % 32 = (1 + k) % 32
thread  2: (258 + k) % 32 = (2 + k) % 32
...
thread 31: (31 × 129 + k) % 32 = (31 + k) % 32
```

Every thread hits a **different bank** — zero bank conflicts. The fix is:

```cpp
// Replace stride of n with stride of (n + 1)
const int stride = n + 1;  // or n | 1 if n is always odd-feasible

int* pop_a   = shared;
int* pop_b   = pop_a + BLOCK_POP_SIZE * stride;
// ... access as pop_a[tid * stride + k]
```

**Shared memory cost of padding:**
```
Before: 2 × 32 × 128 × 4 = 32,768 bytes
After:  2 × 32 × 129 × 4 = 33,024 bytes  (256 bytes extra — negligible)
```

**Potential speedup:** In the best case, going from 32-way serialized to fully parallel shared memory accesses gives a **32× throughput improvement** for population reads and writes. In practice, other bottlenecks limit the realized speedup, but this is one of the highest-leverage single-line fixes available.

---

## Chapter 6 — The Thread 0 Bottleneck: "Who is doing all the work?"

### 6.1 The serialized sort

Inside every generation of `ga_island_kernel`:

```cpp
if (tid == 0) {
    // Initialize order[]
    for (int i = 0; i < BLOCK_POP_SIZE; ++i) order[i] = i;

    // O(P²) selection sort
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

With `BLOCK_POP_SIZE = 32`:
- Inner loop iterations: (31 + 30 + ... + 1) = **496 comparisons**
- All executed by **1 thread** while **31 threads idle**

For 1000 generations × 128 islands = 128,000 sort invocations, each doing 496 comparisons: **63.5 million comparison operations**, entirely serialized on a single thread per block.

### 6.2 The actual need: finding the elite indices

We don't need a full sort. Elitism only needs the top `elite_count` (default 2) individuals. Finding the minimum 2 elements from 32 requires:
- Full sort: 496 comparisons
- Two-pass minimum scan: 31 + 30 = **61 comparisons** (find min, mark, find second min)
- Parallel reduction to find min: log₂(32) = **5 steps** using all 32 threads

### 6.3 Parallel reduction pattern

The pattern to find the minimum of `lengths[]` using all threads:

```cpp
// Step 1: each thread "owns" one element
__shared__ int red_len[BLOCK_POP_SIZE];
__shared__ int red_idx[BLOCK_POP_SIZE];
red_len[tid] = lengths[tid];
red_idx[tid] = tid;
__syncthreads();

// Step 2: tree reduction over 5 steps (log2(32))
for (int stride = BLOCK_POP_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        if (red_len[tid + stride] < red_len[tid]) {
            red_len[tid] = red_len[tid + stride];
            red_idx[tid] = red_idx[tid + stride];
        }
    }
    __syncthreads();
}
// red_idx[0] now holds the best individual's index
```

This runs in **5 `__syncthreads()` cycles** using all 32 threads, compared to 496 serial comparisons on one thread — a **99× reduction in steps**, and the work is parallelized.

**For finding the top `elite_count = 2`:** run two reductions, or use warp-level primitives:

```cpp
// Using warp shuffle to find minimum in 5 instructions (no __syncthreads needed)
int my_len = lengths[tid];
int my_idx = tid;
for (int mask = 16; mask > 0; mask >>= 1) {
    int other_len = __shfl_xor_sync(0xffffffff, my_len, mask);
    int other_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
    if (other_len < my_len) { my_len = other_len; my_idx = other_idx; }
}
// Lane 0 holds the global minimum
```

> **Warp shuffle advantage:** No shared memory needed, no bank conflicts possible, and the reduction completes in 5 warp-synchronous steps without a single `__syncthreads()`.

---

## Chapter 7 — Hidden Local Memory: "Where does `used[]` actually live?"

### 7.1 The `order_crossover_device` stack allocation

```cpp
__device__ void order_crossover_device(const int* parent_a,
                                        const int* parent_b,
                                        int* child,
                                        int n, unsigned int& rng) {
    bool used[MAX_CITIES];   // <-- 128 bools on device stack
    // ...
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (used[gene]) continue;
        child[out] = gene;
        used[gene] = 1;
        out = (out + 1) % n;
    }
}
```

`bool used[MAX_CITIES]` is indexed by a **runtime value** (`parent_a[i]` and `gene`). The compiler cannot allocate this in registers because the index is not statically known. Instead, the compiler spills it to **local memory** — which is physically global DRAM, just privately addressed per thread.

**Local memory access characteristics:**
- Latency: same as global memory (~400–800 cycles)
- Bandwidth: counted against global memory bandwidth
- Cached in L1/L2, but only if the access pattern allows it

### 7.2 Counting the hidden global memory traffic

For each child produced by OX crossover:
- Write phase: `n = 128` stores to `used[]` → 128 local memory writes
- Read phase: up to `n = 128` reads from `used[]` during the gap-fill loop → 128 local memory reads
- Total local memory traffic per child: ~256 × 4 bytes = **1,024 bytes**

With 30 non-elite threads per block per generation, and 128 islands × 1000 generations:
```
30 × 128 × 1000 × 1,024 bytes = 3.93 GB of hidden local memory traffic
```

This is completely invisible if you only measure explicit `cudaMemcpy` transfers — but it competes directly with your global memory bandwidth.

### 7.3 The privatization fix

Move `used[]` into shared memory and share it across the block, or restructure OX crossover to use a bitmask:

```cpp
// Option A: shared bool array (one per thread, padded)
__shared__ bool s_used[BLOCK_POP_SIZE][MAX_CITIES + 1]; // +1 avoids bank conflicts
// thread tid uses s_used[tid]

// Option B: bitmask (128 cities → 4 × 32-bit words = 4 uints)
unsigned int used_mask[4] = {0, 0, 0, 0};
// set city c: used_mask[c >> 5] |= (1u << (c & 31))
// test city c: (used_mask[c >> 5] >> (c & 31)) & 1
```

Option B (bitmask) keeps `used_mask` in 4 registers — zero local memory traffic. The bitmask test and set are 2–3 instructions each, cheaper than a memory load.

---

## Chapter 8 — Tour Length Parallelization: "What if one warp owned one individual?"

### 8.1 Current mapping

```
BLOCK_POP_SIZE = 32 threads, one island
thread t → individual t → computes entire tour length serially
```

Each thread runs a loop of `n = 128` iterations. Within the loop, the work is purely sequential: each step depends on the *previous* city in the tour. This is an **N-length serial chain** per thread.

### 8.2 Alternative mapping: 1 warp per individual

What if we assign one 32-thread warp to compute the length of a single tour? Each thread computes a *subset* of the N edges, then we reduce:

```
Thread t computes edges: t, t+32, t+64, t+96  (for N=128, 4 edges per thread)
Then: warp-level reduction using __reduce_add_sync
```

```cpp
// One warp, one tour
int partial = 0;
for (int k = tid; k < n; k += 32) {
    int a = tour[k];
    int b = tour[(k + 1) % n];
    partial += c_dist[a * n + b];
}
// Warp reduction
partial = __reduce_add_sync(0xffffffff, partial);
// partial in lane 0 = total tour length
```

With N = 128 and 32 threads per warp: each thread computes **4 edges** instead of 128. The warp reduction takes 5 shuffle instructions. Total time: 4 sequential dist lookups + 5 shuffles, versus 128 sequential dist lookups.

**Speedup estimate (ignoring memory latency):**
```
Serial:  128 dist lookups  (one thread)
Warp:    4 dist lookups + 5 shuffles  (32 threads, parallel dist lookups)
Speedup: ~128 / (4 + ε) ≈ 32× (compute-bound ideal)
```

### 8.3 What this does to the island model

If 1 warp handles 1 individual, and you have BLOCK_POP_SIZE = 32 individuals, you need:
```
32 individuals × 32 threads/individual = 1024 threads per block
```

The block size grows from 32 to 1024 threads. Shared memory layout must change accordingly. This is a larger refactor, but it exposes **massive parallelism** inside fitness evaluation and eliminates the serial inner loop entirely.

---

## Chapter 9 — Scan and Reduction: "What other serial patterns are hiding in our code?"

### 9.1 The Fisher-Yates shuffle in `init_random_tour`

```cpp
__device__ void init_random_tour(int* tour, int n, unsigned int& rng) {
    for (int i = 0; i < n; ++i) tour[i] = i;
    for (int i = n - 1; i > 0; --i) {
        int j = rand_bounded(rng, i + 1);
        int tmp = tour[i]; tour[i] = tour[j]; tour[j] = tmp;
    }
}
```

This runs in O(N) on one thread. With 32 threads per island each calling this at init, and 128 islands, initialization is 32 × 128 × 128 ≈ 524,288 sequential swap operations, all independent across threads. This is parallelizable but is a one-time cost — low priority.

### 9.2 The elite copy loop

```cpp
if (tid < elite_count) {
    const int elite_idx = order[tid];
    for (int k = 0; k < n; ++k) {
        next[tid * n + k] = current[elite_idx * n + k];
    }
}
```

With `elite_count = 2`, only 2 threads do this copy. Each copies N = 128 ints from one row of shared memory to another row. With the bank conflict from Chapter 5, each read is serialized. With padding, this becomes efficient.

More importantly: this is a **loop over N** run by 2 threads. With the warp-per-individual mapping from Chapter 8, the elite copy could use **all 32 threads** to copy one individual (each thread copies N/32 elements), cutting the copy time by 16×.

### 9.3 Finding the global best after all islands complete

```cpp
// CPU: iterate over cfg.islands results
int best_island = 0;
for (int island = 1; island < cfg.islands; ++island) {
    if (h_best_lengths[island] < h_best_lengths[best_island])
        best_island = island;
}
```

This runs on CPU. For 128 islands it is negligible. But if you scale to thousands of islands for a large instance, consider a GPU reduction kernel using the standard parallel min-reduction pattern:

```
Step 1: Each block computes its min using shared memory reduction
Step 2: One final block reduces block-level results
```

Two-level reduction for P islands takes O(log P) parallel steps instead of O(P) serial steps.

---

## Chapter 10 — Thread Mapping Deep Dive: "Are our threads doing the right jobs?"

### 10.1 The current mapping hierarchy

```
Grid:  1 block per island
Block: BLOCK_POP_SIZE = 32 threads
thread t → individual t in the population
```

This is a clean 1:1 mapping. The problem is that with 32 threads per block, the GPU can schedule at most:
```
P100 SMs: 56
Max blocks per SM (for 32-thread blocks, 32KB shared): 48
Active blocks: 56 × 48 = 2,688  (but we only launch cfg.islands = 128)
Active threads: 128 × 32 = 4,096
```

P100 has 2,560 CUDA cores. We are feeding it **4,096 threads** — seemingly enough — but with 32-thread blocks, each SM only has one warp to schedule. Warp latency hiding requires **multiple active warps per SM**. With one 32-thread block active per SM, there are zero backup warps to hide memory latency with.

**Occupancy calculation:**

For `ga_island_kernel` with the current layout:
- Registers per thread: estimate 32–48 (xorshift state, local vars, loop counters)
- Shared memory: 33 KB per block
- P100 shared memory per SM: 64 KB → **1 block per SM** (limited by shared memory)
- Warps per SM: 1 warp → **occupancy = 1/64 = 1.56%**

> This is extremely low occupancy. The GPU is not hiding latency — every memory operation stalls the entire SM.

### 10.2 Strategies to increase occupancy

**Option A — Reduce shared memory per block by reducing pop size:**  
BLOCK_POP_SIZE = 16 → shared memory = (2×16×128 + 32)×4 = 16,512 bytes ≈ 16 KB.  
Three blocks fit per SM → 3× occupancy (4.7%), but population quality per island drops.

**Option B — Warp-per-individual mapping (Chapter 8):**  
Block size = 1024 threads (32 warps). 32 warps per SM → **50% occupancy** — excellent for the P100. Requires restructuring the kernel around warp-level primitives.

**Option C — Multiple populations per block:**  
Pack 2–4 small islands into one block, using separate shared memory regions. More complex synchronization but better occupancy without changing the island semantics.

### 10.3 Tiling the distance matrix into shared memory

Instead of reading `c_dist[]` from constant memory (scatter problem, Chapter 4), tile the most frequently accessed rows of the distance matrix into shared memory before the generation loop:

```cpp
// Before the generation loop: cache rows of dist that are "hot" for this island's tours
__shared__ int s_dist_tile[TILE_ROWS][MAX_CITIES];  // e.g., TILE_ROWS = 8

// Cooperative load: each thread loads one int
if (tid < TILE_ROWS * n) {
    int row = tid / n;
    int col = tid % n;
    int global_row = island * TILE_ROWS + row;  // island-specific tile
    if (global_row < n)
        s_dist_tile[row][col] = c_dist[global_row * n + col];
}
__syncthreads();
```

For `n = 128` and `TILE_ROWS = 8`: `8 × 128 × 4 = 4,096 bytes` of shared memory for the tile. The rows to cache should be the cities most frequently visited — which is data-dependent and requires profiling, but even a static partial tile reduces constant memory scatter.

---

## Chapter 11 — Atomic Operations: "Are we paying atomic tax anywhere?"

The current implementations do not use `atomicAdd` or similar operations. This is correct — each thread/island writes to a separate output slot:

```cpp
best_lengths[island] = lengths[best_idx];          // island = blockIdx.x
for (int k = 0; k < n; ++k)
    best_tours[island * n + k] = current[best_idx * n + k];
```

Since each block writes to a distinct `island`-indexed location, there are no races and no atomics needed.

**Where atomics would become necessary:**  
If you implement **island migration** — periodically copying the best tour from one island to a neighbor — concurrent writes to a shared migration buffer would require either atomics or a lock. The recommended pattern for migration is to use a **double-buffered migration ring**: islands write to buffer A in even migrations and read from buffer B, then swap. No atomics required.

**If you add histogram-based diversity tracking:**  
Counting how many individuals share the same length bucket with `atomicAdd` into a histogram array: on P100 with 32-entry histograms, atomic contention is modest. For large histograms (≥ 1024 bins), use the **privatization + reduction** pattern:
1. Each thread maintains a per-thread private histogram in registers or local memory
2. After all threads finish, reduce private histograms into shared memory
3. One final `atomicAdd` per bin from shared to global

This replaces N atomic operations (one per element) with P atomic operations (one per unique bin per block).

---

## Chapter 12 — The Full Optimization Story: Bottleneck → Fix → Expected Gain

| # | Bottleneck (where in code) | Pattern | Expected Gain |
|---|---|---|---|
| 1 | `tours[tid * N + k]` — strided global reads (Naive, CUDA-GA) | Transpose to SoA layout | ~31× L2 pressure reduction |
| 2 | `dist[]` in global memory every generation (CUDA-GA) | Already fixed in GPU-Pop via constant memory; consider L2-warmed global for N>128 | Eliminates per-gen PCIe copy of dist |
| 3 | `cudaMemcpy` population every generation (CUDA-GA) | Fully GPU-resident evolution (GPU-Pop) | Eliminates ~22 ms/sec of PCIe overhead |
| 4 | CPU evolution serializes GPU (CUDA-GA) | GPU island model (GPU-Pop) | GPU utilization: 2% → near 100% |
| 5 | `pop_a[tid * n + k]` with n=128 — 32-way bank conflicts (GPU-Pop) | Pad stride to `n+1` | Up to 32× shared memory throughput |
| 6 | O(P²) selection sort on thread 0 (GPU-Pop) | Parallel reduction or warp shuffle min | ~99× fewer steps, 32 threads used |
| 7 | `bool used[MAX_CITIES]` in local memory (GPU-Pop) | Bitmask in 4 registers | Eliminates ~3.93 GB hidden DRAM traffic |
| 8 | Serial N-step tour length per thread (all versions) | Warp-per-individual + `__reduce_add_sync` | ~32× speedup for fitness evaluation |
| 9 | 1 warp per SM → zero latency hiding (GPU-Pop) | Warp-per-individual block (1024 threads) | Occupancy: 1.56% → ~50% |
| 10 | Scatter reads into constant memory (GPU-Pop) | Shared memory tile for hot dist rows | Reduces constant cache thrashing |

---

## Appendix A — P100 Quick Reference Numbers

| Resource | Value |
|---|---|
| CUDA cores | 2,560 |
| SMs | 56 |
| Global memory bandwidth | 732 GB/s |
| L2 cache | 4 MB |
| Constant cache per SM | 8 KB |
| Max shared memory per SM | 64 KB |
| Max shared memory per block | 48 KB (default) |
| Shared memory banks | 32 × 4 bytes |
| Max warps per SM | 64 |
| Max threads per block | 1,024 |
| Warp size | 32 |
| PCIe 3.0 × 16 (practical) | ~12 GB/s |
| Global memory load latency | ~400–800 cycles |
| Shared memory latency | ~4–32 cycles (bank conflicts add multiples) |

---

## Appendix B — Worked Example: Shared Memory for N=100 (Non-Power-of-2)

With `n = 100`:
```
bank(t, k) = (t × 100 + k) % 32
           = (t × 4 + k) % 32     [since 100 = 3×32 + 4]
```

For k = 0: banks are 0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, ...  
Every 8 threads map to the same bank → **8-way bank conflict** (not as bad as 32-way, but still costly).

With stride = 101:
```
bank(t, k) = (t × 101 + k) % 32
           = (t × 5 + k) % 32     [since 101 = 3×32 + 5]
```
GCD(5, 32) = 1 → the sequence (0, 5, 10, 15, ...) mod 32 cycles through all 32 banks before repeating. For BLOCK_POP_SIZE = 32 threads: each thread hits a **unique bank** → no conflicts.

> **General rule:** A stride of `n+1` eliminates bank conflicts when `gcd(n+1, 32) = 1`, which is true whenever `n+1` is odd (i.e., n is even). Since MAX_CITIES = 128 (even), stride = 129 (odd) always works.

---

## Appendix C — Roofline Position of Each Kernel

Arithmetic intensity (AI) = FLOPs / bytes accessed.

**`eval_tour_lengths_kernel` (Naive/CUDA-GA), AoS layout:**  
- FLOPs: 2N integer ops per tour (adds + index multiply)  
- Bytes: N×4 (tour reads, strided) + N×128 (dist scatter, one cache line per read) ≈ N × 132 bytes  
- AI at N=128: 256 FLOPs / (128 × 132 bytes) ≈ **0.015 FLOP/byte** — deep in memory-bound territory

**`tour_length_const` (GPU-Pop), constant memory:**  
- Same FLOPs  
- Constant cache hit rate varies; scatter pattern means frequent misses  
- Effective AI: similar to above — **memory-bound**

**Target after all optimizations:**  
- SoA layout: dist read is ≈ N/32 cache lines per warp step instead of N → 32× fewer bytes  
- AI: 256 / (128 × 4 + 128/32 × 128) ≈ 256 / (512 + 512) ≈ **0.25 FLOP/byte**  
- Still memory-bound, but at 16× higher intensity — much more cache-friendly

The true compute roof for int ops on P100 is ~5.3 TOPS. To reach it, we would need AI ≈ 5300/732 ≈ 7.2 FLOP/byte. Tour evaluation is inherently pointer-chasing and will remain memory-bound. The goal is to push AI as high as possible while staying bandwidth-bound, not to chase the compute roof.

---

*End of guide. Continue the story by profiling with `nvprof --metrics l2_read_hit_rate,shared_efficiency,warp_execution_efficiency` to verify that each optimization moves the metrics in the predicted direction.*

---

## Chapter 13 — Warp Divergence in Genetic Operators: "The Compute Bottleneck We Missed"

The memory analysis in Chapters 2–7 is necessary but not sufficient. Even if every memory access were perfectly coalesced and every shared memory bank conflict eliminated, the crossover kernel still contains a hidden compute bottleneck: **branch divergence** inside a data-dependent loop.

### 13.1 Where divergence happens in `order_crossover_device`

```cpp
__device__ void order_crossover_device(const int* parent_a,
                                        const int* parent_b,
                                        int* child, int n,
                                        unsigned int& rng) {
    bool used[MAX_CITIES];
    // ... segment copy from parent_a ...

    int out = (right + 1) % n;
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (used[gene]) continue;              // <-- DIVERGENCE POINT
        child[out] = gene;
        used[gene] = 1;
        out = (out + 1) % n;
    }
}
```

Consider what happens when 30 non-elite threads in a warp each call this function simultaneously. At some loop iteration `offset`, thread 0's `gene` may be in `used[]`, causing it to `continue` while threads 1–29 do not. The SIMT hardware must execute **both paths**: threads that `continue` are masked out (do nothing) while the non-continuing threads execute `child[out] = gene`. Then the roles may reverse on the next iteration.

### 13.2 Quantifying the divergence cost

Let `s` = the size of the segment copied from parent_a (chosen uniformly in [1, n-2]). Expected `s = n/3 ≈ 43` for n = 128. The gap-fill loop runs for n = 128 iterations. On average, `s` of those iterations will hit `continue` (gene already used), distributed unevenly across threads.

At any given iteration:
- Each thread independently tests `used[gene]` for its own data-dependent `gene`.
- The probability a given thread hits `continue` ≈ (items placed so far) / n.

**Expected branch divergence fraction:**  
Early in the gap-fill loop, few genes are used — low divergence probability. Late in the loop, most genes are used — high divergence. Averaging over the entire loop, roughly half the `continue` branches will diverge across the warp (different threads take different paths).

```
Branch efficiency ≈ 1 - P(divergent branch)
                 ≈ 1 - 0.5 × (fraction of conditional branches)
```

For a warp of 30 active crossover threads, if on average half the `continue` tests produce a split: **branch efficiency ≈ 50–70%**, meaning the warp does 1.4–2× more instruction cycles than the computationally ideal case.

**Formula: divergent cycles penalty**

NVIDIA reports *branch efficiency* as:
```
branch_efficiency = (non-divergent branches) / (total branches) × 100%
```

A value of 60% means 40% of branch instructions cause the warp to serialize, effectively halving the throughput of that code region.

### 13.3 The structural cause: irregular termination per thread

The `continue` statement makes the loop body irregular: thread `t` does zero work on some iterations and full work on others, with no uniform pattern across the warp. This is the canonical cause of SIMT inefficiency — **data-dependent per-thread control flow**.

**Contrast with the tour-length loop:**
```cpp
for (int k = 0; k < n; ++k) {
    sum += c_dist[a * n + b];
}
```
No branch inside the body. All threads execute exactly the same number of instructions. Branch efficiency = 100%.

### 13.4 Mitigation strategies

**Option A — Predicated execution (compiler hint):**  
Replace `if (used[gene]) continue;` with a predicated write:
```cpp
if (!used[gene]) {
    child[out] = gene;
    used[gene] = 1;
    out += 1;
}
// Always increment offset — no early exit
```
This does not eliminate the divergence at the branch instruction, but it removes the `continue` control transfer and turns the body into predicated stores. The PTX compiler may further optimize this into a single `@p.st` (predicated store) that executes in one cycle for all threads regardless of the predicate value.

**Option B — Divergence-free crossover (PMX instead of OX):**  
Partially Mapped Crossover (PMX) has a fixed-length, branchless inner loop. Every iteration does the same amount of work regardless of the tour data. For a production implementation targeting maximum GPU throughput, PMX or ERX are better choices than OX for exactly this reason.

**Option C — Sort-based elimination:**  
Rewrite the gene-exclusion logic using a sort + scan instead of a per-element membership test. This is a larger change but produces completely uniform execution across the warp.

---

## Chapter 14 — Warp Primitive Scope and Scaling Limits: "When Shuffles Are Not Enough"

### 14.1 The `__shfl_xor_sync` reduction (Chapter 6 revisited)

Chapter 6 proposed finding the elite minimum using warp shuffles:

```cpp
int my_len = lengths[tid];
int my_idx = tid;
for (int mask = 16; mask > 0; mask >>= 1) {
    int other_len = __shfl_xor_sync(0xffffffff, my_len, mask);
    int other_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
    if (other_len < my_len) { my_len = other_len; my_idx = other_idx; }
}
// Lane 0 holds global minimum — BUT ONLY IF BLOCK_POP_SIZE <= 32
```

This is correct and optimal — for exactly one warp. The `0xffffffff` mask means all 32 lanes participate, and `__shfl_xor_sync` exchanges values between lane pairs within the warp in hardware, with no shared memory required.

### 14.2 The breakdown at BLOCK_POP_SIZE > 32

The warp-per-individual redesign from Chapter 8 proposed scaling the block to 1024 threads (32 individuals × 32 threads each). Now the `lengths[]` reduction must span 32 warps.

`__shfl_xor_sync` **cannot communicate across warp boundaries**. Lane 0 of warp 0 and lane 0 of warp 1 cannot exchange values with a single shuffle instruction. The hardware guarantee only applies within one 32-thread warp.

### 14.3 The two-level reduction pattern

The correct architecture for a block-wide reduction is:

```
Level 1: warp-local reduction (5 shuffle steps per warp → 32 warp-level results)
Level 2: shared memory reduction across the 32 warp results (5 more steps)
Total: 10 steps, still O(log P)
```

```cpp
__shared__ int warp_min_len[32];    // one slot per warp
__shared__ int warp_min_idx[32];

// Level 1: each warp reduces its own lanes with shuffles
int lane = tid & 31;
int warp = tid >> 5;

int my_len = ...; int my_idx = tid;
for (int mask = 16; mask > 0; mask >>= 1) {
    int o_len = __shfl_xor_sync(0xffffffff, my_len, mask);
    int o_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
    if (o_len < my_len) { my_len = o_len; my_idx = o_idx; }
}
// Lane 0 of each warp writes warp result
if (lane == 0) { warp_min_len[warp] = my_len; warp_min_idx[warp] = my_idx; }
__syncthreads();

// Level 2: one warp reduces the 32 warp results
if (warp == 0) {
    my_len = warp_min_len[lane];
    my_idx = warp_min_idx[lane];
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_len = __shfl_xor_sync(0xffffffff, my_len, mask);
        int o_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
        if (o_len < my_len) { my_len = o_len; my_idx = o_idx; }
    }
    if (lane == 0) { /* global minimum in my_len, my_idx */ }
}
__syncthreads();
```

Total step count: 5 (level 1) + 1 (write to shared) + 1 (__syncthreads) + 5 (level 2) = 12 steps for any P ≤ 1024. Compare to the serialized O(P²) = 496 steps for P = 32 or 32,640 steps for P = 256.

### 14.4 Summary: when to use each primitive

| Scenario | Tool | Constraint |
|---|---|---|
| Reduction within one warp | `__shfl_xor_sync` | BLOCK_POP_SIZE ≤ 32 |
| Reduction across warps in a block | Shuffle → shared → shuffle | BLOCK_POP_SIZE ≤ 1024 |
| Reduction across blocks | Second kernel pass | Unlimited |
| Scan (prefix sum) within warp | `__shfl_up_sync` | Width ≤ 32 |
| Block-wide scan | CUB `BlockScan` | Production-quality, 1024 threads |

> **Practical advice:** For anything that must scale beyond BLOCK_POP_SIZE = 32, use NVIDIA's **CUB library** (`cub::WarpReduce`, `cub::BlockReduce`). CUB implements all of these patterns correctly and handles edge cases, and it compiles into the same PTX you would write by hand.

---

## Chapter 15 — Constant Memory and the L2 Cache Argument: "When the Cache Makes Constant Memory Redundant"

### 15.1 The capacity argument revisited

Chapter 4 argued that constant memory hurts for scatter reads because warp threads access different addresses, forcing serialization. There is a complementary argument that further limits constant memory's role for this specific problem.

For `n = 128`, the full distance matrix is:
```
128 × 128 × 4 bytes = 65,536 bytes = 64 KB
```

The P100's L2 cache is **4 MB**. The entire distance matrix uses **1.6% of L2**. Once the matrix is warm in L2 (after the first few tours are evaluated), every subsequent thread that touches `dist[]` from global memory will find it already cached. No DRAM fetch required.

### 15.2 Comparing constant cache vs L2 for this workload

| Memory space | Cache size | Access model | Penalty for scatter |
|---|---|---|---|
| Constant (`__constant__`) | 8 KB/SM | Broadcast: 1 cycle if uniform; serialized if divergent | Full serialization per unique address |
| Global (read-only `__restrict__`) | L2: 4 MB shared; L1: 32 KB/SM | Load/store cache; independent per thread | One L2 hit per lane ≈ same latency |
| Global (cached in L2) | L2: 4 MB | Same as above | Parallel L2 lookups, not serialized |

For scatter reads from a 64 KB matrix that fits in L2:
- **Constant memory** serializes 32 unique addresses → 32 sequential lookups × ~10 cycles = ~320 cycles
- **Global memory + L2** handles 32 unique addresses in parallel (each lane independently issues an L2 request) → ~30–80 cycles total (multiple requests in flight simultaneously via the memory pipeline)

**Conclusion:** For n ≤ ~180 (where `n² × 4 ≤ 4 MB × 0.25`), a standard `const int* __restrict__ dist` pointer in global memory with L2 warming is competitive with or superior to constant memory, without any serialization risk.

### 15.3 When constant memory does make sense for TSP

Constant memory excels when **all threads in a warp read the same address** in the same instruction — the hardware then performs a single fetch and broadcasts. In your code, this happens for scalar kernel parameters:
- `n` (all threads read it identically)
- `mutation_rate` (all threads read it identically)
- `generations`, `elite_count` (same)

These are already passed as kernel arguments, which the compiler typically places in constant memory automatically. The explicit `__constant__` region for `c_dist` is where the caution applies.

### 15.4 The N-scaling crossover point

As N grows beyond 128, the matrix eventually overflows the L2:
```
L2 = 4 MB → max matrix: sqrt(4 MB / 4) = 1024 cities
```

For N ≤ 1024, the full matrix fits in L2. Above N = 1024, the matrix no longer fits, L2 misses become frequent, and a tiling strategy into shared memory becomes genuinely justified — but only then.

> **Decision rule:** Use `const int* __restrict__ dist` in global memory with L2 warming for N ≤ 1024. Consider shared-memory tiling for N > 1024. Do not use `__constant__` unless the access pattern is uniform across the warp.

---

## Chapter 16 — Profiling Plan: Corroborating Predictions with nvprof and Nsight Compute

> **Goal of this chapter:** Every claim in the preceding chapters is a *hypothesis*. A hypothesis is only an engineering conclusion after it is confirmed by a profiler. This chapter maps each bottleneck to a specific metric, a specific command, and a specific expected value. If the profiler disagrees with the prediction, the model was wrong — not the profiler.

### 16.0 Tool Setup

**nvprof** (legacy, works on P100 / CC 6.0):
```bash
nvprof --metrics <comma-separated-list> ./<binary> <args>
nvprof --print-gpu-trace ./<binary> <args>        # timeline view
```

**Nsight Compute** (ncu, modern, also works on P100):
```bash
ncu --metrics <section-or-metric-list> ./<binary> <args>
ncu --set full ./<binary> <args>                  # all metrics, expensive
ncu --section MemoryWorkloadAnalysis ./<binary> <args>
```

**nvcc register info** (no profiler needed):
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O2 -o out file.cu 2>&1 | grep "registers\|lmem\|smem"
```
This prints register count, local memory, and shared memory *per kernel* at compile time — use this before profiling to get the static resource picture.

---

### 16.1 Hypothesis: AoS tour reads are poorly coalesced

**Predicted metric value:** Global load efficiency ≈ 3.1% (one useful int per 32-int cache line)

**nvprof command:**
```bash
nvprof --metrics gld_efficiency,gld_transactions,gld_transactions_per_request \
       ./cuda_ga benchmark.tsp 512 1 0.05 4
#                              ^pop ^1gen — use 1 gen to isolate kernel cost
```

**What to look for:**

| Metric | Expected (AoS, n=128) | Expected (SoA fix) |
|---|---|---|
| `gld_efficiency` | ~3% | ~90–100% |
| `gld_transactions_per_request` | ~32 (one cacheline per element) | ~1 |
| `l2_read_hit_rate` | Low on first run; higher after warmup | Higher and stable |

**Nsight Compute equivalent:**
```bash
ncu --metrics \
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  ./cuda_ga benchmark.tsp 512 1
# Efficiency = requests / sectors; sectors >> requests → bad coalescing
```

**Decision gate:** If `gld_efficiency > 80%`, the coalescing fix is working. If it remains below 20% after the SoA change, verify the transpose was applied to the correct buffer (tours vs dist).

---

### 16.2 Hypothesis: 32-way shared memory bank conflicts in GPU-Pop

**Predicted metric value:** `shared_load_transactions_per_request = 32` (worst case) for the population reads in `ga_island_kernel`

**nvprof command:**
```bash
nvprof --metrics \
  shared_efficiency,\
  shared_load_transactions_per_request,\
  shared_store_transactions_per_request \
  ./cuda_ga_gpu_pop benchmark.tsp 128 1000
```

**What to look for:**

| Metric | Before padding (n=128, stride=128) | After padding (stride=129) |
|---|---|---|
| `shared_efficiency` | ~3% (1/32) | ~97–100% |
| `shared_load_transactions_per_request` | 32 | 1 |
| `shared_store_transactions_per_request` | 32 | 1 |

**Nsight Compute equivalent:**
```bash
ncu --section SharedMemory ./cuda_ga_gpu_pop benchmark.tsp 128 1000
# Reports: "Shared Memory Bank Conflicts" and "Wavefronts"
# Target: 0 bank conflicts after the stride fix
```

**Manual sanity check before profiling:**  
At compile time, confirm shared memory per block matches expectations:
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O2 -o gpu_pop CUDA-GA-GPU-Pop.cu tsplib_parser.cpp
# Look for: "Used 33024 bytes of shared data" (before padding)
#           "Used 33280 bytes of shared data" (after stride=129 padding)
```

**Decision gate:** `shared_load_transactions_per_request` must drop from ~32 to ~1. If it stays at 32 after adding the stride pad, check that every access site uses `tid * stride + k` not `tid * n + k`.

---

### 16.3 Hypothesis: Hidden local memory traffic from `bool used[MAX_CITIES]`

**Predicted behavior:** Nonzero `local_load_transactions` and `local_store_transactions` attributed to `ga_island_kernel`

**nvprof command:**
```bash
nvprof --metrics \
  local_load_transactions,\
  local_store_transactions,\
  l1_cache_local_hit_rate \
  ./cuda_ga_gpu_pop benchmark.tsp 128 1000
```

**What to look for:**

| Metric | Before bitmask fix | After bitmask fix |
|---|---|---|
| `local_load_transactions` | > 0 (spilled `used[]` reads) | 0 |
| `local_store_transactions` | > 0 (spilled `used[]` writes) | 0 |

**Compile-time check (do this first):**
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O2 -o gpu_pop CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep lmem
# "lmem = 512" → 512 bytes of local memory per thread (128 bools × 4 bytes aligned)
# "lmem = 0"   → no local memory spills after bitmask fix
```

The `lmem` value in the compiler output is the fastest way to confirm local memory spills without even running the kernel. **If `lmem > 0` for `ga_island_kernel`, the `used[]` array is still spilling.** This is the most direct check.

**Computing expected local memory traffic:**  
With `lmem = 512` bytes, 30 crossover threads per block, 128 blocks, 1000 generations, and N = 128 iterations per crossover:
```
Reads:  30 × 128 × 1000 × 128 × 4 bytes ≈ 1.97 GB   (per run)
Writes: same                              ≈ 1.97 GB
Total:  ≈ 3.93 GB of local memory traffic
```
This should be directly visible in the `local_load_transactions` count.

**Decision gate:** After the bitmask fix, `lmem = 0` in ptxas output AND `local_load_transactions = 0` in nvprof.

---

### 16.4 Hypothesis: Low occupancy due to shared memory pressure

**Predicted value:** Achieved occupancy ≈ 1.56% (1 warp active per SM)

**nvprof command:**
```bash
nvprof --metrics \
  achieved_occupancy,\
  warp_execution_efficiency,\
  eligible_warps_per_cycle \
  ./cuda_ga_gpu_pop benchmark.tsp 128 1000
```

**What to look for:**

| Metric | Expected (current) | Target (after block expansion) |
|---|---|---|
| `achieved_occupancy` | ~0.016 (1.56%) | ~0.50 (50%) |
| `warp_execution_efficiency` | ~100% (only 1 active warp) | ~70–85% |
| `eligible_warps_per_cycle` | ~1 | ~15–20 |

> **Note:** `warp_execution_efficiency` near 100% with low occupancy is a warning sign, not good news. It means "the one warp that is active is always eligible — because there are no other warps to hide behind when it stalls."

**Nsight Compute occupancy analysis:**
```bash
ncu --section Occupancy ./cuda_ga_gpu_pop benchmark.tsp 128 1000
# Reports: theoretical vs achieved occupancy
# Shows: limiting factor (register count? shared memory? thread count?)
```

**Occupancy Calculator sanity check:**  
Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` in code to query this without a profiler:
```cpp
int maxBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, ga_island_kernel, BLOCK_POP_SIZE, shared_bytes);
printf("Max active blocks per SM: %d\n", maxBlocks);
printf("Max active warps per SM: %d\n", maxBlocks * (BLOCK_POP_SIZE / 32));
```

**Decision gate:** After expanding to 1024 threads/block with the warp-per-individual mapping, `achieved_occupancy` should exceed 0.40.

---

### 16.5 Hypothesis: Branch divergence in OX crossover

**Predicted value:** `branch_efficiency` < 80% inside `order_crossover_device`

**nvprof command:**
```bash
nvprof --metrics \
  branch_efficiency,\
  warp_nonpred_exec_efficiency \
  ./cuda_ga_gpu_pop benchmark.tsp 128 1000
```

**What to look for:**

| Metric | Meaning | Expected (OX crossover) | Expected (branchless PMX) |
|---|---|---|---|
| `branch_efficiency` | % of branch instr that are non-divergent | 50–70% | ~95% |
| `warp_nonpred_exec_efficiency` | % of active lanes executing vs predicated-off | 70–85% | ~95% |

**Nsight Compute equivalent (more precise):**
```bash
ncu --metrics \
  smsp__sass_branch_targets_threads_divergent.sum,\
  smsp__sass_branch_targets_threads_uniform.sum \
  ./cuda_ga_gpu_pop benchmark.tsp 128 1000
# Divergence ratio = divergent / (divergent + uniform)
```

**Isolating the kernel function:**  
The divergence metric is reported per-kernel, not per-function. To isolate `order_crossover_device`, you may need to extract it into its own `__global__` wrapper for profiling, or use `ncu --kernel-name` to filter by kernel name:
```bash
ncu --kernel-name ga_island_kernel --metrics branch_efficiency ./gpu_pop bench.tsp
```

**Decision gate:** If `branch_efficiency > 90%`, divergence is not a significant bottleneck. If it is below 70%, investigate the OX `continue` path and consider the PMX replacement.

---

### 16.6 Hypothesis: PCIe transfers dominate CUDA-GA runtime

**Predicted value:** ~22 ms of total memcpy time out of a ~1-second run with 1000 generations

**nvprof timeline command:**
```bash
nvprof --print-gpu-trace ./cuda_ga benchmark.tsp 512 1000
```

**What to look for in the trace:**

```
  Start  Duration  Name
  ...
  123us    21.3ms  [CUDA memcpy HtoD]  ← population upload, should repeat 1000×
  123us     0.08ms [CUDA kernel]        ← kernel is tiny vs transfer
  124us     0.2ms  [CUDA memcpy DtoH]  ← lengths download
```

Sum all `[CUDA memcpy HtoD]` and `[CUDA memcpy DtoH]` durations. Compare to total runtime.

**Expected ratio:**
```
Transfer time: ~22 ms (predicted)
Kernel time:   ~80 µs × 1000 = ~80 ms (predicted)
Total GPU work: ~100 ms
Fraction spent in memcpy: ~22%
```

**Nsight Systems** (system-level view, not kernel-level):
```bash
nsys profile --trace=cuda,osrt ./cuda_ga benchmark.tsp 512 1000
nsys-ui report1.nsys-rep   # open in Nsight Systems GUI
```
In the timeline view, alternating green (kernel) and orange (memcpy) bars make the transfer overhead immediately visible.

**Decision gate:** If memcpy segments are visually prominent and total more than 15% of GPU timeline, the hybrid architecture is the bottleneck. The GPU-Pop island kernel should show a clean, uninterrupted kernel bar.

---

### 16.7 Combined Profiling Run: One Script to Measure All Hypotheses

```bash
#!/bin/bash
# profile_all.sh — run on UA HPC with P100 (sm_60), account u16
# Usage: sbatch profile_all.sh

#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=u16

module load cuda/11.8

BIN_NAIVE=./GPU-Naive
BIN_HYBRID=./CUDA-GA
BIN_ISLAND=./CUDA-GA-GPU-Pop
TSP=./benchmark.tsp
POP=512
GENS=1000

echo "=== [1] Coalescing: AoS global load efficiency (hybrid) ==="
nvprof --metrics gld_efficiency,gld_transactions_per_request \
       $BIN_HYBRID $TSP $POP 1 0.05 4 2>&1 | grep -E "gld_efficiency|gld_transactions"

echo "=== [2] Shared memory bank conflicts (island) ==="
nvprof --metrics shared_efficiency,shared_load_transactions_per_request \
       $BIN_ISLAND $TSP 128 100 2>&1 | grep -E "shared_efficiency|transactions_per_request"

echo "=== [3] Local memory spills (island) ==="
nvprof --metrics local_load_transactions,local_store_transactions \
       $BIN_ISLAND $TSP 128 100 2>&1 | grep -E "local_"

echo "=== [4] Occupancy (island) ==="
nvprof --metrics achieved_occupancy,warp_execution_efficiency \
       $BIN_ISLAND $TSP 128 $GENS 2>&1 | grep -E "occupancy|warp_exec"

echo "=== [5] Branch divergence (island) ==="
nvprof --metrics branch_efficiency,warp_nonpred_exec_efficiency \
       $BIN_ISLAND $TSP 128 $GENS 2>&1 | grep -E "branch|nonpred"

echo "=== [6] PCIe transfer timeline (hybrid) ==="
nvprof --print-gpu-trace $BIN_HYBRID $TSP $POP $GENS 2>&1 | \
       awk '/memcpy/{sum += $2} END {print "Total memcpy time: " sum}'

echo "=== [7] Compile-time local memory per kernel ==="
nvcc -arch=sm_60 --ptxas-options=-v -O2 -o /dev/null \
     CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep -E "lmem|smem|registers"

echo "Done."
```

---

### 16.8 Results Interpretation Matrix

Use this table to triage profiler output. Each row is one measurement, with a verdict and next action:

| Metric | Measured Value | Verdict | Next Action |
|---|---|---|---|
| `gld_efficiency` | < 10% | AoS confirmed critical | Apply SoA transpose |
| `gld_efficiency` | 80–100% | Coalescing is fine | Move to next bottleneck |
| `shared_load_transactions_per_request` | ~32 | Bank conflict confirmed | Add stride padding |
| `shared_load_transactions_per_request` | ~1 | No bank conflicts | Move to next bottleneck |
| `lmem` (ptxas) | > 0 | Local memory spill confirmed | Replace `used[]` with bitmask |
| `lmem` (ptxas) | 0 | No spills | Move to next bottleneck |
| `achieved_occupancy` | < 0.05 | Occupancy critically low | Evaluate larger block sizes |
| `achieved_occupancy` | > 0.40 | Good occupancy | Move to next bottleneck |
| `branch_efficiency` | < 70% | Divergence is real | Profile OX loop, consider PMX |
| `branch_efficiency` | > 90% | Divergence is negligible | Move to next bottleneck |
| memcpy fraction of GPU time | > 15% | PCIe is a bottleneck | Move to island model |
| memcpy fraction of GPU time | < 5% | Transfer is not the issue | Profile kernel internals |

---

### 16.9 The Profiling Principle

> A profiler does not tell you what to fix. It tells you whether your mental model of the hardware is correct.

The correct workflow is:

```
1. Write down the prediction:
   "I expect shared_load_transactions_per_request ≈ 32 because n=128
    causes all threads to map to the same bank."

2. Run the profiler.

3. If measured ≈ predicted → your model is correct, apply the fix.

4. If measured ≠ predicted → your model has a gap, investigate before fixing.

5. After the fix, re-run the profiler and confirm the metric improved.
```

Never apply an optimization and trust that it worked without measuring the before and after metric. The compiler may have already mitigated the issue (in which case the optimization is a no-op), or your fix may have introduced a new bottleneck that erases the gain.

---

## Revised Optimization Roadmap: Low-Risk → Experimental

Based on all chapters, the optimizations are now stratified by confidence and effort:

### Stage 1 — Low-risk, high-confidence (do these first)
Each has a compile-time or single-metric profiler check.

1. **Transpose tours to SoA** — check: `gld_efficiency` → 90%+
2. **Pad shared stride to `n+1`** — check: `shared_load_transactions_per_request` → 1
3. **Replace `used[]` with 4-register bitmask** — check: `lmem` → 0 in ptxas
4. **Keep population GPU-resident (island model)** — check: no memcpy in GPU trace

### Stage 2 — Structural improvements (moderate risk)
These change the kernel's thread mapping or occupancy model.

5. **Replace serialized sort with parallel reduction** — check: `achieved_occupancy` improvement
6. **Use warp shuffle for elite search** (BLOCK_POP_SIZE ≤ 32 only) — check: `branch_efficiency` in sort code
7. **Two-level shuffle+shared reduction** (if BLOCK_POP_SIZE scales beyond 32) — check: `eligible_warps_per_cycle`

### Stage 3 — Experimental (measure ROI before committing)
These are architecturally significant redesigns.

8. **Warp-per-individual block expansion** (1024 threads/block) — risk: shared memory layout overhaul; potential reward: 50% occupancy
9. **Replace OX with PMX crossover** — risk: genetic quality change; reward: 100% branch efficiency in crossover
10. **Shared-memory tiling of dist for N > 1024** — only relevant for problem sizes beyond current MAX_CITIES limit

---

*Append these chapters to the main document. Profile before and after each Stage 1 change. Validate Stage 2 with `achieved_occupancy` before investing in Stage 3.*