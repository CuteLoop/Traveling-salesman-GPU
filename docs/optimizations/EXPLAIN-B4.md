# B4 — Distance Matrix Memory Placement

## The Problem: Constant Memory is a Broadcast Cache, Not a Scatter Cache

The baseline kernel stores the distance matrix in `__constant__` memory:

```c
__constant__ int c_dist[MAX_CITIES * MAX_CITIES];   // 64 KB
// inside tour_length_const():
total += c_dist[a * n + b];
```

Constant memory on the P100 has an **8 KB broadcast cache per SM**.
Its hardware is optimized for exactly one access pattern:

> All 32 threads in a warp read the **same address** → one fetch serves all 32 threads.

This is called a **broadcast**. It is extremely efficient.

### What happens instead: scatter

In `tour_length_const()`, every thread follows a *different* tour.
At step `k`, thread 0 may need `c_dist[3*128 + 17]` while thread 1 needs
`c_dist[8*128 + 52]`, thread 2 needs `c_dist[11*128 + 4]`, and so on.
Every thread needs a **different address**. This is called a **scatter**.

The constant cache hardware handles scatter by **serializing** the requests:
it services thread 0, then thread 1, then thread 2 … then thread 31.
That is 32 sequential fetches from a single warp instruction.

Penalty estimate for one tour-length evaluation (n=128 edges, 32 threads):
```
32 threads × 128 edges × 1 serialized fetch/thread
= 4096 sequential fetches per tour evaluation per warp step
```

Across 128 islands × 1000 generations × (32 non-elite tours):
```
128 × 1000 × 32 × 4096 fetches ≈ 1.7 × 10¹⁰ serialized constant-cache requests
```

Even at ~10 cycles/hit, that is ~170 billion wasted cycles of serialization.

---

## The Fix: Two Alternatives

### B4-global — Move dist to global memory with `__restrict__`

```c
// kernel signature change:
__global__ void ga_island_kernel(int n, int generations, float mutation_rate,
                                  const int* __restrict__ dist, ...)
// inside tour_length_global():
total += dist[a * n + b];
```

**Why global memory is better here:**

Global memory scatter requests are **issued in parallel**.
All 32 threads submit their requests simultaneously to the L2 cache.
The L2 has multiple ports and can service multiple requests per cycle.

The P100's L2 cache is **4 MB**. The distance matrix is `128×128×4 = 64 KB`,
which is only **1.6% of L2 capacity**. After a handful of warm-up tours,
the entire distance matrix lives in L2 and all subsequent reads are L2 hits.

The `__restrict__` qualifier tells the compiler this pointer is not aliased
by any other pointer in the kernel. This enables:
- Compiler-generated prefetch hints (`ld.ca` / `ld.cg` instructions)
- Potentially using the texture read path (LDG instruction), which is
  L1/L2 cached and read-only — exactly the access pattern here

**What changes in the host code:**

```c
// Before (baseline):
cudaMemcpyToSymbol(c_dist, inst.dist.data(), sizeof(int)*inst.dist.size());

// After (B4-global):
int* d_dist = nullptr;
cudaMalloc(&d_dist, sizeof(int) * inst.dist.size());
cudaMemcpy(d_dist, inst.dist.data(), sizeof(int)*inst.dist.size(),
           cudaMemcpyHostToDevice);
// ... pass d_dist as kernel argument ...
cudaFree(d_dist);
```

### B4-smem — Move dist into shared memory (only for n ≤ 99)

```c
extern __shared__ int smem[];
// layout: [pop_a | pop_b | lengths | order | s_dist]
int* s_dist = (int*)(smem + pop_smem_ints + BLOCK_POP_SIZE + BLOCK_POP_SIZE);

// cooperative load at kernel startup (once per island lifetime):
for (int i = tid; i < n * n; i += BLOCK_POP_SIZE)
    s_dist[i] = d_dist[i];
__syncthreads();

// inside tour_length_smem():
total += s_dist[a * n + b];
```

**Why shared memory is the fastest option:**

Shared memory is on-chip SRAM — same chip as the CUDA cores.
Access latency is ~1–4 cycles with **no serialization for scatter**.
Once the distance matrix is loaded into shared memory, all 128×128
entries are served directly from the register file's neighbor with zero
L2 or DRAM traffic.

---

## Shared Memory Capacity Derivation

The P100 has 64 KB of shared memory per SM (software-configurable; we use the full 64 KB).

The kernel already occupies shared memory for:

| Region | Size (bytes) |
|---|---|
| Pop A: `BLOCK_POP_SIZE` tours of `n` ints, stride `n+1` | `32 × (n+1) × 4` |
| Pop B: same | `32 × (n+1) × 4` |
| `lengths[BLOCK_POP_SIZE]` | `32 × 4` |
| `order[BLOCK_POP_SIZE]` | `32 × 4` |
| **Population subtotal** | `256(n+1) + 256` bytes |

Adding the distance matrix:

| Region | Size (bytes) |
|---|---|
| `s_dist[n * n]` | `4n²` |

**Total shared memory constraint:**

```
256(n+1) + 256 + 4n² ≤ 65,536
256n + 256 + 256 + 4n² ≤ 65,536
4n² + 256n + 512 ≤ 65,536
4n² + 256n ≤ 65,024
n² + 64n ≤ 16,256
(n + 32)² ≤ 16,256 + 1,024 = 17,280
n + 32 ≤ 131.45
n ≤ 99.45
```

**Therefore: `SMEM_DIST_MAX_N = 99`.**

| n | Pop bytes | Dist bytes | Total | Fits? |
|---|---|---|---|---|
| 99 | 25,856 | 39,204 | 65,060 | ✓ |
| 100 | 26,112 | 40,000 | 66,112 | ✗ |
| 128 | 33,280 | 65,536 | 98,816 | ✗ |

For `n = 128` (the production case in this GA), **B4-smem does not fit**.
Use B4-global instead.

The B4-smem kernel enforces this at runtime:
```c
if (n > SMEM_DIST_MAX_N)
    throw std::runtime_error("B4-smem requires n <= 99");
```

---

## How to Read the Benchmark Output

### Static analysis (ptxas `--ptxas-options=-v`)

Look for the line:
```
ptxas info: Used X registers, Y bytes smem, Z bytes lmem
```

- **B3-shuffle** and **B4-global** should report the **same smem** (dist is not in smem).
- **B4-smem** should report `smem ≈ 65,060` bytes for `n=99`, or the appropriate value for your fixture's `n`.
- `lmem` should be 0 in all three (B2 already eliminated local memory spill).

### nvprof L2 hit rate

```
l2_read_hit_rate  ≈ ?%
```

- **B4-global**: expect a high hit rate (80–99%) once the L2 is warm.
  The entire 64 KB matrix fits in 1.6% of L2; it will not be evicted.
- **B3-shuffle**: the constant cache is only 8 KB/SM; the 64 KB matrix
  does not fit. Expect repeated constant-cache misses → L2 pressure.
- **B4-smem**: dist reads never reach L2; the metric will be lower
  simply because fewer L2 transactions are issued.

### nvprof L2 read transactions

- **B4-smem** < **B4-global** < **B3-shuffle**.
  B4-smem has essentially zero dist-related L2 reads after the initial load.
  B4-global has parallel in-flight L2 requests, reducing apparent latency.
  B3-shuffle has serialized requests that create a longer pipeline of L2 hits.

### Wall-clock time

Expected ranking for `n = 128`:
```
B4-global  <  B3-shuffle
```

Expected ranking for `n ≤ 99`:
```
B4-smem  <  B4-global  <  B3-shuffle
```

If B4-global is *slower* than B3-shuffle for very small `n` (e.g., `n=20`),
the constant cache may be large enough to broadcast the entire matrix without
conflict. In that case the serialization penalty is hidden by cache capacity.
Always benchmark at `n=128` for production conclusions.

---

## Key Takeaway

Constant memory is a **broadcast cache**. Use it when all threads in a warp
read the *same* address (e.g., a scalar parameter broadcast to all threads).
Do **not** use it for scatter access patterns (each thread reads a different
index), because the hardware serializes those requests, turning a single warp
instruction into 32 sequential fetches.

For scatter reads of a matrix that fits in L2 (≤ a few percent of L2 capacity),
global memory with `__restrict__` is superior: requests are issued in parallel,
L2 handles multiple outstanding transactions, and the entire matrix stays warm
in L2 for the lifetime of the kernel.

For the smallest `n` where the full matrix fits in shared memory (`n ≤ 99`),
shared memory eliminates off-chip traffic entirely, providing the lowest
possible latency and highest throughput for distance lookups.

---

## Cumulative Fix Chain

```
V0 (baseline)
 └─ B1: stride = n+1         → 32-way shared memory bank conflict eliminated
     └─ B2: register bitmask → lmem 512 B → 0, 3.93 GB DRAM traffic removed
         └─ B3-shuffle:       → O(P²) serial sort → 10 XOR-shuffle steps (warp-parallel)
             └─ B4-global:    → const scatter → global __restrict__ (parallel L2 requests)
             └─ B4-smem:      → const scatter → shared memory dist (n ≤ 99; zero L2 traffic)
```
