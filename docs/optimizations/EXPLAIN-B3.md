# EXPLAIN-B3: Eliminating Thread-0 Serialization — Two Reduction Strategies

## The problem

The natural selection phase in the original kernel:

```cpp
if (tid == 0) {
    for (int i = 0; i < BLOCK_POP_SIZE; ++i) order[i] = i;
    for (int i = 0; i < BLOCK_POP_SIZE - 1; ++i) {
        int best = i;
        for (int j = i + 1; j < BLOCK_POP_SIZE; ++j)
            if (lengths[order[j]] < lengths[order[best]]) best = j;
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
    }
}
__syncthreads();
```

Two compounding inefficiencies:

**1. Algorithm is O(P²).** Selection sort on P=32 elements requires P(P-1)/2 = 496 comparisons. We only need the top `elite_count = 2` indices, which requires at most 31 comparisons.

**2. 31 threads are completely idle.** Since BLOCK_POP_SIZE = 32 = one warp, there are zero other warps in flight. The GPU scheduler has no work to hide this latency. Every cycle during the sort is wasted for 31 of 32 functional units.

```
Serial comparisons per full run:
  496 × 128 islands × 1000 generations = 63,488,000 comparisons
  All single-threaded. 31/32 of the warp's compute capability unused.
```

## What we actually need

We do not need a full ranking. We need:
- The index of the **best** individual (minimum `lengths[tid]`)
- The index of the **second-best** (minimum of the remaining 31)

This is a **parallel min-reduction** problem, not a sort problem.

---

## Solution A: XOR-Shuffle Warp Reduction (B3-shuffle)

### The hardware mechanism

`__shfl_xor_sync(mask, val, xorMask)` is a single instruction that:
- Takes `val` from each of the 32 lanes
- Each lane `i` receives the value from lane `i XOR xorMask`
- This is warp-synchronous: the exchange completes before the next instruction
- No shared memory involved — values travel through the warp's dedicated **register file crossbar**
- Latency: ~4 cycles

### The XOR tree pattern

To find the minimum over 32 values in 5 steps:

```
Initial: each lane holds (my_len = lengths[tid], my_idx = tid)

Step 1 (xorMask=16):
  Lane 0  exchanges with lane 16
  Lane 1  exchanges with lane 17
  ...
  Lane 15 exchanges with lane 31
  Each lane keeps the smaller (len, idx) pair.

Step 2 (xorMask=8): similar, halving the "unresolved" range again.
Step 3 (xorMask=4)
Step 4 (xorMask=2)
Step 5 (xorMask=1)

After 5 steps: every lane holds the global minimum's (len, idx).
```

The key insight is that after each XOR step, the minimum of the two exchanged values propagates to both participants. After `log2(32) = 5` steps, the minimum has propagated to every lane simultaneously.

**Code:**
```cpp
for (int xor_mask = 16; xor_mask > 0; xor_mask >>= 1) {
    int other_len = __shfl_xor_sync(0xFFFFFFFF, my_len, xor_mask);
    int other_idx = __shfl_xor_sync(0xFFFFFFFF, my_idx, xor_mask);
    if (other_len < my_len || (other_len == my_len && other_idx < my_idx)) {
        my_len = other_len;
        my_idx = other_idx;
    }
}
int elite0_idx = __shfl_sync(0xFFFFFFFF, my_idx, 0); // broadcast from lane 0
```

### Pass 2: finding elite1

Set lane `elite0_idx`'s length to `INT_MAX`, then run the same 5-step tree:

```cpp
my_len = (threadIdx.x == elite0_idx) ? INT_MAX : lengths[threadIdx.x];
my_idx = threadIdx.x;
// ... same 5-step XOR tree ...
int elite1_idx = __shfl_sync(0xFFFFFFFF, my_idx, 0);
```

**Total cost:** 2 passes × 5 shuffle steps = 10 warp-synchronous shuffle instructions. No `__syncthreads()` needed between steps (warp is implicitly synchronous). No shared memory used.

---

## Solution B: Classical Binary Tree Reduction (B3-reduce)

### The algorithm

Use the existing `order[]` shared memory array as an index scratch space:

```
Initial:  order[0..31] = [0, 1, 2, ..., 31]

half=16: threads 0..15 active:
  thread t compares order[t] vs order[t+16]
  order[t] = whichever index has smaller length
  __syncthreads()

half=8: threads 0..7 active:
  thread t compares order[t] vs order[t+8]
  order[t] = winner
  __syncthreads()

half=4, half=2, half=1: similarly.

After half=1 + __syncthreads(): order[0] = index of global minimum.
```

### Step-by-step trace (P=32)

| Step | Active threads | Comparisons | Survivors |
|------|---------------|-------------|-----------|
| half=16 | 0..15 (16) | 16 | 16 indices in order[0..15] |
| half=8  | 0..7  (8)  | 8  | 8 indices in order[0..7]  |
| half=4  | 0..3  (4)  | 4  | 4 indices in order[0..3]  |
| half=2  | 0..1  (2)  | 2  | 2 indices in order[0..1]  |
| half=1  | 0     (1)  | 1  | 1 index in order[0]       |

Total: 31 comparisons (vs 496 for selection sort). But 5 `__syncthreads()` calls are required (one after each step).

### Two passes for top-2

Pass 1: find elite0 (as above). Save `elite0_idx`.  
Pass 2: reset `order[tid] = tid`, repeat the 5-step tree but treat `elite0_idx` as having `INT_MAX` length. This costs another 5 `__syncthreads()`.

Total: 10 `__syncthreads()` calls per generation (5 per pass) plus 2 more for ordering.

---

## Comparison: Shuffle vs Reduce

| Property | B3-shuffle | B3-reduce |
|----------|-----------|-----------|
| Algorithm | XOR-tree, 5 steps/pass | Binary tree, 5 steps/pass |
| Communication medium | Register file crossbar | Shared memory |
| Steps per step | 2 shuffle instructions | 1 comparison + 1 write |
| Synchronization needed | **None between steps** (warp-synchronous) | **1 `__syncthreads()` per step** |
| Total `__syncthreads()` for top-2 | 2 (post-pass broadcasts) | 10 (5 per pass) + 2 = 12 |
| Shared memory consumed | **None** (extra) | **None** (reuses `order[]`) |
| Extra registers | ~4 (my_len, my_idx, other_len, other_idx) | ~3 (ia, ib, elite0_idx) |
| Works for BLOCK_POP_SIZE > 32 | No (single warp only) | **Yes** (any power of 2) |
| Latency (cycles) per step | ~4 | ~10–20 (shared mem + sync overhead) |

### Why shuffle wins for BLOCK_POP_SIZE = 32

With exactly one warp, `__syncthreads()` still incurs overhead:
1. The compiler must emit a barrier instruction
2. The GPU hardware must wait for all threads to reach the barrier
3. The barrier flushes the instruction pipeline

With `__shfl_*`, there is no barrier — the shuffle is defined to be synchronous at the warp level by the ISA. The hardware issues it as a single instruction with a 4-cycle round-trip through the warp's register crossbar.

For BLOCK_POP_SIZE = 32 (our current setting), B3-shuffle is the correct choice. B3-reduce becomes necessary when `BLOCK_POP_SIZE > 32` (multi-warp blocks), because `__shfl_*` only communicates within a single warp.

### `__syncthreads()` in B3-reduce is not redundant

You might wonder: "With BLOCK_POP_SIZE = 32 = one warp, are the `__syncthreads()` calls in B3-reduce unnecessary?" **No.** The CUDA memory model requires `__syncthreads()` for correctness when communicating through shared memory, even within a single warp. Without the barrier, the hardware is permitted to reorder the store (from the previous iteration) and the load (in the next iteration). Warp-synchronous shared memory access is a known pitfall — valid for specific PTX patterns but not safe in general CUDA C++.

---

## Impact on `order[]`

After either reduction, the code must place `elite0_idx` in `order[0]` and `elite1_idx` in `order[1]`, so the downstream elite copy code (which reads `order[tid]`) works unchanged:

```cpp
// At end of reduction (both variants):
if (tid == 0) {
    order[0] = elite0_idx;
    order[1] = elite1_idx;
}
__syncthreads();
// ... then: if (tid < elite_count) { const int elite_idx = order[tid]; ... }
```

## Side benefit: cooperative global memory output

Both B3 variants also replace the `if (tid == 0)` final output copy with a cooperative loop:

```cpp
// Before (B2 and earlier): 128 sequential writes by thread 0
if (tid == 0)
    for (int k = 0; k < n; ++k)
        best_tours[island * n + k] = current[best_idx * stride + k];

// After (B3): 128/32 = 4 writes per thread, all in parallel
for (int k = tid; k < n; k += BLOCK_POP_SIZE)
    best_tours[island * n + k] = current[best_idx * stride + k];
```

This 32× parallelization of the output memcpy costs nothing extra and provides a free bonus speedup for the final output step.
