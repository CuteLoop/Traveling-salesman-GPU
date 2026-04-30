# EXPLAIN-B1: Shared Memory Bank Conflicts and the Stride Padding Fix

## What is this bottleneck?

Every CUDA kernel that touches shared memory goes through the **shared memory arbiter**, which manages 32 independent banks. The P100's banks are 4 bytes wide. Any element at byte offset `b` in shared memory lives in bank `(b / 4) % 32`.

When a warp (32 threads) issues a shared memory instruction:
- If all 32 threads access **different** banks → 1 transaction, all served in parallel.
- If multiple threads access the **same** bank → those accesses are **serialized** (one per cycle slot).

A 32-way conflict means the hardware turns one logical instruction into 32 sequential memory transactions.

## Where is the conflict in our kernel?

The population array is laid out in shared memory as a flat 1D array of `int`:

```
pop_a[tid * n + k]
     ^^^^^^^^^^^^
     thread t, city k
```

The bank for element `pop_a[t * n + k]`:

```
bank(t, k) = (t * n + k) % 32
```

For `n = 128`:

```
bank(t, k) = (t * 128 + k) % 32
           = (t * (128 % 32) + k) % 32
           = (t * 0 + k) % 32
           = k % 32
```

**The bank index does not depend on the thread index `t` at all.** Every warp thread accessing city `k` hits the same bank, regardless of which individual (row) it belongs to. With 32 threads all reading city `k` of their respective individuals, all 32 go to bank `k % 32` — a perfect 32-way collision. The hardware serializes: instead of 1 cycle, the access takes 32 cycles.

This happens for any `n` that is a multiple of 32: `n = 64, 96, 128, 192, ...`

## The fix: pad the stride to `n + 1`

Add one unused `int` of padding at the end of each individual's row:

```
stride = n + 1 = 129   (for n = 128)
pop_a[tid * stride + k]
```

New bank formula:

```
bank(t, k) = (t * 129 + k) % 32
           = (t * (129 % 32) + k) % 32
           = (t * 1 + k) % 32
           = (t + k) % 32
```

Now thread 0 hits bank `k`, thread 1 hits bank `k+1`, thread 2 hits bank `k+2`, and so on. All 32 banks are distinct. Zero conflicts.

### Verification: why `129 % 32 = 1` matters

The key property is that the stride must be **coprime with 32** (i.e., `gcd(stride, 32) = 1`), so that as `t` increments, the row starting addresses cycle through all 32 bank residues before repeating.

- `stride = 128`: `128 % 32 = 0` → 32-way conflict
- `stride = 129`: `129 % 32 = 1` → 0 conflicts ✓
- `stride = 130`: `130 % 32 = 2` → gcd(2,32)=2 → 2-way conflict (still bad)
- `stride = 133`: `133 % 32 = 5` → gcd(5,32)=1 → 0 conflicts ✓

The `n + 1` rule works for any `n` that is a multiple of 32, since `(32k + 1)` is always odd → gcd with 32 is 1.

**Edge case**: if `n % 32 = 31` (e.g., `n = 31, 63, 95, 127`), then `n + 1` is a multiple of 32 and we'd get a 32-way conflict again. For our codebase with `n = 128`, this edge case does not apply.

## Memory cost of padding

```
Before padding:  2 × 32 × 128 × 4 = 32,768 bytes (pop arrays)
After padding:   2 × 32 × 129 × 4 = 33,024 bytes
Delta:           256 bytes (0.4% of shared memory)
```

Negligible. Occupancy is already limited by the total shared memory footprint (~33 KB out of 64 KB), not this 256-byte delta.

## Every access site must be updated

The stride change must propagate to **every** read and write of the population arrays inside the kernel:

| Access site | Before | After |
|---|---|---|
| `init_random_tour` call | `pop_a + tid * n` | `pop_a + tid * stride` |
| `tour_length_const` call | `current + tid * n` | `current + tid * stride` |
| Elite copy | `next[tid * n + k]` | `next[tid * stride + k]` |
| `order_crossover_device` call | `current + pa * n` | `current + pa * stride` |
| `mutate_swap_device` call | `next + tid * n` | `next + tid * stride` |
| Final output | `current[best_idx * stride + k]` → `best_tours[island * n + k]` | ← output remains packed `n`, not `stride` |

The output array `best_tours` in global memory is packed (stride `n`), since it's just a dense array of integers for the host to read. Only the **internal shared memory** layout uses stride `n+1`.

## How to read the profiler output

After running `bench_b1.slurm`, look for these fields:

```
shared_load_transactions_per_request
```

- **Expected V0**: ~32 (each load request generates 32 serialized transactions)
- **Expected B1**: ~1 (each load request is a single transaction)

If you see a value between 1 and 32, it may indicate a partial conflict (e.g., `n` not a multiple of 32, or mixed access patterns).

The `ptxas` report shows:
```
smem = 33024   (V0) → 33280   (B1)   # +256 bytes, as expected
lmem = 512     (unchanged)            # B2 not yet applied, used[] still spills
```

## Why does this matter so much?

The `ga_island_kernel` touches shared memory on virtually every operation:
- Every fitness evaluation: 128 reads from the tour (= from shared memory)
- Every elite copy: 128 reads + 128 writes
- Every crossover: 128 reads from each parent, 128 writes to child

For 128 islands × 1000 generations × 32 threads, the 32× serialization penalty
compounds into hundreds of millions of wasted memory cycles. Resolving it should
produce a proportional reduction in kernel execution time for the memory-bound phases.
