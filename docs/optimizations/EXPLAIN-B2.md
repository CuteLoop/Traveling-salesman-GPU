# EXPLAIN-B2: Local Memory Spill and the Register Bitmask Fix

## What is local memory?

In CUDA, "local memory" is a misleading name. It is **not** fast on-chip storage — it is a per-thread section of **global DRAM**, addressed with per-thread offsets. Local memory is used as an overflow area when the compiler cannot fit a thread's variables into registers.

P100 global memory bandwidth: 732 GB/s. Latency: ~400–800 cycles per access. Accessing local memory is as expensive as accessing any other DRAM region, and it does not benefit from shared memory caching.

## Where does the spill happen?

In `order_crossover_device`:

```cpp
int used[MAX_CITIES];    // 128 × 4 bytes = 512 bytes
// ...
used[parent_a[i]] = 1;  // index: parent_a[i] — a runtime value
// ...
if (used[gene]) continue; // index: gene — a runtime value
```

The CUDA compiler can keep an array in registers **only if every index is a compile-time constant or a loop variable that becomes a constant after unrolling**. Here, `parent_a[i]` and `gene` are values read from memory — they differ per thread and per iteration. The compiler cannot statically assign a distinct register per element. It must place `used[]` in local memory (DRAM).

The `ptxas -v` output reveals this:

```
ptxas info: Function properties for ga_island_kernel
  lmem = 512    ← 128 ints × 4 bytes: the used[] array
  smem = 33280
  registers = 40
```

## How much DRAM traffic does this generate?

Per crossover call, every element of `used[]` is written once (during the segment copy and fill phases) and read once (during the `if (used[gene]) continue` test). In the worst case:

```
Writes to used[]: 128 × 4 = 512 bytes per child
Reads  from used[]: 128 × 4 = 512 bytes per child
Total: 1,024 bytes per child per generation
```

Across a full run:

```
30 non-elite threads × 128 islands × 1000 generations × 1,024 bytes
= 3,932,160,000 bytes ≈ 3.93 GB
```

This traffic is invisible in standard bandwidth metrics (it doesn't count toward `achieved_bandwidth`), but it directly competes with all other DRAM traffic for L2 bandwidth, worsening latency across the kernel.

## The fix: represent `used[]` as a 128-bit register bitmask

Instead of a 128-element integer array, use **four 32-bit unsigned integers** (128 bits total). Each bit position corresponds to one city:

```
used0 covers cities  0..31
used1 covers cities 32..63
used2 covers cities 64..95
used3 covers cities 96..127
```

To mark city `c` as used, OR in the corresponding bit:

```cpp
uint32_t _bit = 1u << (c & 31);  // bit position within the 32-bit word
if      (c <  32) used0 |= _bit;
else if (c <  64) used1 |= _bit;
else if (c <  96) used2 |= _bit;
else              used3 |= _bit;
```

To test if city `c` is marked:

```cpp
uint32_t _bit = 1u << (c & 31);
bool is_set = (c <  32) ? (used0 & _bit) :
              (c <  64) ? (used1 & _bit) :
              (c <  96) ? (used2 & _bit) :
                          (used3 & _bit);
```

The compiler sees four scalar `uint32_t` variables with no dynamic indexing. It allocates them to four registers. Every mark and test is **pure arithmetic** — one shift and one bitwise operation — with zero memory traffic.

## Why do the if-else chains not hurt performance?

You might wonder whether the branching in `MARK(c)` and `ISSET(c)` introduces divergence. It does not, for two reasons:

1. **The branches are over city index ranges**, not over thread IDs. Within a single thread's crossover call, the same city ranges may be hit differently, but this is intra-thread control flow — not warp divergence (which only occurs when threads in the same warp take different paths at the same instruction).

2. **The compiler often converts these to predicated instructions** (SELP, SETP in PTX). A predicated instruction always executes but only writes its result if the predicate is true — no branch, no pipeline flush.

## After the fix

`ptxas` output changes to:

```
lmem = 0      ← no spill
smem = 33280  ← unchanged (B1 padding still in place)
registers = 44 ← slight increase: 4 extra uint32 registers, but within P100's budget (255 max)
```

And at runtime:

```
local_load_transactions  = 0   ← was: proportional to n × islands × gens
local_store_transactions = 0   ← was: proportional to n × islands × gens
```

## Limitations of this approach

The 4-word bitmask is hardcoded for `n ≤ 128`. If you extend the code to support larger cities:

- `n ≤ 64`: two `uint32_t` words suffice.
- `n ≤ 256`: eight words (256 bits), still register-resident.
- `n > 256`: you'd need a shared memory bitmask, which reintroduces memory traffic but avoids DRAM.

For the current `MAX_CITIES = 128` constraint, the 4-word approach is zero-cost.

## Interaction with B1

B1 (stride padding) and B2 (bitmask) address entirely different memory systems:

| Fix | Memory affected | Mechanism |
|-----|----------------|-----------|
| B1  | Shared memory   | Eliminated 32-way bank conflicts via stride = n+1 |
| B2  | Local memory (DRAM) | Eliminated per-thread used[] spill via register bitmask |

Both fixes are independent and cumulative. B2 is applied on top of B1 in `CUDA-GA-B2-bitmask.cu`.
