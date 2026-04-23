# GPU-Accelerated Genetic Algorithm for the Traveling Salesman Problem
## Living Document — ECE 569 Final Report
### University of Arizona · P100 Tesla · Spring 2026

> **How to use this document.**
> Sections marked **★ WRITE NOW** have enough material to draft today.
> Sections marked **⚙ NEEDS DATA** require profiler output or benchmark results before writing.
> Sections marked **🔬 EXPERIMENT PENDING** are experiments you need to run.
> Sections marked **✅ DONE** are complete and ready to polish.
> Update this file after every profiling session or benchmark run.
> The goal is to demonstrate CUDA programming fluency AND GPU architectural design thinking.

---

## STATUS TRACKER

| Section | Status | Blocking dependency |
|---|---|---|
| Abstract | ⚙ NEEDS DATA | Final benchmark numbers |
| 1. Introduction | ★ WRITE NOW | Nothing |
| 2. Background | ★ WRITE NOW | Nothing |
| 3. Implementation Arc | ★ WRITE NOW | Code already exists |
| 4. Bottleneck Analysis | ★ WRITE NOW (math only) | Profile confirms/refutes |
| 5. Profiling Protocol | ★ WRITE NOW | Tool audit script |
| 6. Optimization Story | ⚙ NEEDS DATA | Experiments E1–E5 |
| 7. 2-opt Extension | ★ WRITE NOW (design) | Results pending |
| 8. Experimental Results | 🔬 EXPERIMENT PENDING | All experiments |
| 9. Discussion | ⚙ NEEDS DATA | Section 8 complete |
| 10. Conclusion | ⚙ NEEDS DATA | Everything above |
| References | ★ WRITE NOW | Literature collected |

---

## PRE-WRITING CHECKLIST

Before touching HPC, complete these:

- [ ] Collect TSPLIB benchmark instances: `berlin52.tsp`, `kroA100.tsp`, `ch130.tsp`, `d198.tsp`
- [ ] Confirm P100 node availability and partition name (`gpu_standard` or equivalent)
- [ ] Run `hpc_cuda_tool_audit.sh` and record which tools are available (nvprof / ncu / nsys)
- [ ] Tag current baseline as `v0-baseline` in your repo
- [ ] Write SLURM template (see Section 5.1)
- [ ] Add `cudaEvent_t` timing wrappers to all binaries before any profiling

---

---

# PART I — REPORT BODY

---

## Section 1 — Introduction ★ WRITE NOW

> **What to write:** 1–2 pages. No data needed. Motivate the problem, state your contributions.

### 1.1 Motivation

**Draft prompt (replace with your own voice):**
The Traveling Salesman Problem (TSP) is an NP-hard combinatorial optimization problem with applications in logistics, circuit design, and genome sequencing. Exact methods scale as O(N!); heuristic search — particularly Genetic Algorithms — offers practical approximate solutions at the cost of computational intensity. General-Purpose GPU computing (GPGPU) provides massive data parallelism suited to the population-level operations in a GA: fitness evaluation, selection, crossover, and mutation can all be parallelized across individuals and islands.

**Key claims to make in the introduction (each must be backed by Section 8 when done):**
- [ ] Our GPU-resident island GA achieves [X×] speedup over the CPU-only GA
- [ ] Adding parallel 2-opt elite refinement improves tour quality by [Y%] with [Z ms] overhead
- [ ] Applying three low-risk CUDA optimizations (stride padding, bitmask, warp reduction) yields [W×] kernel speedup with no change to algorithm semantics

### 1.2 Contributions

List exactly (expand as experiments complete):
1. Three-stage implementation arc demonstrating progressive GPU exploitation
2. Quantitative bottleneck analysis grounded in P100 hardware specifications
3. Shared-memory bank conflict elimination via stride padding
4. Local-memory spill elimination via 4-register bitmask for OX crossover
5. Serialized sort replacement with warp-shuffle min-reduction
6. Integration of parallel 2-opt local search on elite individuals
7. Experimental evaluation across [N problem sizes, M thread configurations]

### 1.3 Scope and Hardware

All experiments run on the NVIDIA Tesla P100 (Pascal, sm_60):

| Spec | Value |
|---|---|
| CUDA Cores | 2,560 |
| SMs | 56 |
| Global Memory | 16 GB HBM2 |
| Memory Bandwidth | 732 GB/s |
| L2 Cache | 4 MB |
| Shared Memory / SM | 64 KB |
| Shared Memory Banks | 32 × 4 B |
| Constant Cache / SM | 8 KB |
| Max Warps / SM | 64 |
| Max Threads / Block | 1,024 |
| Warp Size | 32 |
| PCIe (host–device) | ~12 GB/s (PCIe 3.0 ×16) |

---

## Section 2 — Background ★ WRITE NOW

> **What to write:** 2–3 pages. Literature review, algorithm background, GPU architecture recap.

### 2.1 Traveling Salesman Problem

Define formally. Cite standard NP-hardness reference. Note that exact solvers (Concorde, branch-and-bound) are impractical beyond ~1000 cities in real time.

**Placeholder citation targets:**
- Applegate et al., *The Traveling Salesman Problem* (2006)
- Lin & Kernighan (1973) — LK heuristic as the classical local search baseline

### 2.2 Genetic Algorithms for TSP

Describe the GA loop:
1. Initialize population of tours
2. Evaluate fitness (tour length)
3. Select parents (tournament selection)
4. Apply crossover (Order Crossover, OX)
5. Apply mutation (swap mutation)
6. Elitism: carry top-k individuals forward
7. Repeat for G generations

**Note island model:** Each sub-population (island) evolves independently. Parallelism across islands = parallelism across CUDA blocks.

**Key citations:**
- Holland (1975) — original GA
- Goldberg (1989) — GA for combinatorial optimization
- Whitley et al. (1989) — island model GA

### 2.3 GPU Computing and CUDA

Brief recap of SIMT execution model, thread hierarchy (grid → block → warp → thread), memory hierarchy (global, shared, constant, register, local). Emphasize:
- **Warp divergence:** divergent branches serialize execution across the warp
- **Memory coalescing:** threads in a warp should access consecutive addresses
- **Occupancy:** ratio of active warps to maximum warps/SM; determines latency hiding capacity

**Key citations:**
- NVIDIA CUDA Programming Guide (2023)
- Kirk & Hwu, *Programming Massively Parallel Processors* (2022)

### 2.4 Related Work on GPU-Accelerated TSP

**Must cite and summarize:**

| Work | Method | Hardware | Key result |
|---|---|---|---|
| Ermiş & Çatay (2017) | Parallel 2-opt | GPU | 14× speedup on 4000-city instance |
| Ding et al. (2019) | GPU-GA island model | — | Block size sweep methodology |
| Younis & Boland (2016) | CUDA TSP heuristics | — | Constant memory for dist matrix |
| [add 2–3 more] | | | |

> **Write note:** Explicitly distinguish your work from each of these in the intro. State what you borrowed (Ermiş & Çatay's block-sweep methodology) and what is novel (bitmask fix, warp-shuffle elite selection, integrated in-kernel 2-opt).

---

## Section 3 — Implementation Arc ★ WRITE NOW

> **What to write:** 3–4 pages. Describe the three implementations. Code snippets included. No profiling data needed — this is design documentation.

### 3.1 Overview

| Version | Evolution | Fitness | H↔D Transfers / gen | Thread model |
|---|---|---|---|---|
| `GPU-Naive.cu` | None (correctness test) | GPU | 2 | 1 thread → 1 tour |
| `CUDA-GA.cu` | CPU | GPU | 2 every generation | 1 thread → 1 tour |
| `CUDA-GA-GPU-Pop.cu` | GPU (island model) | GPU | 1 (outputs only) | 1 thread → 1 individual |

### 3.2 GPU-Naive: Correctness Baseline

**What to describe:**
- Purpose: plumbing test, not a GA
- Kernel: `eval_tour_lengths_kernel` — one thread per tour, serial loop over N cities
- Thread mapping: `tid = blockIdx.x * blockDim.x + threadIdx.x`
- Data layout: AoS — tours stored row-major, `tours[tid * N + k]`
- Output: GPU lengths cross-checked against CPU lengths

**Include snippet:**
```cpp
__global__ void eval_tour_lengths_kernel(const int* tours,
                                          const int* dist,
                                          int* lengths,
                                          int num_tours, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tours) return;
    int base = tid * N;
    int sum = 0;
    for (int k = 0; k < N; ++k) {
        int a = tours[base + k];
        int b = tours[base + ((k + 1) % N)];
        sum += dist[a * N + b];
    }
    lengths[tid] = sum;
}
```

**Identify the AoS access problem here** (foreshadow Section 4, B1).

### 3.3 CUDA-GA: Hybrid CPU–GPU GA

**What to describe:**
- CPU owns the full GA loop (selection, crossover, mutation)
- GPU evaluates fitness each generation via `eval_tour_lengths_kernel`
- `cudaMemcpy` population to GPU every generation; copy lengths back
- PCIe bottleneck: 512 × 128 × 4 = 262,144 bytes uploaded per generation

**Include the generation loop structure** showing the CPU/GPU alternation.

**Calculate the PCIe cost:**
```
Population upload: 512 tours × 128 cities × 4 bytes = 262,144 bytes/gen
Lengths download:  512 × 4 bytes = 2,048 bytes/gen
Total: ≈ 264 KB/gen × 1000 gens = 264 MB
At 12 GB/s PCIe: 264 MB / 12 GB/s ≈ 22 ms of pure transfer
```

### 3.4 CUDA-GA-GPU-Pop: GPU-Resident Island Model

**What to describe:**
- Architecture: 1 block = 1 island = independent GA population
- Thread mapping: `threadIdx.x` = individual index within island
- Shared memory layout: `pop_a`, `pop_b`, `lengths`, `order`
- Constant memory: `c_dist[MAX_CITIES × MAX_CITIES]`
- Evolution entirely on GPU — `__syncthreads()` gates each phase
- Host receives only island-best results after all generations complete

**Include the shared memory partitioning snippet and kernel launch config.**

**Note the constants:** `MAX_CITIES = 128`, `BLOCK_POP_SIZE = 32`, `TOURNAMENT_SIZE = 3`

---

## Section 4 — Bottleneck Analysis ★ WRITE NOW (math) / ⚙ NEEDS DATA (profiler confirmation)

> **What to write now:** All mathematical derivations. Mark profiler columns as [PENDING] until experiments run.

### 4.1 Priority Matrix

| # | Bottleneck | Code location | Perf impact ★ | GPU-friendly ★ | Stage |
|---|---|---|---|---|---|
| B1 | 32-way shared memory bank conflict | `pop_a[tid * n + k]` | ★★★★★ | ★★★★★ | 1 |
| B2 | `used[MAX_CITIES]` local memory spill | `order_crossover_device` | ★★★★★ | ★★★★☆ | 1 |
| B3 | Thread-0 O(P²) serialized sort | `if (tid == 0)` block | ★★★★☆ | ★★★★★ | 1 |
| B4 | Constant memory scatter penalty | `c_dist`, `tour_length_const` | ★★★☆☆ | ★★★☆☆ | 1 (benchmark) |
| B5 | Low occupancy (1.56%) | 32-thread block, 33 KB smem | ★★★☆☆ | ★★☆☆☆ | 2 |
| B6 | Serial per-thread tour evaluation | `tour_length_const` loop | ★★★★☆ | ★★☆☆☆ | 2 |
| B7 | OX branch divergence | `if (used[gene]) continue` | ★★☆☆☆ | ★★★☆☆ | 2 |
| B8 | No local search on elites | — | ★★★★★ (quality) | ★★☆☆☆ | 3 |

### 4.2 B1 — Shared Memory Bank Conflict (★★★★★)

**Math derivation:**

Bank assignment: `bank(e) = e % 32`

For `pop_a[t * n + k]` with `n = 128`:
```
bank(t, k) = (t × 128 + k) % 32 = k % 32     [since 128 % 32 = 0]
```
All 32 threads in the warp map to bank `k % 32`. **32-way conflict. Hardware serializes into 32 transactions.**

Fix with stride = n + 1 = 129:
```
bank(t, k) = (t × 129 + k) % 32 = (t + k) % 32
```
Thread 0 → bank k, Thread 1 → bank k+1, ..., Thread 31 → bank k+31 (mod 32). **Zero conflicts.**

**Shared memory budget:**
```
Before: (2 × 32 × 128 + 2 × 32) × 4 = 33,024 bytes
After:  (2 × 32 × 129 + 2 × 32) × 4 = 33,280 bytes  (+256 bytes, negligible)
```

**Profiler target:** `shared_load_transactions_per_request`: 32 → 1  
**Profiler result:** [PENDING — fill after E3]

### 4.3 B2 — Local Memory Spill (★★★★★)

**Why it spills:**
`int used[MAX_CITIES]` in `order_crossover_device` is indexed by runtime values (`gene`, `parent_a[i]`). The PTX compiler cannot allocate runtime-indexed arrays to registers → spills to local memory (physically DRAM).

**Traffic calculation:**
```
Per crossover: 128 reads + 128 writes × 4 bytes = 1,024 bytes
Per full run:  30 threads × 128 islands × 1000 gens × 1,024 = 3.93 GB
```

**Fix:** Replace `int used[MAX_CITIES]` with a 128-bit bitmask in 4 registers:
```cpp
uint32_t used0 = 0, used1 = 0, used2 = 0, used3 = 0;
```

**Compile-time check:** `lmem = 512` → `lmem = 0` in `ptxas -v` output  
**Profiler result:** [PENDING — fill after E3]

### 4.4 B3 — Thread-0 Serialized Sort (★★★★☆)

**Math:**
Selection sort for P = 32: `P(P-1)/2 = 496 comparisons` per generation.
```
Total: 496 × 128 islands × 1000 gens = 63,488,000 serial comparisons
31 threads idle during each sort.
```

**Fix:** Warp-shuffle min-reduction (5 shuffle steps, 0 `__syncthreads()`, all 32 threads used).
For elite_count = 2: two passes × 5 steps = **10 steps total** vs 496.

**Profiler target:** `smsp__warp_issue_stalled_barrier_per_warp_active.pct` decreases  
**Profiler result:** [PENDING — fill after E3]

### 4.5 B4 — Constant Memory Scatter (★★★☆☆)

**Analysis:**
Constant cache: 8 KB/SM, optimized for broadcast (all threads read same address).
For `c_dist[a * n + b]` where `a`, `b` differ per thread: **serializes 32 fetches**.

**L2 argument:**
Matrix: `128 × 128 × 4 = 64 KB`. P100 L2 = 4 MB.
Matrix is 1.6% of L2. After warm-up, all accesses are L2 hits.
L2 serves scatter reads in parallel (multiple in-flight requests).

**Decision:** benchmark both. Do not assume.
**A/B result:** [PENDING — fill after E4]

### 4.6 B5 — Occupancy (★★★☆☆)

```
Shared memory / block: 33,280 bytes (after padding)
Max blocks / SM (smem):  floor(65,536 / 33,280) = 1
Warps / block: 32 / 32 = 1
Active warps / SM: 1
P100 max warps / SM: 64
Occupancy: 1 / 64 = 1.56%
```

1 active warp → no backup warps to hide L2 miss latency (~80 cycles).

**`cudaOccupancyMaxActiveBlocksPerMultiprocessor` result:** [PENDING — fill after E3]

### 4.7 B7 — OX Branch Divergence (★★☆☆☆)

`if (used[gene]) continue` creates data-dependent divergence: threads within a warp take different paths based on their tour's gene membership set. Expected branch efficiency: 50–70%.

**Profiler target:** `branch_efficiency`  
**Profiler result:** [PENDING — fill after E3]

---

## Section 5 — Profiling Protocol ★ WRITE NOW

> **Write this section in full now. It is methodology — no data required. It demonstrates HPC maturity.**

### 5.1 HPC Etiquette: Allocation Strategy

**Golden rule:** Never profile on a login node.

```bash
# Interactive allocation
salloc -p gpu_standard --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00
srun --pty bash
module load cuda/12.4        # adjust to available version
nvidia-smi                   # confirm P100 detected

# Batch submission template
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_standard
#SBATCH --time=01:00:00
#SBATCH --output=profile_%j.out
module load cuda/12.4
```

### 5.2 Tool Discovery Audit

Run before every profiling session:
```bash
#!/bin/bash
# hpc_cuda_tool_audit.sh
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "=== Compiler ==="
nvcc --version 2>&1 | head -3

echo "=== Tools ==="
for tool in nvprof ncu nsys compute-sanitizer cuda-gdb; do
    if command -v $tool &> /dev/null; then
        echo "$tool: FOUND at $(which $tool)"
    else
        echo "$tool: NOT FOUND"
    fi
done

echo "=== Profiling permission (nsys dry run) ==="
nsys status -e 2>&1 | head -5
```

**Environment matrix (fill after running):**

| Tool | Found? | Executes? | Notes |
|---|---|---|---|
| nvprof | [PENDING] | [PENDING] | Legacy; available on P100 |
| ncu | [PENDING] | [PENDING] | Modern; preferred |
| nsys | [PENDING] | [PENDING] | System-level timeline |
| compute-sanitizer | [PENDING] | [PENDING] | Correctness checking |
| cuda-gdb | [PENDING] | [PENDING] | Device debugging |

### 5.3 Two-Build Strategy

```bash
# Debug build — for compute-sanitizer, correctness
nvcc -O0 -g -G -lineinfo -arch=sm_60 \
     -o app_debug CUDA-GA-GPU-Pop.cu tsplib_parser.cpp

# Profile build — for ncu, nsys, benchmarking
# CRITICAL: remove -G; it disables hardware optimizations
nvcc -O3 -lineinfo -arch=sm_60 \
     -o app_prof CUDA-GA-GPU-Pop.cu tsplib_parser.cpp

# Static resource check (free — run this first always)
nvcc -O3 -lineinfo -arch=sm_60 --ptxas-options=-v \
     CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 \
     | grep -E "ga_island_kernel|lmem|smem|registers"
```

### 5.4 Phase 1: Correctness Before Performance

```bash
# Memory safety
compute-sanitizer --tool memcheck --log-file memcheck.log ./app_debug <args>

# Shared memory races (critical for island model with __syncthreads)
compute-sanitizer --tool racecheck --log-file racecheck.log ./app_debug <args>

# Barrier synchronization errors
compute-sanitizer --tool synccheck --log-file synccheck.log ./app_debug <args>

# Goal: all logs report "no errors detected" before proceeding to profiling
```

### 5.5 Phase 2: System-Level Trace (nsys)

```bash
nsys profile \
     --trace=cuda,nvtx,osrt \
     --stats=true \
     -o nsys_baseline \
     ./app_prof benchmark.tsp 128 1000

# Open with: nsys-ui nsys_baseline.nsys-rep
# Or summarize CLI: nsys stats nsys_baseline.nsys-rep
```

**What to look for in nsys output:**
- For `CUDA-GA.cu`: alternating memcpy + kernel bars → memcpy fraction of total time
- For `CUDA-GA-GPU-Pop.cu`: single kernel bar → confirms GPU-resident design
- Kernel name, duration, grid size in the CUDA API summary

### 5.6 Phase 3: Kernel-Level Profile (ncu)

```bash
# Quick profile
ncu --kernel-name ga_island_kernel ./app_prof benchmark.tsp 128 100

# Full metrics, save to report
ncu --set full \
    --kernel-name ga_island_kernel \
    -o ncu_report \
    ./app_prof benchmark.tsp 128 100

# Target-specific metrics per bottleneck:

# B1: bank conflicts
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --kernel-name ga_island_kernel ./app_prof benchmark.tsp 128 100

# B5: occupancy
ncu --section Occupancy --kernel-name ga_island_kernel ./app_prof benchmark.tsp 128 100

# B7: branch divergence
ncu --metrics \
  smsp__sass_branch_targets_threads_divergent.sum,\
  smsp__sass_branch_targets_threads_uniform.sum \
  --kernel-name ga_island_kernel ./app_prof benchmark.tsp 128 100
```

### 5.7 Legacy nvprof (if ncu unavailable)

```bash
# Bank conflicts
nvprof --metrics \
  shared_load_transactions_per_request,\
  shared_store_transactions_per_request \
  ./app_prof benchmark.tsp 128 100

# Local memory
nvprof --metrics local_load_transactions,local_store_transactions \
  ./app_prof benchmark.tsp 128 1000

# Occupancy
nvprof --metrics achieved_occupancy,eligible_warps_per_cycle \
  ./app_prof benchmark.tsp 128 1000

# Branch divergence
nvprof --metrics branch_efficiency,warp_nonpred_exec_efficiency \
  ./app_prof benchmark.tsp 128 1000
```

### 5.8 Timing Wrapper (add to all builds)

```cpp
// Add to run_gpu_population_ga() before kernel launch:
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
printf("kernel_time_ms=%.3f best_length=%d\n", ms, best.length);
```

---

## Section 6 — Optimization Story ⚙ NEEDS DATA

> **What to write now:** Subsection headers, code snippets, math. Leave result rows blank until experiments complete.

### 6.1 Optimization Versions

| Version | Optimization applied | Code change |
|---|---|---|
| V0 | Baseline (no changes) | — |
| V1 | B1: Stride padding | `stride = n + 1` |
| V2 | B2: Bitmask for `used[]` | 4 × `uint32_t` registers |
| V3 | B3: Warp-shuffle elite reduction | Replace `tid==0` sort |
| V4 | B4: Global `__restrict__` dist | Remove `__constant__` |
| V5 | B8: In-kernel 2-opt (K=10) | 2-opt pass every 10 gens |

Each version: describe change, show before/after code, state metric prediction, show measurement.

Results table (fill after Experiment E5):

| Version | lmem | smem | Bank tx/req | Occupancy | Kernel time (ms) | Best tour |
|---|---|---|---|---|---|---|
| V0 | 512 B | 33,024 B | ~32 | 1.56% | [PENDING] | [PENDING] |
| V1 | 512 B | 33,280 B | ~1 | 1.56% | [PENDING] | [PENDING] |
| V2 | 0 | 33,280 B | ~1 | 1.56% | [PENDING] | [PENDING] |
| V3 | 0 | 33,280 B | ~1 | 1.56% | [PENDING] | [PENDING] |
| V4 | 0 | 33,280 B | ~1 | 1.56% | [PENDING] | [PENDING] |
| V5 | 0 | 33,280 B | ~1 | 1.56% | [PENDING] | [PENDING] |

---

## Section 7 — Parallel 2-opt Elite Refinement ★ WRITE NOW (design) / ⚙ NEEDS DATA (results)

> **Design section is writeable now. Insert results from E6 when available.**

### 7.1 Motivation

GA convergence is stochastic. 2-opt is a deterministic local search that provably improves any tour until no improving edge swap exists (local optimum). Applying it to elite individuals after selection reinforces the best tours found by the GA, accelerating convergence to the global optimum.

### 7.2 Mathematical Foundation

For tour T of length N, a 2-opt move reverses the segment between cities i and j:
```
delta(i, j) = dist(T[i], T[j]) + dist(T[i+1], T[j+1])
            - dist(T[i], T[i+1]) - dist(T[j], T[j+1])
```
Accept move if `delta < 0`.

Total candidate pairs for N = 128: `N(N-1)/2 = 8,128`

### 7.3 Parallelization Strategy

**Thread mapping:** Each of the 32 block threads scans a subset of the 8,128 pairs:
```
254 pairs per thread = 8,128 / 32
```

Each thread tracks its local best move. Warp-shuffle reduction finds the global best move across all threads in 5 steps. Lane 0 applies the winning reversal.

**Cost per 2-opt pass inside the island kernel:**
```
Pairs: 8,128 / 32 threads = 254 per thread
Ops per pair: 4 dist lookups + 3 arithmetic = 7 ops
Total: 254 × 7 = 1,778 ops/thread
All 128 islands run concurrently → wall time ≈ 1 island's cost
```

### 7.4 Integration Design

```cpp
// Inside ga_island_kernel, after elite selection:
if (generation % TWOOPT_INTERVAL == 0 && TWOOPT_INTERVAL > 0) {
    int* elite_tour = current + order[0] * stride;
    int my_best_delta = 0, my_best_i = -1, my_best_j = -1;

    for (int pair = tid; pair < n*(n-1)/2; pair += BLOCK_POP_SIZE) {
        int pi = (int)floorf((-1.0f + sqrtf(1.0f + 8.0f*pair)) * 0.5f);
        int pj = pair - pi*(pi+1)/2 + pi + 1;
        int a = elite_tour[pi], b = elite_tour[(pi+1)%n];
        int c = elite_tour[pj], d = elite_tour[(pj+1)%n];
        int delta = c_dist[a*n+c] + c_dist[b*n+d]
                  - c_dist[a*n+b] - c_dist[c*n+d];
        if (delta < my_best_delta) {
            my_best_delta = delta; my_best_i = pi; my_best_j = pj;
        }
    }
    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_d = __shfl_xor_sync(0xffffffff, my_best_delta, mask);
        int o_i = __shfl_xor_sync(0xffffffff, my_best_i, mask);
        int o_j = __shfl_xor_sync(0xffffffff, my_best_j, mask);
        if (o_d < my_best_delta) { my_best_delta=o_d; my_best_i=o_i; my_best_j=o_j; }
    }
    if (tid == 0 && my_best_delta < 0) {
        int lo = my_best_i + 1, hi = my_best_j;
        while (lo < hi) {
            int tmp = elite_tour[lo]; elite_tour[lo] = elite_tour[hi];
            elite_tour[hi] = tmp; ++lo; --hi;
        }
    }
    __syncthreads();
}
```

### 7.5 Results

2-opt interval sweep results from Experiment E6 (fill after running):

| K (2-opt interval) | Kernel time (ms) | Best length | Quality gain vs V3 |
|---|---|---|---|
| ∞ (disabled) | [PENDING] | [PENDING] | baseline |
| 100 | [PENDING] | [PENDING] | [PENDING] |
| 50 | [PENDING] | [PENDING] | [PENDING] |
| 20 | [PENDING] | [PENDING] | [PENDING] |
| 10 | [PENDING] | [PENDING] | [PENDING] |
| 5 | [PENDING] | [PENDING] | [PENDING] |
| 1 | [PENDING] | [PENDING] | [PENDING] |

---

## Section 8 — Experimental Results 🔬 EXPERIMENT PENDING

> All tables below have structure. Fill with measured values after each experiment run.

### 8.1 Experiment E1: Three-Implementation Comparison (★ RUN FIRST)

**Purpose:** Quantify the architectural leap from hybrid CPU–GPU to GPU-resident island GA.  
**Instances:** `berlin52.tsp` (N=52), `kroA100.tsp` (N=100), `ch130.tsp` (N=128)  
**Config:** 512 individuals (for CUDA-GA), 128 islands × 32 (for GPU-Pop), 1000 gens, seed=42  
**Repetitions:** 10 per configuration

**Table E1 — Implementation Comparison**

| Implementation | Instance | N | Wall time (ms) | Kernel time (ms) | Best tour length | % vs optimal |
|---|---|---|---|---|---|---|
| CPU-only | berlin52 | 52 | [PENDING] | N/A | [PENDING] | [PENDING] |
| CUDA-GA (hybrid) | berlin52 | 52 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| GPU-Pop (island) | berlin52 | 52 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| CPU-only | ch130 | 128 | [PENDING] | N/A | [PENDING] | [PENDING] |
| CUDA-GA (hybrid) | ch130 | 128 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| GPU-Pop (island) | ch130 | 128 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

**Figure E1:** Bar chart — wall time by implementation and problem size.  
**Figure E2:** nsys timeline screenshots for CUDA-GA (showing memcpy bottleneck) vs GPU-Pop (clean kernel bar).

**Profiling commands for E1:**
```bash
# Timeline for CUDA-GA (show memcpy bottleneck)
nsys profile --trace=cuda,osrt --stats=true -o nsys_cudaga \
     ./CUDA-GA berlin52.tsp 512 1000 0.05 4 42

# Timeline for GPU-Pop (show clean kernel)
nsys profile --trace=cuda,osrt --stats=true -o nsys_gpupop \
     ./CUDA-GA-GPU-Pop berlin52.tsp 128 1000 0.05 2 42
```

---

### 8.2 Experiment E2: Problem Size Scaling 🔬

**Purpose:** How does GPU-Pop performance scale with N (number of cities)?  
**Config:** 128 islands, 1000 gens, seed=42  
**Run for each of:** N = 32, 48, 64, 96, 128  
**Use synthetic random instances if TSPLIB instances don't exist for small N**

**Table E2 — Scaling with Problem Size**

| N | Shared mem / block | Matrix size | L2 fit? | Kernel time (ms) | Best length | Gens/sec |
|---|---|---|---|---|---|---|
| 32 | ~8,448 B | 4 KB | Yes | [PENDING] | [PENDING] | [PENDING] |
| 48 | ~12,544 B | 9 KB | Yes | [PENDING] | [PENDING] | [PENDING] |
| 64 | ~16,768 B | 16 KB | Yes | [PENDING] | [PENDING] | [PENDING] |
| 96 | ~25,088 B | 36 KB | Yes | [PENDING] | [PENDING] | [PENDING] |
| 128 | ~33,280 B | 64 KB | Yes | [PENDING] | [PENDING] | [PENDING] |

**Figure E2:** Line chart — kernel time vs N.

---

### 8.3 Experiment E3: Island Count Sweep 🔬

**Purpose:** Find the island count that saturates the P100 (56 SMs).  
**Config:** N=128, 1000 gens, seed=42  
**Vary:** islands = 14, 28, 56, 112, 224, 512

**Table E3 — Island Count vs Performance**

| Islands | Blocks launched | SMs used | Time-sharing? | Kernel time (ms) | Best length | Gens/sec |
|---|---|---|---|---|---|---|
| 14 | 14 | 14 | No (under-saturated) | [PENDING] | [PENDING] | [PENDING] |
| 28 | 28 | 28 | No | [PENDING] | [PENDING] | [PENDING] |
| 56 | 56 | 56 | No (full saturation) | [PENDING] | [PENDING] | [PENDING] |
| 112 | 112 | 56 | Yes (2/SM) | [PENDING] | [PENDING] | [PENDING] |
| 224 | 224 | 56 | Yes (4/SM) | [PENDING] | [PENDING] | [PENDING] |
| 512 | 512 | 56 | Yes (9/SM) | [PENDING] | [PENDING] | [PENDING] |

**Hypothesis:** Kernel time plateaus at 56 islands (1/SM). Solution quality continues improving beyond 56 due to search diversity.  
**Figure E3:** Dual-axis — kernel time (line) and best tour length (bars) vs island count.

---

### 8.4 Experiment E4: Instance Type Comparison — Clustered vs Random 🔬

**Purpose:** Does tour structure (clustered cities vs random) affect convergence speed or GPU utilization?  
**Config:** 128 islands, 1000 gens, seed=42, N=128

**Instance types:**
- **Random uniform:** cities distributed uniformly in [0, 1000]² — generate synthetic
- **Clustered:** 8 cluster centers, each with 16 cities in radius 50 — generate synthetic
- **TSPLIB real-world:** `ch130.tsp` (130 cities, structured)

**Hypothesis:** Clustered tours converge faster (GA finds good local structure quickly). Random tours have higher branch divergence in crossover. TSPLIB instances benchmark against known optima.

**Table E4 — Instance Type Comparison**

| Instance type | Example | Best length | Generations to convergence | Kernel time (ms) | Branch efficiency |
|---|---|---|---|---|---|
| Random uniform | synthetic_128_r | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Clustered | synthetic_128_c | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| TSPLIB real-world | ch130.tsp | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

**Figure E4:** Convergence curves — best tour length vs generation number for each instance type.

---

### 8.5 Experiment E5: Optimization Version Story 🔬 ★ CORE CONTRIBUTION

**Purpose:** Show cumulative speedup from each optimization. This is the central narrative of the paper.  
**Config:** 128 islands, N=128, 1000 gens, 10 repetitions, seed=42  
**Build one binary per version (V0 through V5)**

**Before running E5:** Collect static profile for V0:
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O3 \
     -o V0 CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep -E "lmem|smem|registers"

nvprof --metrics shared_load_transactions_per_request,\
local_load_transactions,achieved_occupancy,branch_efficiency \
./V0 ch130.tsp 128 100 0.05 2 42
```

**Table E5 — Optimization Version Comparison**

| Version | Fix applied | lmem | smem Bank tx/req | Occ. | Kernel (ms) mean ± σ | Best tour | Speedup vs V0 |
|---|---|---|---|---|---|---|---|
| V0 | Baseline | 512 B | ~32 | 1.56% | [P] ± [P] | [P] | 1.0× |
| V1 | +stride pad | 512 B | ~1 | 1.56% | [P] ± [P] | [P] | [P]× |
| V2 | +bitmask | 0 | ~1 | 1.56% | [P] ± [P] | [P] | [P]× |
| V3 | +warp reduce | 0 | ~1 | 1.56% | [P] ± [P] | [P] | [P]× |
| V4 | +global dist | 0 | ~1 | 1.56% | [P] ± [P] | [P] | [P]× |
| V5 | +2-opt K=10 | 0 | ~1 | 1.56% | [P] ± [P] | [P] | [P]× |

**Figure E5:** Waterfall / stacked bar — cumulative speedup per optimization.  
**Figure E6:** Profiler screenshots — before/after `shared_load_transactions_per_request`.

---

### 8.6 Experiment E6: 2-opt Interval Sweep 🔬

**Purpose:** Find optimal 2-opt interval K balancing solution quality and runtime overhead.  
**Config:** V3 as base, 128 islands, N=128, 1000 gens, 10 repetitions, seed=42

**Table E6 — 2-opt Interval vs Quality/Speed**

| K | Kernel time (ms) | Best length | vs V3 (no 2-opt) | Overhead (ms) | Quality gain per ms |
|---|---|---|---|---|---|
| ∞ (off) | [P] | [P] | baseline | 0 | — |
| 100 | [P] | [P] | [P] | [P] | [P] |
| 50 | [P] | [P] | [P] | [P] | [P] |
| 20 | [P] | [P] | [P] | [P] | [P] |
| 10 | [P] | [P] | [P] | [P] | [P] |
| 5 | [P] | [P] | [P] | [P] | [P] |
| 1 | [P] | [P] | [P] | [P] | [P] |

**Figure E7:** Pareto curve — tour quality vs kernel time for different K values.

---

### 8.7 Experiment E7: Block Size and Occupancy Analysis 🔬

After Ermiş & Çatay (2017) methodology. Find the configuration that maximizes occupancy within the P100's shared memory constraint.

**Table E7 — Block/Thread Configuration**

| Config | Block size | smem/block | Max blocks/SM | Warps/SM | Occ. (%) | Kernel time (ms) |
|---|---|---|---|---|---|---|
| A: current | 32 | 33,280 B | 1 | 1 | 1.56% | [P] |
| B: BPOP=16 | 16 | ~16,640 B | 3 | 1.5 | 4.69% | [P] |
| C: 2 islands/block | 64 | ~66,048 B | 0 (no fit) | N/A | N/A | — |
| D: warp/indiv | 1024 | TBD | TBD | TBD | TBD | [P] |

**Figure E8:** Bar chart — occupancy and kernel time by configuration (inspired by Table 5 in Ermiş & Çatay).

---

## Section 9 — Discussion ⚙ NEEDS DATA

> Structure only. Write after Section 8 is complete.

### 9.1 What the numbers say

**Template:** For each major finding in Section 8, write one paragraph:
- What we predicted (from Section 4 math)
- What the profiler measured
- Whether they agreed
- What the disagreement reveals (if any)

### 9.2 Occupancy vs. solution quality trade-off

The 1.56% occupancy is a known structural limitation of the island model's shared memory footprint. The key insight: **the island model trades occupancy for data locality** — by keeping the entire population in shared memory, we eliminate global memory traffic for evolution. The question for discussion is whether this trade-off is net positive.

### 9.3 2-opt as a Pareto frontier shift

The 2-opt addition does not simply improve tour quality — it shifts the quality-vs-time Pareto frontier. Discuss which K value represents the best Pareto point for your use case.

### 9.4 Lessons for CUDA architecture design

Synthesize the three ECE 569 design lessons demonstrated:
1. **Memory space selection is not obvious:** constant memory hurt; L2-backed global memory helped
2. **Compile-time checks are profiling:** `ptxas -v` with `lmem > 0` caught a 3.93 GB hidden cost before running a single profile
3. **Serial code inside GPU kernels is expensive in proportion to idle threads:** the `tid==0` sort idled 31 threads; 10 warp-shuffle steps replaced 496 serial comparisons

---

## Section 10 — Conclusion ⚙ NEEDS DATA

> Write last. One page. Summarize contributions, state main numbers, suggest future work.

**Future work bullets (writeable now):**
- Island migration via double-buffered ring (no atomics needed)
- 3-opt or Lin–Kernighan move evaluation using CUDA dynamic parallelism
- Scaling beyond N = 128 with dynamic shared memory and L2-tiled dist matrix
- Multi-GPU execution across node interconnect (NVLink or MPI + CUDA)

---

---

# PART II — APPENDICES

---

## Appendix A — P100 Hardware Reference

| Resource | Value | Role in this work |
|---|---|---|
| CUDA cores | 2,560 | — |
| SMs | 56 | Limits useful island count |
| DRAM bandwidth | 732 GB/s | Limits scatter-heavy kernels |
| L2 cache | 4 MB | Fits full 64 KB dist matrix |
| Constant cache/SM | 8 KB | Bottleneck for scatter reads |
| Shared memory/SM | 64 KB | Limits blocks/SM to 1 for current kernel |
| Shared memory banks | 32 × 4 B | Source of B1 bank conflict |
| Max warps/SM | 64 | Occupancy denominator |
| Max threads/block | 1,024 | Ceiling for Pattern B |
| Warp size | 32 | Shuffle primitive scope |
| PCIe 3.0 × 16 | ~12 GB/s | Bottleneck for CUDA-GA.cu |

---

## Appendix B — TSPLIB Instances Used

| Instance | N | Type | Optimal | Source |
|---|---|---|---|---|
| berlin52 | 52 | EUC_2D | 7,542 | TSPLIB |
| kroA100 | 100 | EUC_2D | 21,282 | TSPLIB |
| ch130 | 130 | EUC_2D | 6,110 | TSPLIB |
| d198 | 198 | EUC_2D | 15,780 | TSPLIB |
| Synthetic random 128 | 128 | Random uniform | — | Generated |
| Synthetic clustered 128 | 128 | 8-cluster | — | Generated |

---

## Appendix C — Full Profiling Command Reference

```bash
# ── STATIC ──────────────────────────────────────────────
nvcc -arch=sm_60 --ptxas-options=-v -O3 \
     CUDA-GA-GPU-Pop.cu tsplib_parser.cpp 2>&1 | grep -E "lmem|smem|registers"

# ── BANK CONFLICTS ──────────────────────────────────────
nvprof --metrics shared_load_transactions_per_request,\
shared_store_transactions_per_request ./app benchmark.tsp 128 100

ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --kernel-name ga_island_kernel ./app benchmark.tsp 128 100

# ── LOCAL MEMORY ────────────────────────────────────────
nvprof --metrics local_load_transactions,local_store_transactions \
    ./app benchmark.tsp 128 1000

# ── OCCUPANCY ───────────────────────────────────────────
nvprof --metrics achieved_occupancy,eligible_warps_per_cycle \
    ./app benchmark.tsp 128 1000

ncu --section Occupancy --kernel-name ga_island_kernel \
    ./app benchmark.tsp 128 1000

# ── BRANCH DIVERGENCE ───────────────────────────────────
nvprof --metrics branch_efficiency,warp_nonpred_exec_efficiency \
    ./app benchmark.tsp 128 1000

# ── TIMELINE ────────────────────────────────────────────
nsys profile --trace=cuda,nvtx,osrt --stats=true -o report ./app benchmark.tsp 128 1000
```

---

## Appendix D — All Experiment Shell Scripts

**E5 — Version story benchmark:**
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --time=01:00:00
module load cuda/12.4
TSP=ch130.tsp; GENS=1000; ISLANDS=128; REPS=10

for version in V0 V1 V2 V3 V4 V5; do
    echo "=== $version ==="
    for i in $(seq 1 $REPS); do
        ./$version $TSP $ISLANDS $GENS 0.05 2 42 2>&1 | grep -E "kernel_time_ms|tour length"
    done
done
```

**E6 — 2-opt interval sweep:**
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --time=00:30:00
module load cuda/12.4
TSP=ch130.tsp; GENS=1000; ISLANDS=128; REPS=10

for K in 0 1 5 10 20 50 100; do
    # Build with -DTWOOPT_INTERVAL=$K
    nvcc -O3 -lineinfo -arch=sm_60 -DTWOOPT_INTERVAL=$K \
         -o twoopt_K$K CUDA-GA-GPU-Pop.cu tsplib_parser.cpp
    echo "=== K=$K ==="
    for i in $(seq 1 $REPS); do
        ./twoopt_K$K $TSP $ISLANDS $GENS 0.05 2 42 2>&1 | grep -E "kernel_time_ms|tour length"
    done
done
```

---

## References ★ WRITE NOW

> Collect these now. Fill in full citation details as you verify each source.

**CUDA and GPU Architecture**

[1] NVIDIA Corporation. *CUDA C++ Programming Guide*, Version 12.4, 2024.

[2] D. Kirk and W. Hwu, *Programming Massively Parallel Processors: A Hands-on Approach*, 4th ed. Morgan Kaufmann, 2022.

[3] V. Volkov, "Understanding Latency Hiding on GPUs," Ph.D. dissertation, UC Berkeley, 2016.

**TSP and Local Search**

[4] D. Applegate, R. Bixby, V. Chvátal, and W. Cook, *The Traveling Salesman Problem: A Computational Study*. Princeton University Press, 2006.

[5] S. Lin and B. W. Kernighan, "An Effective Heuristic Algorithm for the Traveling-Salesman Problem," *Operations Research*, vol. 21, no. 2, pp. 498–516, 1973.

**Genetic Algorithms**

[6] J. H. Holland, *Adaptation in Natural and Artificial Systems*. University of Michigan Press, 1975.

[7] D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley, 1989.

[8] D. Whitley, T. Starkweather, and D. Fuquay, "Scheduling Problems and Traveling Salesmen: The Genetic Edge Recombination Operator," in *Proc. Int. Conf. on Genetic Algorithms*, 1989.

**GPU-Accelerated TSP**

[9] G. Ermiş and B. Çatay, "Solving the Traveling Salesman Problem with the Help of a GPU," *Transportation Research Procedia*, vol. 22, pp. 409–418, 2017. ← **Primary reference for 2-opt methodology and block-size sweep**

[10] [ADD: Ding et al. GPU-GA island model — find full citation]

[11] [ADD: GPU TSP survey paper — find and cite]

**Profiling and Tools**

[12] NVIDIA Corporation. *Nsight Compute CLI Documentation*, 2024.

[13] NVIDIA Corporation. *Nsight Systems User Guide*, 2024.

[14] NVIDIA Corporation. *CUDA Best Practices Guide*, 2024.

---

## Change Log

| Date | Section updated | What changed | Author |
|---|---|---|---|
| [DATE] | All | Initial document created | Joel |
| | | | |

---

*Last updated: [DATE]. Next milestone: Run E1 (implementation comparison) and E5 (version story) before the next revision.*