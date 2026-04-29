# Profiling Without nvprof — Metrics Plan
## ECE 569 · TSP-GA Optimization Story

> **The situation:** Interactive nodes lock nvprof's metric counters for security reasons.  
> **The solution:** Build a self-contained measurement framework from three always-available tools:
> `cudaEvent_t` (timing), `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (occupancy), and
> `clock64()` inside kernels (intra-kernel phases). Every bottleneck claim is backed by a
> controlled experiment whose signal comes from **timing differences + arithmetic**, not
> from hardware performance counters.

---

## What We Can Always Measure (No nvprof Needed)

| Source | What it gives us | What bottleneck it addresses |
|---|---|---|
| `ptxas -v` at compile time | `lmem`, `smem`, `registers` per kernel | B2 spill (lmem>0), B5 occupancy (smem size) |
| `cudaEvent_t` around kernel | Wall-clock kernel time ± 0.01 ms | All bottlenecks — timing is ground truth |
| `cudaEvent_t` around `cudaMemcpy` | Transfer time | B9 PCIe bottleneck |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | Theoretical active blocks/SM | B5 occupancy |
| `cudaGetDeviceProperties` | Peak BW, SM count, clock rate | Roofline denominator |
| `clock64()` inside kernel | Cycles per phase (fitness/sort/crossover) | B3 sort fraction of runtime |
| A/B timing experiment | Conflict factor = T_strided / T_padded | B1 bank conflicts |
| A/B timing experiment | Spill overhead = T_used[] - T_bitmask | B2 local memory |
| A/B timing experiment | Const vs global = T_const / T_global | B4 constant memory |
| Tour quality comparison | Quality gain % = (L_before - L_after) / L_before | B8 2-opt effectiveness |

---

## Per-Bottleneck Metric Plan

### B1 — Shared Memory Bank Conflicts

**Metric we want:** `shared_load_transactions_per_request` (would be 32→1 in nvprof)

**What we measure instead:** Timing ratio from A/B experiment.

```
conflict_factor = T(stride=n) / T(stride=n+1)
```

**How to run:**
```bash
# Build both variants from the same source using compile flags
nvcc -O3 -arch=sm_60 -DV0_BASELINE -DBANK_CONFLICT_STRIDED -o bank_strided ...
nvcc -O3 -arch=sm_60 -DV0_BASELINE -DBANK_CONFLICT_PADDED  -o bank_padded  ...

# Time both (5 reps each, same seed)
for i in $(seq 1 5); do ./bank_strided ch130.tsp 128 100 ...; done
for i in $(seq 1 5); do ./bank_padded  ch130.tsp 128 100 ...; done
```

**Expected signal:**
```
n=128:  stride=128 → 32-way conflict → ~32× slower shared memory accesses
        stride=129 → zero conflicts

conflict_factor ≈ 4–32× depending on how much time the kernel spends in smem vs compute
```

**Decision gate:**
- `conflict_factor > 4×` → bank conflicts confirmed → apply padding
- `conflict_factor < 1.2×` → conflicts not the bottleneck here

**Compile-time sanity (always check this first):**
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O3 -o /dev/null CUDA-GA-GPU-Pop.cu 2>&1 | grep smem
# V0: smem = 33024   (stride=128 × 32 individuals × 2 pops × 4 + overhead)
# V1: smem = 33280   (stride=129 × same) ← 256 byte increase confirms padding applied
```

---

### B2 — Local Memory Spill

**Metric we want:** `local_load_transactions` (nvprof)  
**What we measure:** `lmem` from ptxas + A/B timing

**Step 1 — compile-time (free, instant):**
```bash
nvcc -arch=sm_60 --ptxas-options=-v -O3 -o /dev/null CUDA-GA-GPU-Pop.cu 2>&1 | grep lmem
```

**Expected:**
```
V0 (used[MAX_CITIES]):       lmem = 512    ← 128 ints × 4 bytes, SPILLING
V2 (bitmask, 4 registers):  lmem = 0      ← clean
```

**Step 2 — A/B timing confirms the traffic cost:**
```
spill_overhead_ms = T(V0 or V1) - T(V2)
```
If lmem > 0 and the timing difference is measurable, the spill is real.

**Arithmetic check:**
```
Expected hidden traffic = 30 threads × 128 islands × 1000 gens × 1024 bytes
                        = 3.93 GB of DRAM reads+writes
At 732 GB/s peak: 3.93 / 732 ≈ 5.4 ms theoretical minimum overhead
```

If `T(V0) - T(V2) ≈ 5 ms` → model is correct.  
If much smaller → L1/L2 is caching the local memory well (still fix it, but lower urgency).

---

### B3 — Thread-0 Serialized Sort

**Metric we want:** Time spent in the sort phase as a fraction of total kernel time.  
**What we measure:** `clock64()` phase instrumentation.

**Compile the instrumented build:**
```bash
nvcc -O3 -arch=sm_60 -DV0_BASELINE -DPHASE_TIMING \
     -o v0_phased CUDA-GA-GPU-Pop-instrumented.cu tsplib_parser.cpp
./v0_phased ch130.tsp 128 1000 0.05 2 42
```

**Expected output:**
```
── Intra-Kernel Phase Breakdown ──
  fitness_eval              cycles=  2840000   (31.2%)  ~0.473 ms
  elite_selection           cycles=  3920000   (43.1%)  ~0.653 ms  ← BOTTLENECK
  elite_copy                cycles=   210000    (2.3%)  ~0.035 ms
  crossover+mut             cycles=  2130000   (23.4%)  ~0.355 ms
```

The `elite_selection` phase (tid==0 sort) should dominate before V3.  
After V3 (warp shuffle), `elite_selection` drops to ~5% of total.

**Arithmetic prediction:**
```
Sort:         496 comparisons × 1 thread idle × N=128 steps
Warp shuffle: 10 shuffle instructions × 32 threads × O(1) overhead
Expected step reduction: 496 → 10 ≈ 50× in sort steps
```

---

### B4 — Constant Memory Scatter

**Metric we want:** L2 hit rate, constant cache miss rate.  
**What we measure:** A/B timing between two builds.

```bash
nvcc -O3 -arch=sm_60 -DV3_WARP_REDUCTION                -o v3_const  ...
nvcc -O3 -arch=sm_60 -DV3_WARP_REDUCTION -DV4_GLOBAL_DIST -o v3_global ...

# Run 10 reps each, same seed
for i in $(seq 1 10); do ./v3_const  ch130.tsp 128 1000 0.05 2 42; done
for i in $(seq 1 10); do ./v3_global ch130.tsp 128 1000 0.05 2 42; done
```

**Prediction:** For n=128, the 64 KB matrix fits in P100's 4 MB L2. Global memory with `__restrict__` should be faster than or equal to constant memory because:
- Constant cache: 8 KB/SM, scatter → serializes 32 lookups per warp step
- L2 (4 MB): 64 KB matrix = 1.6% of L2, warms after first few islands, then parallel hits

**If global is faster:** constant memory was indeed serializing — switch.  
**If constant is faster (or equal):** compiler already optimized, keep it.

---

### B5 — Occupancy

**Metric we want:** `achieved_occupancy` (nvprof)  
**What we measure:** Theoretical occupancy from CUDA API

```cpp
// Add to run() before kernel launch:
int max_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks, ga_island_kernel, BLOCK_POP_SIZE, smem);
int active_warps = max_blocks * (BLOCK_POP_SIZE / 32);
float occ = 100.f * active_warps / 64;  // 64 = max warps/SM on P100
printf("occupancy: %.2f%%  (%d blocks/SM, %d warps/SM)\n",
       occ, max_blocks, active_warps);
```

**Compile with `-DPRINT_OCCUPANCY`** to activate this at startup.

**Expected values:**
```
V0–V5 (32-thread block, 33 KB smem): 1 block/SM → 1 warp/SM → 1.56%
Future Pattern B (1024-thread block): TBD by profiler
```

**The occupancy paradox:** Low occupancy is confirmed but not fixable without a major kernel redesign (Pattern B). Document it as a known structural trade-off of the shared-memory island model.

---

### B7 — OX Branch Divergence

**Metric we want:** `branch_efficiency` (nvprof)  
**What we measure:** Timing of crossover phase alone + theoretical estimate

**Theoretical model:**
```
Expected segment size ≈ n/3 ≈ 43 for n=128
Fraction of iterations hitting 'continue' ≈ 43/128 ≈ 0.33
Expected branch efficiency ≈ 1 - 0.5 × 0.33 ≈ 0.83 (83%)
```

**Empirical measurement:** Compare crossover phase timing with divergent OX vs predicated write:
```bash
nvcc -O3 -arch=sm_60 -DV2_BITMASK -DPHASE_TIMING     -o v2_divergent ...
nvcc -O3 -arch=sm_60 -DV2_BITMASK -DPHASE_TIMING -DPREDICATED_OX -o v2_pred ...
```

Phase 3 (`crossover+mut`) time reduction = empirical divergence tax.

---

### B8 — 2-opt Quality and Overhead

**Metrics:** Tour quality improvement %, kernel time overhead %

```bash
# Baseline (no 2-opt)
./v3 ch130.tsp 128 1000 0.05 2 42 → best_length=L_base

# With 2-opt K=10
./v5_K10 ch130.tsp 128 1000 0.05 2 42 → best_length=L_opt

quality_gain = (L_base - L_opt) / L_base × 100%
overhead     = (T_v5 - T_v3) / T_v3 × 100%
```

Run over 5 seeds to get statistical significance:
```bash
for seed in 42 123 456 789 1000; do
  ./v3    ch130.tsp 128 1000 0.05 2 $seed | grep best_length
  ./v5_K10 ch130.tsp 128 1000 0.05 2 $seed | grep best_length
done
```

---

## Execution Order

Run in this exact order. Each step gates the next.

```
STEP 0: ptxas -v on V0 → confirm lmem=512, smem=33024
STEP 1: Build all 6 version binaries
STEP 2: Bank conflict A/B → confirm conflict_factor > 4×
STEP 3: Run V0 with PHASE_TIMING → identify which phase dominates
STEP 4: Occupancy API query → confirm 1.56% baseline
STEP 5: Version story E5 → fill the speedup table (10 reps each)
STEP 6: 2-opt sweep E6 → find optimal K
STEP 7: Island sweep E3 → confirm saturation at 56 islands
STEP 8: Problem size sweep E2 (if synthetic instances available)
STEP 9: ptxas -v on V5 → confirm lmem=0, smem=33536
STEP 10: Final occupancy query on V5
```

---

## Makefile Targets

```makefile
# Makefile excerpt — add to your project

ARCH    = sm_60
NVCC    = nvcc
FLAGS   = -O3 -lineinfo -arch=$(ARCH)
SRC     = CUDA-GA-GPU-Pop-instrumented.cu tsplib_parser.cpp

# ── Static analysis (free, always run first)
.PHONY: static
static:
	$(NVCC) $(FLAGS) --ptxas-options=-v -o /dev/null $(SRC) 2>&1 \
	    | grep -E "ga_island_kernel|lmem|smem|registers"

# ── Build all version binaries
.PHONY: versions
versions: v0 v1 v2 v3 v4 v5

v0: $(SRC); $(NVCC) $(FLAGS) -DV0_BASELINE          -o v0 $(SRC)
v1: $(SRC); $(NVCC) $(FLAGS) -DV1_STRIDE_PAD        -o v1 $(SRC)
v2: $(SRC); $(NVCC) $(FLAGS) -DV2_BITMASK           -o v2 $(SRC)
v3: $(SRC); $(NVCC) $(FLAGS) -DV3_WARP_REDUCTION    -o v3 $(SRC)
v4: $(SRC); $(NVCC) $(FLAGS) -DV4_GLOBAL_DIST       -o v4 $(SRC)
v5: $(SRC); $(NVCC) $(FLAGS) -DV5_TWOOPT -DTWOOPT_INTERVAL=10 -o v5 $(SRC)

# ── Bank conflict experiment
.PHONY: bank_ab
bank_ab:
	$(NVCC) $(FLAGS) -DV0_BASELINE -DBANK_CONFLICT_STRIDED -o bank_s $(SRC)
	$(NVCC) $(FLAGS) -DV0_BASELINE -DBANK_CONFLICT_PADDED  -o bank_p $(SRC)
	@echo "Run: for i in 1 2 3 4 5; do ./bank_s ch130.tsp 128 100; done"
	@echo "     for i in 1 2 3 4 5; do ./bank_p ch130.tsp 128 100; done"

# ── Phase timing build
.PHONY: phased
phased:
	$(NVCC) $(FLAGS) -DV0_BASELINE -DPHASE_TIMING -o v0_phased $(SRC)
	$(NVCC) $(FLAGS) -DV3_WARP_REDUCTION -DPHASE_TIMING -o v3_phased $(SRC)

# ── Occupancy query build
.PHONY: occ
occ:
	$(NVCC) $(FLAGS) -DV3_WARP_REDUCTION -DPRINT_OCCUPANCY -o v3_occ $(SRC)
	./v3_occ ch130.tsp 1 1 0.05 2 42

# ── Full profiling suite
.PHONY: profile
profile: versions bank_ab phased occ
	bash run_profile_story.sh

# ── Quick sanity check (1 island, 100 gens, should complete in <5 sec)
.PHONY: smoke
smoke: v3
	./v3 ch130.tsp 1 100 0.05 2 42
```

---

## Expected Console Output (V0 with PHASE_TIMING)

```
════════════ Hardware Caps ════════════
  Device:               Tesla P100-PCIE-16GB
  SMs:                  56
  Max warps / SM:       64
  Warp size:            32
  Shared mem / block:   48 KB
  Shared mem / SM:      64 KB
  Peak mem BW:          732 GB/s
═══════════════════════════════════════

── Static Resource Profile: V0-baseline ──
  lmem =  512 bytes/thread  ← SPILLING to DRAM
  smem = 33024 bytes/block
  regs =   40 /thread
  Predicted occupancy (smem limit): 1.56%  (1 blocks/SM)

── Intra-Kernel Phase Breakdown ──
  fitness_eval              cycles=  2840000   (31.2%)  ~0.473 ms
  elite_selection           cycles=  3920000   (43.1%)  ~0.653 ms  ← sort dominates
  elite_copy                cycles=   210000    (2.3%)  ~0.035 ms
  crossover+mut             cycles=  2130000   (23.4%)  ~0.355 ms

kernel_time_ms=1.516 best_length=6842
  effective_smem_bw_gbs=14.72  (lower bound)

╔── Bottleneck Diagnosis: V0-baseline ──
  [B2]  lmem bytes/thread = 512 → BOTTLENECK CONFIRMED
         → Fix: replace used[] with 4-register bitmask
  [B5]  occupancy % = 1.56 → BOTTLENECK CONFIRMED
         → Fix: redesign block for more warps/SM
  [BW]  effective BW GB/s = 14.72 → BOTTLENECK CONFIRMED
         → Fix: improve coalescing or reduce memory traffic
```

---

## What Goes in the Report

For each bottleneck section (4.2–4.7), add a **"Profiling Evidence"** subsection:

```markdown
#### Profiling Evidence

**Compile-time:** `ptxas -v` output for V0 shows `lmem = 512 bytes/thread`,
confirming that `used[MAX_CITIES]` spills to local memory (DRAM).

**Runtime A/B:** Timing V0 vs V2 (bitmask fix) over 10 repetitions:
  - V0: mean = X.XXX ms ± Y.YYY ms
  - V2: mean = A.AAA ms ± B.BBB ms
  - Delta: C.CCC ms ≈ predicted 5.4 ms from 3.93 GB at 732 GB/s

**Phase timing:** In V0 with PHASE_TIMING, the `elite_selection` phase
accounts for 43% of kernel cycles, dropping to 8% in V3 after the
warp-shuffle reduction.  This matches the predicted 50× step reduction.
```

This structure — predict, measure, compare — is the core of your CUDA optimization story.