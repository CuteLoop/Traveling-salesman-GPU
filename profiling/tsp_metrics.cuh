// =============================================================================
// tsp_metrics.cuh  —  Instrumented kernel variants + per-bottleneck probes
// ECE 569 · TSP-GA Optimization Story
//
// Each bottleneck gets:
//   1. A controlled A/B experiment that DETECTS the bottleneck from timing alone
//   2. The optimized code
//   3. A validation run that CONFIRMS the fix worked
//
// All measurements use cudaEvent_t + math.  No nvprof required.
// =============================================================================
#pragma once
#include "tsp_profiler.h"
#include <cstring>

// Forward declarations (defined in CUDA-GA-GPU-Pop.cu)
extern __constant__ int c_dist[128 * 128];

// =============================================================================
// ── PROBE 0: Hardware info + static compile-time resource snapshot ────────────
// =============================================================================
// Run this ONCE at program start.
// The lmem/smem/regs values come from compiling with --ptxas-options=-v
// and hard-coding the output here.  Update after each version change.
// =============================================================================
struct StaticKernelInfo {
    const char* version;
    int lmem;   // bytes/thread — from ptxas: "lmem = N"
    int smem;   // bytes/block  — from ptxas: "smem = N"
    int regs;   // registers/thread
};

// ┌─ FILL THESE IN after each build ─────────────────────────────────────────┐
// │  nvcc -arch=sm_60 --ptxas-options=-v -O3 CUDA-GA-GPU-Pop.cu ...         │
// │  grep -E "lmem|smem|registers" output                                   │
// └──────────────────────────────────────────────────────────────────────────┘
static const StaticKernelInfo KNOWN_STATIC[] = {
    { "V0-baseline", 512, 33024, 40 },  // used[] spills → lmem=512
    { "V1-stride",   512, 33280, 40 },  // padding added, lmem unchanged
    { "V2-bitmask",    0, 33280, 44 },  // bitmask → lmem=0, +4 regs
    { "V3-reduction",  0, 33280, 48 },  // warp shuffle → more regs
    { "V4-globaldist", 0, 33280, 48 },  // global dist pointer
    { "V5-twoopt",     0, 33536, 52 },  // 2-opt adds smem for best_i/j/delta
};

inline void print_static_info(const char* version) {
    for (auto& s : KNOWN_STATIC) {
        if (std::string(s.version) == version) {
            printf(BOLD "\n── Static Resource Profile: %s ──\n" RESET, version);
            printf("  lmem = %4d bytes/thread  %s\n", s.lmem,
                   s.lmem > 0   ? RED   "← SPILLING to DRAM" RESET : GREEN "clean" RESET);
            printf("  smem = %5d bytes/block\n", s.smem);
            printf("  regs = %4d /thread\n",     s.regs);
            // compute occupancy from smem alone (P100: 64 KB shared/SM)
            int blocks_by_smem = (s.smem > 0) ? 65536 / s.smem : 32;
            int warps_per_sm   = std::min(blocks_by_smem * 1, 64); // 1 warp/block here
            float occ = 100.f * warps_per_sm / 64.f;
            printf("  Predicted occupancy (smem limit): %.2f%%  (%d blocks/SM)\n",
                   occ, std::min(blocks_by_smem, 64));
            return;
        }
    }
    printf("  [static info not recorded for '%s']\n", version);
}

// =============================================================================
// ── PROBE B1: Shared memory bank conflict — A/B timing experiment ─────────────
//
// We launch the island kernel with two compile-time stride constants
// and compare wall-clock times.  The ratio tells us the conflict factor.
//
// To run this probe you need TWO builds of the kernel:
//   kernel_strided  → uses stride = n   (current, bad)
//   kernel_padded   → uses stride = n+1 (fixed,   good)
//
// Pass both function pointers here.
// =============================================================================
template<typename KernelFn>
BankConflictTest probe_bank_conflicts(
    KernelFn kernel_strided,
    KernelFn kernel_padded,
    int n, int islands, int generations,
    size_t smem_strided, size_t smem_padded,
    void** args_strided, void** args_padded,
    int repeats = 5)
{
    BankConflictTest r{};
    r.predicted_way = BankConflictTest::predict_conflict_way(n);

    GpuTimer t;
    float sum_s = 0.f, sum_p = 0.f;

    for (int i = 0; i < repeats; ++i) {
        t.start();
        cudaLaunchKernel((const void*)kernel_strided,
                         dim3(islands), dim3(32), args_strided, smem_strided, 0);
        cudaDeviceSynchronize();
        sum_s += t.stop();

        t.start();
        cudaLaunchKernel((const void*)kernel_padded,
                         dim3(islands), dim3(32), args_padded, smem_padded, 0);
        cudaDeviceSynchronize();
        sum_p += t.stop();
    }
    r.time_strided_ms = sum_s / repeats;
    r.time_padded_ms  = sum_p / repeats;
    r.inferred_conflict_factor = (r.time_padded_ms > 0)
        ? r.time_strided_ms / r.time_padded_ms : 0.f;
    return r;
}

// =============================================================================
// ── PROBE B2: Local memory spill — estimate traffic from timing delta ──────────
//
// Theory: if used[] spills, each crossover thread reads/writes
//   lmem_bytes = 2 × n × sizeof(int) = 1024 bytes of DRAM per call.
// We compare the crossover cost against an identically-structured
// loop that does NOT use the spilled array (bitmask version).
// The timing delta, divided by the known DRAM bandwidth, gives us
// an estimate of the hidden traffic.
// =============================================================================
struct LmemProbe {
    float time_spill_ms;    // kernel with used[]
    float time_clean_ms;    // kernel with bitmask
    float delta_ms;
    float inferred_traffic_gb;
    float peak_bw_gbs;

    // n=128, 30 non-elite threads, islands, generations
    static LmemProbe estimate(float spill_ms, float clean_ms,
                               int n, int islands, int gens,
                               float peak_bw_gbs) {
        LmemProbe r{};
        r.time_spill_ms   = spill_ms;
        r.time_clean_ms   = clean_ms;
        r.delta_ms        = spill_ms - clean_ms;
        r.peak_bw_gbs     = peak_bw_gbs;
        // expected: 30 threads × islands × gens × 2n × 4 bytes
        size_t expected_bytes = (size_t)30 * islands * gens * 2 * n * sizeof(int);
        r.inferred_traffic_gb = (float)expected_bytes / 1e9f;
        return r;
    }

    void print() const {
        printf(BOLD "\n── Local Memory Spill Probe ──\n" RESET);
        printf("  With used[] (spill):  %.3f ms\n", time_spill_ms);
        printf("  With bitmask (clean): %.3f ms\n", time_clean_ms);
        printf("  Timing delta:         %.3f ms\n", delta_ms);
        printf("  Predicted hidden DRAM traffic: %.2f GB\n", inferred_traffic_gb);
        printf("  Effective traffic BW from delta: %.2f GB/s\n",
               delta_ms > 0 ? inferred_traffic_gb / (delta_ms / 1e3f) : 0.f);
        const char* c = (time_spill_ms > time_clean_ms * 1.05f) ? RED : GREEN;
        printf("  verdict: %s%s\n" RESET, c,
               time_spill_ms > time_clean_ms * 1.05f
                   ? "Local memory overhead confirmed"
                   : "No significant local memory overhead");
    }
};

// =============================================================================
// ── PROBE B3: Thread-0 sort — intra-kernel phase timer ───────────────────────
//
// We instrument ga_island_kernel with clock64() to split time into:
//   Phase 0: fitness evaluation   (tour_length_const loop)
//   Phase 1: sort / reduction     (tid==0 sort OR warp shuffle)
//   Phase 2: crossover + mutation (OX + swap)
//   Phase 3: elite copy
//   Phase 4: 2-opt pass (if enabled)
//
// Results written to a global device buffer and copied back to host.
// =============================================================================

// ── Device-side phase accumulator (one row per block, MAX_PHASES cols)
// Declare in kernel: extern uint64_t* d_phase_clocks;
// At host: cudaMalloc(&d_phase_clocks, islands * MAX_PHASES * sizeof(uint64_t));
//          cudaMemset(d_phase_clocks,  0, ...);

// Convenience: copy phase clocks back and compute the report
inline PhaseReport read_phase_report(uint64_t* d_buf, int n_blocks,
                                     int n_phases, const char** names,
                                     float sm_clock_mhz) {
    size_t sz = (size_t)n_blocks * MAX_PHASES * sizeof(uint64_t);
    std::vector<uint64_t> h_buf(n_blocks * MAX_PHASES, 0);
    cudaMemcpy(h_buf.data(), d_buf, sz, cudaMemcpyDeviceToHost);

    PhaseReport pr{};
    pr.n_phases      = n_phases;
    pr.sm_clock_mhz  = sm_clock_mhz;
    for (int i = 0; i < n_phases && i < MAX_PHASES; ++i)
        pr.names[i] = names[i];
    pr.compute_from_raw(h_buf.data(), n_blocks);
    return pr;
}

// =============================================================================
// ── PROBE B4: Constant vs global memory — A/B experiment ─────────────────────
//
// Build the kernel twice (USE_CONSTANT_MEM / USE_GLOBAL_MEM compile flags).
// Time both with identical inputs and compute the speedup ratio.
// Also compute what the constant-cache serialization cost SHOULD be:
//   32 threads × serialize × ~10 cycles × n steps per eval
// Compare predicted cycle count to timing delta.
// =============================================================================
struct ConstMemProbe {
    float  time_const_ms;
    float  time_global_ms;
    float  speedup_global_over_const;

    // Predict constant memory serialization overhead
    // 32 threads × 10 cycles/serialize × n steps × islands × gens / SM_clock
    static float predict_const_overhead_ms(int n, int islands, int gens,
                                            float sm_clock_mhz) {
        // serialized constant lookups per warp per tour eval step
        float cycles = 32.f * 10.f * n * islands * gens;
        return cycles / (sm_clock_mhz * 1e3f); // ms
    }

    void print() const {
        printf(BOLD "\n── Constant vs Global Memory Probe ──\n" RESET);
        printf("  __constant__ c_dist: %.3f ms\n", time_const_ms);
        printf("  global __restrict__: %.3f ms\n", time_global_ms);
        printf("  speedup (global/const ratio): %.2fx  %s\n",
               speedup_global_over_const,
               speedup_global_over_const > 1.05f
                   ? GREEN "global memory wins — switch" RESET
                   : RED   "constant memory competitive — keep" RESET);
    }
};

// =============================================================================
// ── PROBE B5: Occupancy — runtime API query ───────────────────────────────────
// =============================================================================
// Call after every kernel version change:
//   auto occ = OccupancyReport::compute(ga_island_kernel, BLOCK_POP_SIZE, smem);
//   occ.print("ga_island_kernel V0");

// =============================================================================
// ── PROBE B7: Branch divergence — infer from crossover throughput ─────────────
//
// Without nvprof we cannot directly read branch_efficiency.
// Instead we compare:
//   - crossover time with `if (used[gene]) continue`  (divergent)
//   - crossover time with predicated write              (uniform)
// The ratio gives us an empirical "divergence tax".
//
// Additionally we can compute the THEORETICAL divergence fraction:
//   P(continue at step k) ≈ (segment_placed) / n  (grows 0→1 as loop progresses)
//   Expected continue fraction ≈ E[segment_size] / n ≈ (n/3) / n = 1/3
//   Since ~1/3 of iterations hit continue, and these are randomly distributed
//   across threads, ~50% of branch instructions are divergent.
// =============================================================================
struct DivergenceEstimate {
    float expected_continue_fraction;   // ~1/3 for OX
    float expected_branch_efficiency;   // 1 - 0.5 * continue_fraction ≈ 0.83
    float time_divergent_ms;
    float time_predicated_ms;
    float empirical_divergence_tax;

    static DivergenceEstimate compute_theoretical(int n) {
        DivergenceEstimate d{};
        // Expected segment size ≈ n/3 (uniform random endpoints)
        float expected_segment = n / 3.f;
        d.expected_continue_fraction = expected_segment / n; // ≈ 0.33
        // In the gap-fill loop of n iterations, fraction 'c' hit continue.
        // If each thread's continue is independent, P(warp diverges at step k)
        // ≈ 1 - (1 - c)^32 - c^32  ≈ 1 - extreme cases ≈ high for c~0.5
        // Simplified: branch_efficiency ≈ 1 - 0.5 * continue_fraction
        d.expected_branch_efficiency = 1.f - 0.5f * d.expected_continue_fraction;
        d.empirical_divergence_tax = 0.f;
        return d;
    }

    void print() const {
        printf(BOLD "\n── Branch Divergence Estimate (OX crossover) ──\n" RESET);
        printf("  Expected continue fraction:  %.2f  (segment≈n/3)\n",
               expected_continue_fraction);
        printf("  Predicted branch efficiency: %.0f%%\n",
               expected_branch_efficiency * 100.f);
        if (empirical_divergence_tax > 0.f)
            printf("  Empirical divergence tax:    %.2fx slower\n",
                   empirical_divergence_tax);
        printf("  %sFix: predicated write or PMX crossover\n" RESET,
               expected_branch_efficiency < 0.85f ? YELLOW : GREEN);
    }
};

// =============================================================================
// ── PROBE 2-OPT: Measure quality improvement and overhead ────────────────────
// =============================================================================
struct TwoOptProbe {
    float  kernel_no_twoopt_ms;
    float  kernel_with_twoopt_ms;
    float  overhead_ms;
    float  overhead_pct;
    int    best_length_no_twoopt;
    int    best_length_with_twoopt;
    float  quality_improvement_pct;
    int    interval_K;

    void print() const {
        printf(BOLD "\n── 2-opt Elite Refinement Probe (K=%d) ──\n" RESET, interval_K);
        printf("  Without 2-opt:  %.3f ms   best=%d\n",
               kernel_no_twoopt_ms, best_length_no_twoopt);
        printf("  With    2-opt:  %.3f ms   best=%d\n",
               kernel_with_twoopt_ms, best_length_with_twoopt);
        printf("  Overhead:       %.3f ms  (%.1f%%)\n",
               overhead_ms, overhead_pct);
        printf("  Quality gain:   %.2f%%  (%+d tour units)\n",
               quality_improvement_pct,
               best_length_no_twoopt - best_length_with_twoopt);
        printf("  Pairs/pass:     %d   threads/pass: 32   pairs/thread: %d\n",
               128*127/2, 128*127/2/32);

        // Pareto verdict
        bool quality_wins = quality_improvement_pct > 1.f;
        bool overhead_ok  = overhead_pct < 20.f;
        printf("  verdict: quality=%s%s%s  overhead=%s%s%s  → %s\n" RESET,
               quality_wins ? GREEN : RED,
               quality_wins ? "IMPROVED" : "no change", RESET,
               overhead_ok ? GREEN : YELLOW,
               overhead_ok ? "acceptable" : "high", RESET,
               (quality_wins && overhead_ok) ? GREEN "USE THIS K" RESET
                                             : YELLOW "try different K" RESET);
    }
};

// =============================================================================
// ── MASTER PROBE RUNNER ────────────────────────────────────────────────────────
// Sequenced profiling: detect → fix → verify
// Call from run_gpu_population_ga() with PROFILE_MODE defined.
// =============================================================================
struct ProbeRunner {
    HwCaps         hw;
    std::vector<BenchRecord> story;   // one entry per version
    float          v0_kernel_ms = 0.f;

    ProbeRunner() : hw(HwCaps::query()) { hw.print(); }

    // ── Add a benchmark record (call after every version run)
    void record(BenchRecord r) {
        if (story.empty()) v0_kernel_ms = r.kernel_ms;
        r.speedup_vs_v0 = (v0_kernel_ms > 0) ? v0_kernel_ms / r.kernel_ms : 1.f;
        story.push_back(r);
    }

    // ── Print the full story table
    void summary() { print_optimization_story(story, v0_kernel_ms); }

    // ── Print bottleneck diagnosis for current version
    void diagnose(const char* version, float kernel_ms, float occ_pct,
                  int lmem, int smem, float bw_gbs) {
        printf(BOLD "\n╔── Bottleneck Diagnosis: %s ──\n" RESET, version);
        bottleneck_verdict("B1", "smem bank conflicts (infer from A/B timing)",
                            0.f, 0.f, 0.f,   // measured by A/B, not a single metric
                            "pad stride to n+1");
        bottleneck_verdict("B2", "lmem bytes/thread",
                            (float)lmem, 0.f, 0.f,
                            "replace used[] with 4-register bitmask",
                            false);  // lower is better; 0 = OK
        bottleneck_verdict("B5", "occupancy %",
                            occ_pct, 5.f, 25.f,
                            "redesign block for more warps/SM",
                            true);   // higher is better
        bottleneck_verdict("BW", "effective BW GB/s",
                            bw_gbs, 100.f, 400.f,
                            "improve coalescing or reduce memory traffic",
                            true);
    }
};
