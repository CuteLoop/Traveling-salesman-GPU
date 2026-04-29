// =============================================================================
// tsp_profiler.h  —  Self-contained CUDA profiling without nvprof
// ECE 569 · TSP-GA Optimization Story
//
// Collects the metrics that matter for each bottleneck using only:
//   • cudaEvent_t         → wall-clock kernel timing
//   • cudaOccupancy*      → theoretical + achieved occupancy
//   • cudaGetDeviceProperties → hardware caps
//   • clock64() inside kernels → intra-kernel phase timing
//   • compile-time ptxas  → lmem / smem / register counts
//   • arithmetic          → effective bandwidth, transaction counts
//
// Usage: #include "tsp_profiler.h" in CUDA-GA-GPU-Pop.cu
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

// ─── colour helpers for terminal output ───────────────────────────────────────
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define GREEN   "\033[32m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"
#define RESET   "\033[0m"

// =============================================================================
// 1.  TIMER  —  cudaEvent wrapping with fluent API
// =============================================================================
struct GpuTimer {
    cudaEvent_t _start, _stop;
    float       _ms = 0.f;

    GpuTimer()  { cudaEventCreate(&_start); cudaEventCreate(&_stop); }
    ~GpuTimer() { cudaEventDestroy(_start);  cudaEventDestroy(_stop); }

    void start() { cudaEventRecord(_start); }
    float stop()  {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&_ms, _start, _stop);
        return _ms;
    }
    float ms()  const { return _ms; }
    float gb_per_sec(size_t bytes) const {
        return (_ms > 0) ? (float)bytes / _ms / 1e6f : 0.f;
    }
};

// =============================================================================
// 2.  HARDWARE CAPS  —  P100 baseline from device query
// =============================================================================
struct HwCaps {
    char  name[256];
    int   sm_count;
    int   max_warps_per_sm;
    int   max_threads_per_block;
    int   warp_size;
    int   shared_mem_per_block;   // bytes (configurable max)
    int   shared_mem_per_sm;      // bytes
    int   regs_per_sm;
    float mem_clock_khz;
    int   mem_bus_width;          // bits
    float peak_bw_gbs;            // GB/s  = 2 × clock × width / 8 / 1e6

    static HwCaps query(int dev = 0) {
        HwCaps h{};
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);
        snprintf(h.name, sizeof(h.name), "%s", p.name);
        h.sm_count             = p.multiProcessorCount;
        h.max_warps_per_sm     = p.maxThreadsPerMultiProcessor / p.warpSize;
        h.max_threads_per_block= p.maxThreadsPerBlock;
        h.warp_size            = p.warpSize;
        h.shared_mem_per_block = (int)p.sharedMemPerBlockOptin;
        h.shared_mem_per_sm    = (int)p.sharedMemPerMultiprocessor;
        h.regs_per_sm          = p.regsPerMultiprocessor;
        h.mem_clock_khz        = (float)p.memoryClockRate;
        h.mem_bus_width        = p.memoryBusWidth;
        h.peak_bw_gbs = 2.f * p.memoryClockRate * 1e3f
                            * (p.memoryBusWidth / 8.f) / 1e9f;
        return h;
    }

    void print() const {
        printf(BOLD CYAN "\n════════════ Hardware Caps ════════════\n" RESET);
        printf("  Device:               %s\n", name);
        printf("  SMs:                  %d\n", sm_count);
        printf("  Max warps / SM:       %d\n", max_warps_per_sm);
        printf("  Warp size:            %d\n", warp_size);
        printf("  Shared mem / block:   %d KB\n", shared_mem_per_block/1024);
        printf("  Shared mem / SM:      %d KB\n", shared_mem_per_sm/1024);
        printf("  Peak mem BW:          %.0f GB/s\n", peak_bw_gbs);
        printf(CYAN "═══════════════════════════════════════\n\n" RESET);
    }
};

// =============================================================================
// 3.  OCCUPANCY  —  theoretical + API query
// =============================================================================
struct OccupancyReport {
    int   max_active_blocks_per_sm;
    int   active_warps_per_sm;
    int   max_warps_per_sm;
    float occupancy_pct;

    template<typename Kernel>
    static OccupancyReport compute(Kernel k, int block_size,
                                   size_t smem_bytes, int dev = 0) {
        OccupancyReport r{};
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        r.max_warps_per_sm = p.maxThreadsPerMultiProcessor / p.warpSize;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &r.max_active_blocks_per_sm, k, block_size, smem_bytes);
        r.active_warps_per_sm = r.max_active_blocks_per_sm
                                * (block_size / p.warpSize);
        r.occupancy_pct = 100.f * r.active_warps_per_sm / r.max_warps_per_sm;
        return r;
    }

    void print(const char* label = "Kernel") const {
        const char* color = (occupancy_pct < 5.f)  ? RED :
                            (occupancy_pct < 25.f) ? YELLOW : GREEN;
        printf("  %-28s  blocks/SM=%d  warps/SM=%d/%d  "
               "%s occupancy=%.2f%%\n" RESET,
               label,
               max_active_blocks_per_sm,
               active_warps_per_sm, max_warps_per_sm,
               color, occupancy_pct);
    }
};

// =============================================================================
// 4.  BENCHMARK RECORD  —  one entry per version per experiment
// =============================================================================
struct BenchRecord {
    std::string label;
    float  kernel_ms   = 0.f;
    float  transfer_ms = 0.f;
    float  total_ms    = 0.f;
    float  eff_bw_gbs  = 0.f;   // effective memory bandwidth
    float  occupancy   = 0.f;
    int    best_length = 0;
    // static resource info (filled from ptxas -v output or hardcoded)
    int    lmem_bytes  = -1;     // -1 = not measured
    int    smem_bytes  = -1;
    int    regs        = -1;
    // derived metric predictions
    float  speedup_vs_v0 = 1.f;

    void print_header() {
        printf(BOLD
            "\n%-10s %10s %10s %10s %10s %9s %8s %7s %7s %7s\n" RESET,
            "Version", "kernel_ms", "xfer_ms", "total_ms",
            "BW_GB/s", "occ_%", "best_L",
            "lmem", "smem", "speedup");
        printf("%-10s %10s %10s %10s %10s %9s %8s %7s %7s %7s\n",
            "──────────", "─────────", "─────────", "─────────",
            "─────────", "────────", "───────",
            "──────", "──────", "───────");
    }

    void print_row() const {
        const char* c = (speedup_vs_v0 > 2.f) ? GREEN :
                        (speedup_vs_v0 > 1.1f) ? YELLOW : RESET;
        printf("%-10s %10.3f %10.3f %10.3f %10.2f %9.2f %8d"
               " %7d %7d %s%7.2fx\n" RESET,
               label.c_str(),
               kernel_ms, transfer_ms, total_ms,
               eff_bw_gbs, occupancy, best_length,
               lmem_bytes, smem_bytes,
               c, speedup_vs_v0);
    }
};

// =============================================================================
// 5.  BOTTLENECK DIAGNOSTICS  —  controlled A/B experiments in pure CUDA
// =============================================================================

// ── 5a.  Measure effective memory bandwidth ───────────────────────────────────
// Knowing: data bytes moved + kernel time → BW = bytes / time
inline float compute_eff_bw(size_t bytes_total, float ms) {
    return (ms > 0.f) ? (float)bytes_total / ms / 1e6f : 0.f; // GB/s
}

// ── 5b.  Bank conflict ratio inference ───────────────────────────────────────
// Run the kernel twice: once with stride=n (bad), once with stride=n+1 (good).
// Ratio of times approximates the bank conflict serialization factor.
// Expected: ~32× for n=128 (power of 2).
struct BankConflictTest {
    float  time_strided_ms;    // stride = n   (conflict)
    float  time_padded_ms;     // stride = n+1 (no conflict)
    float  inferred_conflict_factor;
    int    predicted_way;      // n % 32 == 0 → 32-way

    void print() const {
        printf(BOLD "\n── Bank Conflict Diagnostic ──\n" RESET);
        printf("  stride=n   (conflicted):  %.3f ms\n", time_strided_ms);
        printf("  stride=n+1 (padded):      %.3f ms\n", time_padded_ms);
        printf("  inferred conflict factor: %.1fx  (predicted: %d-way)\n",
               inferred_conflict_factor, predicted_way);
        const char* verdict = (inferred_conflict_factor > 8.f) ? RED :
                              (inferred_conflict_factor > 2.f) ? YELLOW : GREEN;
        printf("  verdict: %s%s\n" RESET, verdict,
               inferred_conflict_factor > 8.f ? "SEVERE bank conflicts confirmed" :
               inferred_conflict_factor > 2.f ? "Moderate bank conflicts" :
                                                "No significant bank conflicts");
    }

    static int predict_conflict_way(int n) {
        // worst case: n % 32 == 0 → 32-way
        int g = __gcd(n, 32);
        return 32 / g;
    }
};

// ── 5c.  PCIe transfer fraction ──────────────────────────────────────────────
struct PcieReport {
    float  transfer_ms;
    float  kernel_ms;
    float  total_ms;
    float  transfer_fraction_pct;
    float  bytes_uploaded;
    float  bytes_downloaded;
    float  achieved_bw_gbs;

    void print() const {
        printf(BOLD "\n── PCIe Transfer Analysis ──\n" RESET);
        printf("  Bytes uploaded:     %.2f KB\n", bytes_uploaded / 1024.f);
        printf("  Bytes downloaded:   %.2f KB\n", bytes_downloaded / 1024.f);
        printf("  Transfer time:      %.3f ms  (%.1f%%)\n",
               transfer_ms, transfer_fraction_pct);
        printf("  Kernel time:        %.3f ms\n", kernel_ms);
        printf("  Achieved H↔D BW:    %.2f GB/s\n", achieved_bw_gbs);
        const char* color = (transfer_fraction_pct > 15.f) ? RED : GREEN;
        printf("  verdict: %sPCIe is %s\n" RESET, color,
               transfer_fraction_pct > 15.f
                   ? "a bottleneck — move data to GPU permanently"
                   : "acceptable");
    }
};

// ── 5d.  Local memory spill check  (compile-time only — checked via ptxas -v)
// We encode the expected values here so the runtime report can compare.
struct LmemReport {
    int lmem_bytes_before;  // with used[MAX_CITIES]
    int lmem_bytes_after;   // with bitmask
    size_t estimated_dram_traffic_bytes;

    void print() const {
        printf(BOLD "\n── Local Memory Spill Report ──\n" RESET);
        printf("  lmem before fix: %d bytes/thread  (%s)\n",
               lmem_bytes_before,
               lmem_bytes_before > 0 ? RED "SPILLING" RESET : GREEN "clean" RESET);
        printf("  lmem after fix:  %d bytes/thread  (%s)\n",
               lmem_bytes_after,
               lmem_bytes_after > 0 ? RED "still spilling" RESET : GREEN "clean" RESET);
        if (lmem_bytes_before > 0) {
            printf("  estimated hidden DRAM traffic: %.2f GB\n",
                   (float)estimated_dram_traffic_bytes / 1e9f);
        }
    }
};

// =============================================================================
// 6.  INTRA-KERNEL PHASE TIMING  —  clock64() instrumentation
//     Add these macros inside the kernel to time phases.
//     Results must be copied back to host via a d_phase_times[] buffer.
// =============================================================================
// In kernel: declare  uint64_t* d_phase_times = ...  (one slot per phase per block)
// Then:
//   PHASE_START(0);   // start timer for phase 0
//   ... code ...
//   PHASE_STOP(0);    // accumulate cycles for phase 0

// Maximum phases we track per kernel invocation
#define MAX_PHASES 8

// Use these INSIDE __global__ kernels (device clock64())
#define PHASE_START(i) uint64_t __t_start_##i = clock64()
#define PHASE_STOP(i, buf, block_base)  \
    atomicAdd((unsigned long long*)&(buf)[(block_base) + (i)], \
              (unsigned long long)(clock64() - __t_start_##i))

// Host-side interpretation
struct PhaseReport {
    const char* names[MAX_PHASES];
    float       cycles[MAX_PHASES];
    float       pct[MAX_PHASES];
    int         n_phases;
    float       sm_clock_mhz;  // from device props

    void compute_from_raw(uint64_t* raw, int n_blocks) {
        double total = 0;
        for (int i = 0; i < n_phases; ++i) {
            cycles[i] = 0;
            for (int b = 0; b < n_blocks; ++b)
                cycles[i] += (float)raw[b * MAX_PHASES + i];
            cycles[i] /= n_blocks; // average across blocks
            total += cycles[i];
        }
        for (int i = 0; i < n_phases; ++i)
            pct[i] = (total > 0) ? 100.f * cycles[i] / (float)total : 0.f;
    }

    void print() const {
        printf(BOLD "\n── Intra-Kernel Phase Breakdown ──\n" RESET);
        for (int i = 0; i < n_phases; ++i) {
            float ms = (sm_clock_mhz > 0) ? cycles[i] / (sm_clock_mhz * 1e3f) : -1.f;
            printf("  %-28s  cycles=%10.0f  (%5.1f%%)  ~%.3f ms\n",
                   names[i], cycles[i], pct[i], ms);
        }
    }
};

// =============================================================================
// 7.  SUMMARY PRINTER  —  full optimization story table
// =============================================================================
inline void print_optimization_story(std::vector<BenchRecord>& recs,
                                     float v0_kernel_ms) {
    printf(BOLD CYAN
        "\n╔══════════════════════════════════════════════════════════════════╗\n"
        "║            OPTIMIZATION STORY — ECE 569 TSP-GA                 ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
        RESET);

    for (auto& r : recs) {
        if (v0_kernel_ms > 0)
            r.speedup_vs_v0 = v0_kernel_ms / r.kernel_ms;
    }
    if (!recs.empty()) {
        recs[0].print_header();
        for (auto& r : recs) r.print_row();
    }
}

// =============================================================================
// 8.  BOTTLENECK VERDICT  —  call after each measurement
// =============================================================================
inline void bottleneck_verdict(const char* id, const char* metric,
                                float measured, float threshold_bad,
                                float threshold_ok,
                                const char* fix, bool higher_is_better = false) {
    bool is_bad = higher_is_better ? (measured < threshold_bad)
                                   : (measured > threshold_bad);
    bool is_ok  = higher_is_better ? (measured >= threshold_ok)
                                   : (measured <= threshold_ok);
    const char* color = is_bad ? RED : (is_ok ? GREEN : YELLOW);
    const char* verdict = is_bad ? "BOTTLENECK CONFIRMED" :
                          is_ok  ? "OK"                   : "MARGINAL";
    printf("  %s[%s]%s  %s = %.3f → %s%s%s\n" RESET,
           BOLD, id, RESET,
           metric, measured,
           color, verdict, RESET);
    if (is_bad)
        printf("         → Fix: %s\n", fix);
}
