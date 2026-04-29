// =============================================================================
// CUDA-GA-GPU-Pop-instrumented.cu
// ECE 569 · TSP-GA — Drop-in replacement for CUDA-GA-GPU-Pop.cu
//
// Compile-time feature flags (pass via -D flags):
//   -DV0_BASELINE          → original code
//   -DV1_STRIDE_PAD        → B1 fix: shared memory stride = n+1
//   -DV2_BITMASK           → B2 fix: used[] → 4-register bitmask
//   -DV3_WARP_REDUCTION    → B3 fix: tid==0 sort → warp shuffle top-2
//   -DV4_GLOBAL_DIST       → B4 fix: __constant__ → global __restrict__
//   -DV5_TWOOPT            → B8 new: in-kernel 2-opt on elite
//   -DTWOOPT_INTERVAL=K    → run 2-opt every K generations
//   -DPHASE_TIMING         → enable intra-kernel clock64() instrumentation
//   -DPRINT_OCCUPANCY      → print occupancy via CUDA API before launch
//   -DBANK_CONFLICT_STRIDED → force stride=n  for A/B experiment
//   -DBANK_CONFLICT_PADDED  → force stride=n+1 for A/B experiment
//
// Printf output format (parsed by run_profile_story.sh):
//   kernel_time_ms=X.XXX best_length=NNNNNN
// =============================================================================
#include "tsp_profiler.h"
#include "tsp_metrics.cuh"
#include "tsplib_parser.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

// ─── Constants ───────────────────────────────────────────────────────────────
constexpr int MAX_CITIES    = 128;
constexpr int BLOCK_POP_SIZE= 32;
constexpr int TOURNAMENT_SIZE=3;

#ifndef TWOOPT_INTERVAL
#define TWOOPT_INTERVAL 0   // disabled unless explicitly set
#endif

// ─── Distance matrix in constant memory (V0–V3) or global (V4+) ──────────────
#ifndef V4_GLOBAL_DIST
__constant__ int c_dist[MAX_CITIES * MAX_CITIES];
#endif

// ─── Phase timing buffer (optional, enabled with -DPHASE_TIMING) ─────────────
#ifdef PHASE_TIMING
__device__ uint64_t d_phase_clocks_dev[128 * MAX_PHASES]; // 128 islands × 8 phases
#endif

#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(e));            \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// =============================================================================
// DEVICE UTILITIES
// =============================================================================
__device__ unsigned int xorshift32(unsigned int& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
__device__ int  rand_bounded(unsigned int& s, int b) {
    return (int)(xorshift32(s) % (unsigned)b);
}
__device__ float rand_unit(unsigned int& s) {
    return (float)xorshift32(s) / 4294967295.f;
}

// =============================================================================
// TOUR LENGTH
// =============================================================================
#ifdef V4_GLOBAL_DIST
__device__ int tour_length(const int* __restrict__ dist,
                            const int* tour, int n) {
    int t = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k], b = tour[(k+1)%n];
        t += dist[a*n + b];
    }
    return t;
}
#else
__device__ int tour_length(const int* tour, int n) {
    int t = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k], b = tour[(k+1)%n];
        t += c_dist[a*n + b];
    }
    return t;
}
#endif

// =============================================================================
// INITIALIZATION
// =============================================================================
__device__ void init_random_tour(int* tour, int n, unsigned int& rng) {
    for (int i = 0; i < n; ++i) tour[i] = i;
    for (int i = n-1; i > 0; --i) {
        int j = rand_bounded(rng, i+1);
        int tmp = tour[i]; tour[i] = tour[j]; tour[j] = tmp;
    }
}

// =============================================================================
// TOURNAMENT SELECTION
// =============================================================================
__device__ int tournament_select(const int* lengths, unsigned int& rng) {
    int best = rand_bounded(rng, BLOCK_POP_SIZE);
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        int c = rand_bounded(rng, BLOCK_POP_SIZE);
        if (lengths[c] < lengths[best]) best = c;
    }
    return best;
}

// =============================================================================
// ORDER CROSSOVER — V0/V1: original with local memory spill
//                  V2+:    bitmask in registers
// =============================================================================
#ifdef V2_BITMASK  // also V3, V4, V5
__device__ void order_crossover(const int* pa, const int* pb,
                                 int* child, int n, unsigned int& rng) {
    int left  = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) { int t=left; left=right; right=t; }

    // ── 4-register bitmask replaces int used[MAX_CITIES] (zero local memory)
    uint32_t u0=0, u1=0, u2=0, u3=0;

    #define MARK(c) do { uint32_t bit=1u<<((c)&31); \
        if((c)<32) u0|=bit; else if((c)<64) u1|=bit; \
        else if((c)<96) u2|=bit; else u3|=bit; } while(0)
    #define ISSET(c) ( (c)<32 ? (u0>>((c)   ))&1 : \
                       (c)<64 ? (u1>>((c)-32 ))&1 : \
                       (c)<96 ? (u2>>((c)-64 ))&1 : \
                                (u3>>((c)-96 ))&1 )

    for (int i = 0; i < n; ++i) child[i] = -1;
    for (int i = left; i <= right; ++i) { child[i] = pa[i]; MARK(pa[i]); }

    int out = (right+1) % n;
    for (int off = 1; off <= n; ++off) {
        int gene = pb[(right+off) % n];
        if (ISSET(gene)) continue;
        child[out] = gene;
        MARK(gene);
        out = (out+1) % n;
    }
    #undef MARK
    #undef ISSET
}

#else  // V0 / V1 — original (used[] spills to local memory)
__device__ void order_crossover(const int* pa, const int* pb,
                                 int* child, int n, unsigned int& rng) {
    int left  = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) { int t=left; left=right; right=t; }

    int used[MAX_CITIES];                // ← SPILLS to local memory (DRAM)
    for (int i = 0; i < n; ++i) { child[i] = -1; used[i] = 0; }
    for (int i = left; i <= right; ++i) { child[i] = pa[i]; used[pa[i]] = 1; }

    int out = (right+1) % n;
    for (int off = 1; off <= n; ++off) {
        int gene = pb[(right+off) % n];
        if (used[gene]) continue;
        child[out] = gene;
        used[gene] = 1;
        out = (out+1) % n;
    }
}
#endif

// =============================================================================
// MUTATION
// =============================================================================
__device__ void mutate_swap(int* tour, int n, float mut_rate, unsigned int& rng) {
    if (rand_unit(rng) > mut_rate) return;
    int a = rand_bounded(rng, n), b;
    do { b = rand_bounded(rng, n); } while (b == a);
    int tmp = tour[a]; tour[a] = tour[b]; tour[b] = tmp;
}

// =============================================================================
// ELITE SELECTION  — V0–V2: tid==0 sort   V3+: warp shuffle reduction
// =============================================================================
#ifdef V3_WARP_REDUCTION  // also V4, V5
__device__ void find_top2(const int* lengths, int& e0, int& e1) {
    int my_len = lengths[threadIdx.x], my_idx = threadIdx.x;

    // Pass 1 — find global minimum via warp shuffle
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_len = __shfl_xor_sync(0xffffffff, my_len, mask);
        int o_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
        if (o_len < my_len || (o_len == my_len && o_idx < my_idx))
            { my_len = o_len; my_idx = o_idx; }
    }
    e0 = __shfl_sync(0xffffffff, my_idx, 0);

    // Pass 2 — find second minimum (exclude e0)
    my_len = (threadIdx.x == e0) ? INT_MAX : lengths[threadIdx.x];
    my_idx = threadIdx.x;
    for (int mask = 16; mask > 0; mask >>= 1) {
        int o_len = __shfl_xor_sync(0xffffffff, my_len, mask);
        int o_idx = __shfl_xor_sync(0xffffffff, my_idx, mask);
        if (o_len < my_len || (o_len == my_len && o_idx < my_idx))
            { my_len = o_len; my_idx = o_idx; }
    }
    e1 = __shfl_sync(0xffffffff, my_idx, 0);
}
#endif  // V3_WARP_REDUCTION

// =============================================================================
// STRIDE SELECTION  — controlled by compile flags
// =============================================================================
#if defined(BANK_CONFLICT_PADDED) || defined(V1_STRIDE_PAD) || \
    defined(V2_BITMASK) || defined(V3_WARP_REDUCTION) || \
    defined(V4_GLOBAL_DIST) || defined(V5_TWOOPT)
  #define STRIDE(n) ((n) + 1)
#else
  #define STRIDE(n) (n)            // BANK_CONFLICT_STRIDED or V0_BASELINE
#endif

// =============================================================================
// MAIN ISLAND KERNEL
// =============================================================================
__global__ void ga_island_kernel(
        int n, int generations, float mutation_rate, int elite_count,
        unsigned int seed,
#ifdef V4_GLOBAL_DIST
        const int* __restrict__ dist,
#endif
#ifdef PHASE_TIMING
        uint64_t* phase_buf,     // islands × MAX_PHASES
#endif
        int* best_tours, int* best_lengths)
{
    extern __shared__ int shared[];
    const int tid    = threadIdx.x;
    const int island = blockIdx.x;

    // ── Shared memory layout
    const int stride  = STRIDE(n);
    int* pop_a   = shared;
    int* pop_b   = pop_a + BLOCK_POP_SIZE * stride;
    int* lengths = pop_b + BLOCK_POP_SIZE * stride;
    int* order   = lengths + BLOCK_POP_SIZE;   // only used in V0–V2
#ifdef V5_TWOOPT
    // 2-opt needs 3 extra shared ints: best_delta, best_i, best_j
    int* s_2opt  = order + BLOCK_POP_SIZE;
#endif

    // ── RNG
    unsigned int rng = seed
        ^ ((unsigned)(island+1) * 747796405u)
        ^ ((unsigned)(tid   +1) * 2891336453u);
    if (!rng) rng = 1;

    // ── Initialize population
    init_random_tour(pop_a + tid * stride, n, rng);
    __syncthreads();

    int* current = pop_a;
    int* next    = pop_b;

    // ── Generation loop ──────────────────────────────────────────────────────
    for (int gen = 0; gen < generations; ++gen) {

        // PHASE 0: fitness evaluation
#ifdef PHASE_TIMING
        PHASE_START(0);
#endif
#ifdef V4_GLOBAL_DIST
        lengths[tid] = tour_length(dist, current + tid * stride, n);
#else
        lengths[tid] = tour_length(current + tid * stride, n);
#endif
        __syncthreads();
#ifdef PHASE_TIMING
        PHASE_STOP(0, phase_buf, island * MAX_PHASES);
#endif

        // PHASE 1: elite selection
#ifdef PHASE_TIMING
        PHASE_START(1);
#endif
#if defined(V3_WARP_REDUCTION) || defined(V4_GLOBAL_DIST) || defined(V5_TWOOPT)
        int e0, e1;
        find_top2(lengths, e0, e1);
        // All threads now know e0, e1 via shuffle broadcast
#else
        // V0–V2: tid==0 selection sort  ← BOTTLENECK B3
        if (tid == 0) {
            for (int i = 0; i < BLOCK_POP_SIZE; ++i) order[i] = i;
            for (int i = 0; i < BLOCK_POP_SIZE-1; ++i) {
                int best = i;
                for (int j = i+1; j < BLOCK_POP_SIZE; ++j)
                    if (lengths[order[j]] < lengths[order[best]]) best = j;
                int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
            }
        }
        __syncthreads();
        int e0 = order[0], e1 = order[1];
#endif
        __syncthreads();
#ifdef PHASE_TIMING
        PHASE_STOP(1, phase_buf, island * MAX_PHASES);
#endif

        // PHASE 2: elite copy
#ifdef PHASE_TIMING
        PHASE_START(2);
#endif
        if (tid < elite_count) {
            int src = (tid == 0) ? e0 : e1;
            for (int k = 0; k < n; ++k)
                next[tid * stride + k] = current[src * stride + k];
        }
        __syncthreads();
#ifdef PHASE_TIMING
        PHASE_STOP(2, phase_buf, island * MAX_PHASES);
#endif

        // PHASE 3: crossover + mutation
#ifdef PHASE_TIMING
        PHASE_START(3);
#endif
        if (tid >= elite_count && tid < BLOCK_POP_SIZE) {
            int pa_idx = tournament_select(lengths, rng);
            int pb_idx = tournament_select(lengths, rng);
            const int* pa = current + pa_idx * stride;
            const int* pb = current + pb_idx * stride;
            int*       ch = next    + tid    * stride;
            order_crossover(pa, pb, ch, n, rng);
            mutate_swap(ch, n, mutation_rate, rng);
        }
        __syncthreads();
#ifdef PHASE_TIMING
        PHASE_STOP(3, phase_buf, island * MAX_PHASES);
#endif

        // PHASE 4 (optional): 2-opt on elite
#if defined(V5_TWOOPT) && TWOOPT_INTERVAL > 0
#ifdef PHASE_TIMING
        PHASE_START(4);
#endif
        if (gen % TWOOPT_INTERVAL == 0) {
            if (tid == 0) { s_2opt[0] = 0; s_2opt[1] = -1; s_2opt[2] = -1; }
            __syncthreads();

            int* elite_tour = next + 0 * stride;  // elite 0 is at next[0]
            int my_d = 0, my_i = -1, my_j = -1;
            int total_pairs = n * (n-1) / 2;

            for (int pair = tid; pair < total_pairs; pair += BLOCK_POP_SIZE) {
                int pi = (int)floorf((-1.f + sqrtf(1.f + 8.f*pair)) * .5f);
                int pj = pair - pi*(pi+1)/2 + pi + 1;
                int a = elite_tour[pi], b = elite_tour[(pi+1)%n];
                int c = elite_tour[pj], d = elite_tour[(pj+1)%n];
                int delta = c_dist[a*n+c] + c_dist[b*n+d]
                           - c_dist[a*n+b] - c_dist[c*n+d];
                if (delta < my_d) { my_d=delta; my_i=pi; my_j=pj; }
            }

            // Warp reduce to find globally best move
            for (int mask = 16; mask > 0; mask >>= 1) {
                int od = __shfl_xor_sync(0xffffffff, my_d, mask);
                int oi = __shfl_xor_sync(0xffffffff, my_i, mask);
                int oj = __shfl_xor_sync(0xffffffff, my_j, mask);
                if (od < my_d) { my_d=od; my_i=oi; my_j=oj; }
            }

            if (tid == 0 && my_d < 0) {
                // Reverse elite_tour[my_i+1 .. my_j]
                int lo = my_i+1, hi = my_j;
                while (lo < hi) {
                    int tmp = elite_tour[lo]; elite_tour[lo]=elite_tour[hi];
                    elite_tour[hi]=tmp; ++lo; --hi;
                }
            }
            __syncthreads();
        }
#ifdef PHASE_TIMING
        PHASE_STOP(4, phase_buf, island * MAX_PHASES);
#endif
#endif  // V5_TWOOPT

        // Swap buffers
        { int* t = current; current = next; next = t; }
        __syncthreads();
    }

    // ── Final evaluation + write best tour to global memory ─────────────────
#ifdef V4_GLOBAL_DIST
    lengths[tid] = tour_length(dist, current + tid * stride, n);
#else
    lengths[tid] = tour_length(current + tid * stride, n);
#endif
    __syncthreads();

    if (tid == 0) {
        int best_idx = 0;
        for (int i = 1; i < BLOCK_POP_SIZE; ++i)
            if (lengths[i] < lengths[best_idx]) best_idx = i;

        best_lengths[island] = lengths[best_idx];
        for (int k = 0; k < n; ++k)
            best_tours[island * n + k] = current[best_idx * stride + k];
    }
}

// =============================================================================
// HOST ORCHESTRATION
// =============================================================================
static int cpu_tour_length(const std::vector<int>& dist,
                            const std::vector<int>& tour, int n) {
    int t = 0;
    for (int k = 0; k < n; ++k)
        t += dist[tour[k]*n + tour[(k+1)%n]];
    return t;
}

struct GaConfig {
    int   islands       = 128;
    int   generations   = 1000;
    float mutation_rate = 0.05f;
    int   elite_count   = 2;
    unsigned seed       = 0;
};

struct TourResult { std::vector<int> tour; int length = INT_MAX; };

static TourResult run(const TspMatrixInstance& inst, const GaConfig& cfg) {
    const int n = inst.dimension;
    if (n < 2 || n > MAX_CITIES)
        throw std::runtime_error("dimension out of range");

    unsigned seed = cfg.seed ? cfg.seed
        : (unsigned)std::chrono::high_resolution_clock::now()
              .time_since_epoch().count();

    // ── Upload dist ──────────────────────────────────────────────────────────
#ifdef V4_GLOBAL_DIST
    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, inst.dist.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(),
                          inst.dist.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
#else
    CUDA_CHECK(cudaMemcpyToSymbol(c_dist, inst.dist.data(),
                                  sizeof(int) * inst.dist.size()));
#endif

    // ── Output buffers ───────────────────────────────────────────────────────
    int* d_best_tours   = nullptr;
    int* d_best_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_tours,   (size_t)cfg.islands * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, (size_t)cfg.islands     * sizeof(int)));

    // ── Phase timing buffer (optional) ───────────────────────────────────────
#ifdef PHASE_TIMING
    uint64_t* d_phase_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phase_buf,
                          (size_t)cfg.islands * MAX_PHASES * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_phase_buf, 0,
                          (size_t)cfg.islands * MAX_PHASES * sizeof(uint64_t)));
#endif

    // ── Shared memory size ───────────────────────────────────────────────────
    const int stride = STRIDE(n);
    size_t smem = sizeof(int) * (
        2 * BLOCK_POP_SIZE * stride   // pop_a + pop_b
        + 2 * BLOCK_POP_SIZE          // lengths + order
#if defined(V5_TWOOPT)
        + 3                            // s_2opt[3]
#endif
    );

    // ── Occupancy report (enabled with -DPRINT_OCCUPANCY) ────────────────────
#ifdef PRINT_OCCUPANCY
    {
        auto occ = OccupancyReport::compute(ga_island_kernel,
                                            BLOCK_POP_SIZE, smem);
        occ.print("ga_island_kernel");
    }
#endif

    // ── Kernel launch with timing ─────────────────────────────────────────────
    GpuTimer timer;
    timer.start();

    // Build args array for kernel — adjust for version
    dim3 grid(cfg.islands), block(BLOCK_POP_SIZE);

#ifdef V4_GLOBAL_DIST
  #ifdef PHASE_TIMING
    ga_island_kernel<<<grid, block, smem>>>(n, cfg.generations,
        cfg.mutation_rate, cfg.elite_count, seed, d_dist,
        d_phase_buf, d_best_tours, d_best_lengths);
  #else
    ga_island_kernel<<<grid, block, smem>>>(n, cfg.generations,
        cfg.mutation_rate, cfg.elite_count, seed, d_dist,
        d_best_tours, d_best_lengths);
  #endif
#else
  #ifdef PHASE_TIMING
    ga_island_kernel<<<grid, block, smem>>>(n, cfg.generations,
        cfg.mutation_rate, cfg.elite_count, seed,
        d_phase_buf, d_best_tours, d_best_lengths);
  #else
    ga_island_kernel<<<grid, block, smem>>>(n, cfg.generations,
        cfg.mutation_rate, cfg.elite_count, seed,
        d_best_tours, d_best_lengths);
  #endif
#endif

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = timer.stop();

    // ── Phase report ─────────────────────────────────────────────────────────
#ifdef PHASE_TIMING
    const char* phase_names[MAX_PHASES] = {
        "fitness_eval", "elite_selection", "elite_copy",
        "crossover+mut", "2-opt_pass",
        "phase5", "phase6", "phase7"
    };
    int active_phases = TWOOPT_INTERVAL > 0 ? 5 : 4;
    cudaDeviceProp dp; cudaGetDeviceProperties(&dp, 0);
    auto pr = read_phase_report(d_phase_buf, cfg.islands,
                                active_phases, phase_names,
                                (float)dp.clockRate / 1e3f);
    pr.print();
    CUDA_CHECK(cudaFree(d_phase_buf));
#endif

    // ── Copy results back ─────────────────────────────────────────────────────
    std::vector<int> h_tours(  (size_t)cfg.islands * n);
    std::vector<int> h_lengths(cfg.islands);
    CUDA_CHECK(cudaMemcpy(h_tours.data(),   d_best_tours,
                          (size_t)cfg.islands * n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_lengths.data(), d_best_lengths,
                          (size_t)cfg.islands     * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_best_tours));
    CUDA_CHECK(cudaFree(d_best_lengths));
#ifdef V4_GLOBAL_DIST
    CUDA_CHECK(cudaFree(d_dist));
#endif

    // ── Find best island ──────────────────────────────────────────────────────
    int best_island = 0;
    for (int i = 1; i < cfg.islands; ++i)
        if (h_lengths[i] < h_lengths[best_island]) best_island = i;

    TourResult result;
    result.length = h_lengths[best_island];
    result.tour.assign(h_tours.begin() + best_island * n,
                       h_tours.begin() + (best_island+1) * n);

    // CPU cross-check
    int checked = cpu_tour_length(inst.dist, result.tour, n);
    if (checked != result.length)
        throw std::runtime_error("CPU cross-check mismatch");

    // ── Machine-readable output (parsed by run_profile_story.sh) ─────────────
    printf("kernel_time_ms=%.3f best_length=%d\n", kernel_ms, result.length);

    // ── Human-readable bandwidth estimate ─────────────────────────────────────
    // Total shared memory traffic (lower bound): 2 pops × 32 × n × 4 × gens
    size_t smem_traffic = 2ULL * BLOCK_POP_SIZE * n * 4 * cfg.generations * cfg.islands;
    float eff_bw = compute_eff_bw(smem_traffic, kernel_ms);
    printf("  effective_smem_bw_gbs=%.2f  (lower bound)\n", eff_bw);

    return result;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.tsp> [islands] [gens] "
                "[mut] [elite] [seed]\n", argv[0]);
        return 1;
    }

    // Print static info for this version
#if defined(V0_BASELINE)
    print_static_info("V0-baseline");
#elif defined(V1_STRIDE_PAD)
    print_static_info("V1-stride");
#elif defined(V2_BITMASK)
    print_static_info("V2-bitmask");
#elif defined(V3_WARP_REDUCTION)
    print_static_info("V3-warp-red");
#elif defined(V4_GLOBAL_DIST)
    print_static_info("V4-globdist");
#elif defined(V5_TWOOPT)
    print_static_info("V5-twoopt");
#endif

    GaConfig cfg;
    TspMatrixInstance inst = load_tsplib_matrix(argv[1]);
    if (argc > 2) cfg.islands       = std::stoi(argv[2]);
    if (argc > 3) cfg.generations   = std::stoi(argv[3]);
    if (argc > 4) cfg.mutation_rate = std::stof(argv[4]);
    if (argc > 5) cfg.elite_count   = std::stoi(argv[5]);
    if (argc > 6) cfg.seed          = (unsigned)std::stoul(argv[6]);

    try {
        auto result = run(inst, cfg);
        printf("BEST TOUR: ");
        for (int c : result.tour) printf("%d ", c);
        printf("%d\n", result.tour.front());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
