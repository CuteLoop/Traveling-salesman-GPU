// CUDA-GA-B4-global.cu
// ============================================================
// Optimization B4 (Variant A): Distance Matrix in Global Memory
// Replace __constant__ int c_dist[] with const int* __restrict__ d_dist
// ============================================================
// Cumulative fixes applied: B1 + B2 + B3-shuffle + B4-global
//
// THE PROBLEM (B4 — constant memory scatter):
//
//   __constant__ int c_dist[MAX_CITIES * MAX_CITIES];   // 64 KB
//   total += c_dist[a * n + b];   // scatter: a,b differ per thread
//
//   Constant memory has an 8 KB/SM broadcast cache.
//   BROADCAST: if all 32 warp threads read the SAME address -> 1 fetch serves all.
//   SCATTER: if threads read DIFFERENT addresses -> accesses are SERIALIZED.
//
//   In tour_length_const(), each thread follows a different tour. At step k,
//   thread 0 may read c_dist[3*128+17] while thread 1 reads c_dist[8*128+52].
//   This is maximum scatter: the hardware serializes 32 requests into 32
//   sequential fetches from the constant cache. If not cached: 32 trips to DRAM.
//
// THE FIX (B4-global):
//
//   Switch dist to a global memory pointer with __restrict__.
//   __restrict__ tells the compiler the pointer is not aliased (no other
//   pointer in the kernel reads the same memory), enabling:
//     - Prefetching and out-of-order loads
//     - The compiler may use texture/LDG instructions (cache in L1/L2)
//
//   Global memory with scatter:
//     Requests are issued in parallel (multiple in-flight).
//     L2 hit latency: ~30–80 cycles (parallel, multiple ports).
//     L2 is 4 MB on P100; dist matrix = 128×128×4 = 64 KB = 1.6% of L2.
//     After ~10 tours warm L2, all subsequent dist reads are L2 hits.
//     Multiple requests from the same warp can be in flight simultaneously.
//
//   Constant memory scatter:
//     Requests are SERIALIZED by the constant cache hardware.
//     32 threads -> 32 sequential requests -> ~320 cycles wasted.
//     Even if cached, the serialization cannot be bypassed.
//
// NOTE: This must be BENCHMARKED, not assumed. For small N, the L2 warming
//   argument holds. For large N (much larger than 128), L2 thrashing may
//   favor constant memory. The bench_b4.slurm script measures both.
//
// EXPECTED RESULT:
//   Kernel time decreases for N=128 because scatter through global L2
//   is higher throughput than serialized constant cache requests.
//   Tour quality: identical (same RNG, same algorithm).

#include "tsplib_parser.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <climits>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int MAX_CITIES     = 128;
constexpr int BLOCK_POP_SIZE = 32;
constexpr int TOURNAMENT_SIZE = 3;

// ── B4 CHANGE: remove __constant__ c_dist. dist is now a kernel parameter. ──
// __constant__ int c_dist[MAX_CITIES * MAX_CITIES];  // REMOVED

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

struct TourResult {
    std::vector<int> tour;
    int length = std::numeric_limits<int>::max();
};

struct GaConfig {
    int islands         = 128;
    int generations     = 1000;
    float mutation_rate = 0.05f;
    int elite_count     = 2;
    unsigned int seed   = 0;
};

// ─── device utilities ────────────────────────────────────────────────────────

__device__ unsigned int xorshift32(unsigned int& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
__device__ int rand_bounded(unsigned int& state, int bound) {
    return static_cast<int>(xorshift32(state) % static_cast<unsigned int>(bound));
}
__device__ float rand_unit(unsigned int& state) {
    return static_cast<float>(xorshift32(state)) / 4294967295.0f;
}

#define STOUR(pool, idx, k) ((pool)[(idx) * stride + (k)])

// ── B4 CHANGE: tour_length now takes __restrict__ global dist pointer ──────
__device__ int tour_length_global(const int* __restrict__ dist,
                                   const int* pool, int idx, int stride, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        int a = STOUR(pool, idx, k);
        int b = STOUR(pool, idx, (k + 1) % n);
        total += dist[a * n + b];   // L2-cached scatter; requests in parallel
    }
    return total;
}

__device__ int tournament_select_device(const int* lengths, unsigned int& rng) {
    int best = rand_bounded(rng, BLOCK_POP_SIZE);
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        int candidate = rand_bounded(rng, BLOCK_POP_SIZE);
        if (lengths[candidate] < lengths[best]) best = candidate;
    }
    return best;
}

#define MARK(c)  do {                                              \
    uint32_t _bit = 1u << ((c) & 31);                             \
    if      ((c) <  32) used0 |= _bit;                            \
    else if ((c) <  64) used1 |= _bit;                            \
    else if ((c) <  96) used2 |= _bit;                            \
    else                used3 |= _bit;                            \
} while (0)
#define ISSET(c) (                                                 \
    (c) <  32 ? (used0 >> ((c)     )) & 1u :                      \
    (c) <  64 ? (used1 >> ((c) - 32)) & 1u :                      \
    (c) <  96 ? (used2 >> ((c) - 64)) & 1u :                      \
                (used3 >> ((c) - 96)) & 1u )

__device__ void order_crossover_device(const int* pool_a, int a_idx,
                                       const int* pool_b, int b_idx,
                                       int* pool_c, int c_idx,
                                       int stride, int n, unsigned int& rng) {
    int left  = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) { int tmp = left; left = right; right = tmp; }
    uint32_t used0 = 0u, used1 = 0u, used2 = 0u, used3 = 0u;
    for (int i = 0; i < n; ++i) STOUR(pool_c, c_idx, i) = -1;
    for (int i = left; i <= right; ++i) {
        STOUR(pool_c, c_idx, i) = STOUR(pool_a, a_idx, i);
        MARK(STOUR(pool_a, a_idx, i));
    }
    int out = (right + 1) % n;
    for (int offset = 1; offset <= n; ++offset) {
        int gene = STOUR(pool_b, b_idx, (right + offset) % n);
        if (ISSET(gene)) continue;
        STOUR(pool_c, c_idx, out) = gene; MARK(gene); out = (out + 1) % n;
    }
}
#undef MARK
#undef ISSET

__device__ void mutate_swap_device(int* pool, int idx, int stride, int n,
                                   float mutation_rate,
                                   unsigned int& rng) {
    if (rand_unit(rng) > mutation_rate) return;
    int a = rand_bounded(rng, n);
    int b = rand_bounded(rng, n);
    while (b == a) b = rand_bounded(rng, n);
    int tmp = STOUR(pool, idx, a); STOUR(pool, idx, a) = STOUR(pool, idx, b); STOUR(pool, idx, b) = tmp;
}

__device__ void find_top_k_reduce(const int* lengths,
                                  int* order,
                                  int* s_red,
                                  int tid,
                                  int elite_count) {
    int found[8];

    for (int pass = 0; pass < elite_count; ++pass) {
        int key = lengths[tid];
        for (int p = 0; p < pass; ++p) {
            if (tid == found[p]) {
                key = INT_MAX;
                break;
            }
        }
        s_red[tid] = key;
        order[tid] = tid;
        __syncthreads();

        for (int half = BLOCK_POP_SIZE >> 1; half > 0; half >>= 1) {
            if (tid < half) {
                bool right_wins =
                    (s_red[tid + half] < s_red[tid]) ||
                    (s_red[tid + half] == s_red[tid] &&
                     order[tid + half] < order[tid]);
                if (right_wins) {
                    s_red[tid] = s_red[tid + half];
                    order[tid] = order[tid + half];
                }
            }
            __syncthreads();
        }

        found[pass] = order[0];
        if (tid == 0) order[pass] = found[pass];
        __syncthreads();
    }
}

// ─── main kernel (B1 + B2 + B3-shuffle + B4-global) ──────────────────────────
// New parameter: const int* __restrict__ dist (global memory)

__global__ void ga_island_kernel(int n, int generations, float mutation_rate,
                                  int elite_count, unsigned int seed,
                                  const int* __restrict__ dist,   // ← B4: was __constant__
                                  int* best_tours, int* best_lengths) {
    extern __shared__ int shared[];
    const int tid    = threadIdx.x;
    const int island = blockIdx.x;
    const int stride = n + 1;   // B1

    int* pop_a   = shared;
    int* pop_b   = pop_a  + BLOCK_POP_SIZE * stride;
    int* lengths = pop_b  + BLOCK_POP_SIZE * stride;
    int* order   = lengths + BLOCK_POP_SIZE;
    int* s_red   = order   + BLOCK_POP_SIZE;

    unsigned int rng = seed
        ^ (static_cast<unsigned int>(island + 1) * 747796405u)
        ^ (static_cast<unsigned int>(tid    + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    if (tid < BLOCK_POP_SIZE) {
        for (int i = 0; i < n; ++i) STOUR(pop_a, tid, i) = i;
        for (int i = n - 1; i > 0; --i) {
            int j = rand_bounded(rng, i + 1);
            int tmp = STOUR(pop_a, tid, i);
            STOUR(pop_a, tid, i) = STOUR(pop_a, tid, j);
            STOUR(pop_a, tid, j) = tmp;
        }
    }
    __syncthreads();

    int* current = pop_a;
    int* next    = pop_b;

    for (int generation = 0; generation < generations; ++generation) {
        if (tid < BLOCK_POP_SIZE)
            lengths[tid] = tour_length_global(dist, current, tid, stride, n);
        __syncthreads();

        find_top_k_reduce(lengths, order, s_red, tid, elite_count);

        if (tid < elite_count) {
            const int ei = order[tid];
            for (int k = 0; k < n; ++k)
                STOUR(next, tid, k) = STOUR(current, ei, k);
        } else if (tid < BLOCK_POP_SIZE) {
            const int pa = tournament_select_device(lengths, rng);
            const int pb = tournament_select_device(lengths, rng);
            order_crossover_device(current, pa, current, pb,
                                   next, tid, stride, n, rng);
            mutate_swap_device(next, tid, stride, n, mutation_rate, rng);
        }
        __syncthreads();

        int* tmp = current; current = next; next = tmp;
        __syncthreads();
    }

    if (tid < BLOCK_POP_SIZE)
        lengths[tid] = tour_length_global(dist, current, tid, stride, n);
    __syncthreads();

    find_top_k_reduce(lengths, order, s_red, tid, elite_count);

    if (tid == 0) best_lengths[island] = lengths[order[0]];
    for (int k = tid; k < n; k += BLOCK_POP_SIZE)
        best_tours[island * n + k] = STOUR(current, order[0], k);
    __syncthreads();

    #undef STOUR
}

// ─── host code ───────────────────────────────────────────────────────────────

static int cpu_tour_length(const std::vector<int>& dist,
                           const std::vector<int>& tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k)
        total += dist[tour[k] * n + tour[(k + 1) % n]];
    return total;
}

static GaConfig parse_config(int argc, char* argv[]) {
    GaConfig cfg;
    if (argc > 2) cfg.islands       = std::stoi(argv[2]);
    if (argc > 3) cfg.generations   = std::stoi(argv[3]);
    if (argc > 4) cfg.mutation_rate = std::stof(argv[4]);
    if (argc > 5) cfg.elite_count   = std::stoi(argv[5]);
    if (argc > 6) cfg.seed          = static_cast<unsigned int>(std::stoul(argv[6]));
    if (cfg.islands < 1)         throw std::runtime_error("islands must be >= 1");
    if (cfg.generations < 1)     throw std::runtime_error("generations must be >= 1");
    if (cfg.mutation_rate < 0.f || cfg.mutation_rate > 1.f)
        throw std::runtime_error("mutation_rate in [0,1]");
    if (cfg.elite_count < 1 || cfg.elite_count > 8 || cfg.elite_count >= BLOCK_POP_SIZE)
        throw std::runtime_error("elite_count in [1, min(8, BLOCK_POP_SIZE))");
    return cfg;
}

static TourResult run_gpu_population_ga(const TspMatrixInstance& inst,
                                        const GaConfig& cfg) {
    const int n = inst.dimension;
    if (n < 2)          throw std::runtime_error("dimension must be >= 2");
    if (n > MAX_CITIES) throw std::runtime_error("dimension exceeds MAX_CITIES");

    const unsigned int seed = cfg.seed == 0
        ? static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count())
        : cfg.seed;

    // ── B4: allocate dist in global memory (no cudaMemcpyToSymbol) ───────
    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(int) * inst.dist.size()));
    CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(),
                          sizeof(int) * inst.dist.size(), cudaMemcpyHostToDevice));
    // ──────────────────────────────────────────────────────────────────────

    int* d_best_tours   = nullptr;
    int* d_best_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_tours,
                          sizeof(int) * static_cast<size_t>(cfg.islands) * n));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, sizeof(int) * cfg.islands));

    const int stride = n + 1;
    const size_t shared_ints =
        2 * static_cast<size_t>(BLOCK_POP_SIZE) * stride +
        3 * static_cast<size_t>(BLOCK_POP_SIZE);
    const size_t shared_bytes = sizeof(int) * shared_ints;

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
        n, cfg.generations, cfg.mutation_rate, cfg.elite_count, seed,
        d_dist,   // ← pass global dist pointer
        d_best_tours, d_best_lengths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_best_tours(static_cast<size_t>(cfg.islands) * n);
    std::vector<int> h_best_lengths(cfg.islands);
    CUDA_CHECK(cudaMemcpy(h_best_tours.data(),   d_best_tours,
                          sizeof(int) * h_best_tours.size(),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_lengths.data(), d_best_lengths,
                          sizeof(int) * h_best_lengths.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_best_tours));
    CUDA_CHECK(cudaFree(d_best_lengths));

    int best_island = 0;
    for (int i = 1; i < cfg.islands; ++i)
        if (h_best_lengths[i] < h_best_lengths[best_island]) best_island = i;

    TourResult best;
    best.length = h_best_lengths[best_island];
    best.tour.assign(h_best_tours.begin() + best_island * n,
                     h_best_tours.begin() + (best_island + 1) * n);

    const int checked = cpu_tour_length(inst.dist, best.tour, n);
    if (checked != best.length) throw std::runtime_error("CPU cross-check mismatch");

    return best;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <file.tsp> [islands=128] [generations=1000]"
                  << " [mutation_rate=0.05] [elite_count=2] [seed=auto]\n";
        return 1;
    }
    try {
        GaConfig cfg = parse_config(argc, argv);
        TspMatrixInstance inst = load_tsplib_matrix(argv[1]);
        std::cout << "VERSION: GPU-Pop B1+B2+B3-shuffle+B4-global\n";
        std::cout << "NAME: " << inst.name << "  DIMENSION: " << inst.dimension << "\n";
        std::cout << "Islands: " << cfg.islands << "  Generations: " << cfg.generations << "\n";
        TourResult best = run_gpu_population_ga(inst, cfg);
        std::cout << "\nBest tour length: " << best.length << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
