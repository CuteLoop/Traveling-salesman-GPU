// CUDA-GA-B4-smem.cu
// ============================================================
// Optimization B4 (Variant B): Distance Matrix in Shared Memory
// Cooperatively load dist into shared memory at kernel start.
// Eliminates ALL off-chip traffic for dist lookups after loading.
// ============================================================
// Cumulative fixes applied: B1 + B2 + B3-shuffle + B4-smem
//
// SHARED MEMORY CAPACITY ANALYSIS (P100, n=128):
//
//   P100 shared memory per SM: 64 KB = 65,536 bytes
//
//   Population shared memory (with B1 stride=n+1):
//     pop_a:   BLOCK_POP_SIZE × (n+1) × 4 = 32 × 129 × 4 = 16,512 bytes
//     pop_b:   32 × 129 × 4               = 16,512 bytes
//     lengths: 32 × 4                     =    128 bytes
//     order:   32 × 4                     =    128 bytes
//     Subtotal: 33,280 bytes
//
//   Distance matrix:
//     n × n × 4 = 128 × 128 × 4 = 65,536 bytes  ← CANNOT FIT (same as total smem!)
//
//   Constraint: pop_bytes + dist_bytes <= 65,536
//     (256n + 512) + 4n² ≤ 65,536
//     4n² + 256n + 512 ≤ 65,536
//     4n² + 256n ≤ 65,024
//     n² + 64n  ≤ 16,256
//     (n + 32)² ≤ 17,280
//     n ≤ floor(sqrt(17280) - 32) = floor(131.45 - 32) = 99
//
//   MAXIMUM n FOR SHARED MEMORY DIST: n = 99
//
//   For n = 99:
//     pop:  (256×99 + 512) × 1 = 25,856 bytes
//     dist: 99 × 99 × 4        = 39,204 bytes
//     total: 65,060 bytes  (<= 65,536 ✓)
//
//   For n = 100:
//     pop:  (256×100 + 512) = 26,112 bytes
//     dist: 100×100×4       = 40,000 bytes
//     total: 66,112 bytes   (> 65,536 ✗)
//
//   For n = 128 (our target):
//     Pop alone is 33,280 bytes. Dist alone is 65,536 bytes. Combined: 98,816 bytes.
//     DOES NOT FIT. B4-smem is INCOMPATIBLE with n=128.
//
// ARCHITECTURAL DECISION:
//   - Use B4-smem for n <= 99 (full dist in shared memory; fastest possible dist access)
//   - Use B4-global for n = 128 (global memory __restrict__; L2-cached scatter)
//   This file implements B4-smem and enforces n <= 99 at runtime.
//
// HOW LOADING WORKS:
//   All BLOCK_POP_SIZE threads cooperatively load dist into shared memory:
//     for (int i = tid; i < n*n; i += BLOCK_POP_SIZE)
//         s_dist[i] = d_dist[i];
//   For n=99: 9801 elements / 32 threads = ~306 loads per thread.
//   Loaded once at kernel start, reused for all generations.
//   After loading: all dist reads are ~4 cycle shared memory accesses.
//
// BANK CONFLICT ANALYSIS FOR s_dist:
//   s_dist[a * n + b], where a,b depend on tour values (per-thread, different).
//   With n=99 and stride=99: bank(a,b) = (a*99 + b) % 32.
//   99 % 32 = 3. GCD(3,32) = 1. So the access pattern is non-trivially spread.
//   Typical tours have random (a,b), so bank conflicts are random and average
//   much less than 32-way. In practice this is far better than constant memory.

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
constexpr int SMEM_DIST_MAX_N = 99;  // Maximum n for shared memory dist to fit

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

// B4-smem: tour_length reads from shared memory s_dist instead of constant/global
__device__ int tour_length_smem(const int* __restrict__ s_dist,
                                 const int* tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        total += s_dist[a * n + b];   // shared memory: ~4 cycle latency
    }
    return total;
}

__device__ void init_random_tour(int* tour, int n, unsigned int& rng) {
    for (int i = 0; i < n; ++i) tour[i] = i;
    for (int i = n - 1; i > 0; --i) {
        int j = rand_bounded(rng, i + 1);
        int tmp = tour[i]; tour[i] = tour[j]; tour[j] = tmp;
    }
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

__device__ void order_crossover_device(const int* parent_a, const int* parent_b,
                                       int* child, int n, unsigned int& rng) {
    int left  = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) { int tmp = left; left = right; right = tmp; }
    uint32_t used0 = 0u, used1 = 0u, used2 = 0u, used3 = 0u;
    for (int i = 0; i < n; ++i) child[i] = -1;
    for (int i = left; i <= right; ++i) { child[i] = parent_a[i]; MARK(parent_a[i]); }
    int out = (right + 1) % n;
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (ISSET(gene)) continue;
        child[out] = gene; MARK(gene); out = (out + 1) % n;
    }
}
#undef MARK
#undef ISSET

__device__ void mutate_swap_device(int* tour, int n, float mutation_rate,
                                   unsigned int& rng) {
    if (rand_unit(rng) > mutation_rate) return;
    int a = rand_bounded(rng, n);
    int b = rand_bounded(rng, n);
    while (b == a) b = rand_bounded(rng, n);
    int tmp = tour[a]; tour[a] = tour[b]; tour[b] = tmp;
}

__device__ void find_top2_warp(const int* lengths, int* out_e0, int* out_e1) {
    const unsigned int FULL_MASK = 0xFFFFFFFFu;
    int my_len = lengths[threadIdx.x], my_idx = threadIdx.x;
    for (int m = 16; m > 0; m >>= 1) {
        int ol = __shfl_xor_sync(FULL_MASK, my_len, m);
        int oi = __shfl_xor_sync(FULL_MASK, my_idx, m);
        if (ol < my_len || (ol == my_len && oi < my_idx)) { my_len = ol; my_idx = oi; }
    }
    int e0 = __shfl_sync(FULL_MASK, my_idx, 0);
    my_len = (threadIdx.x == e0) ? INT_MAX : lengths[threadIdx.x];
    my_idx = threadIdx.x;
    for (int m = 16; m > 0; m >>= 1) {
        int ol = __shfl_xor_sync(FULL_MASK, my_len, m);
        int oi = __shfl_xor_sync(FULL_MASK, my_idx, m);
        if (ol < my_len || (ol == my_len && oi < my_idx)) { my_len = ol; my_idx = oi; }
    }
    int e1 = __shfl_sync(FULL_MASK, my_idx, 0);
    if (threadIdx.x == 0) { *out_e0 = e0; *out_e1 = e1; }
}

__device__ int find_best_warp(const int* lengths) {
    const unsigned int FULL_MASK = 0xFFFFFFFFu;
    int my_len = lengths[threadIdx.x], my_idx = threadIdx.x;
    for (int m = 16; m > 0; m >>= 1) {
        int ol = __shfl_xor_sync(FULL_MASK, my_len, m);
        int oi = __shfl_xor_sync(FULL_MASK, my_idx, m);
        if (ol < my_len || (ol == my_len && oi < my_idx)) { my_len = ol; my_idx = oi; }
    }
    return __shfl_sync(FULL_MASK, my_idx, 0);
}

// ─── main kernel (B1 + B2 + B3-shuffle + B4-smem) ────────────────────────────
//
// Shared memory layout:
//   [pop_a: P*(n+1) ints][pop_b: P*(n+1) ints][lengths: P ints][order: P ints]
//   [s_dist: n*n ints]
//
// Total: (2*P*(n+1) + 2*P + n*n) * 4 bytes
// Valid only for n <= SMEM_DIST_MAX_N = 99.

__global__ void ga_island_kernel(int n, int generations, float mutation_rate,
                                  int elite_count, unsigned int seed,
                                  const int* __restrict__ d_dist,  // source (global memory)
                                  int* best_tours, int* best_lengths) {
    extern __shared__ int shared[];
    const int tid    = threadIdx.x;
    const int island = blockIdx.x;
    const int stride = n + 1;   // B1

    // Shared memory layout
    int* pop_a   = shared;
    int* pop_b   = pop_a  + BLOCK_POP_SIZE * stride;
    int* lengths = pop_b  + BLOCK_POP_SIZE * stride;
    int* order   = lengths + BLOCK_POP_SIZE;
    int* s_dist  = order   + BLOCK_POP_SIZE;   // ← B4-smem: dist in shared memory

    // ── B4-smem: cooperative load of dist matrix ─────────────────────────
    // All 32 threads load 9801/32 ≈ 306 elements each (for n=99).
    // This loop runs ONCE at kernel startup and amortizes over all generations.
    for (int i = tid; i < n * n; i += BLOCK_POP_SIZE) {
        s_dist[i] = d_dist[i];
    }
    __syncthreads();   // ensure s_dist is fully loaded before any dist lookup
    // ──────────────────────────────────────────────────────────────────────

    unsigned int rng = seed
        ^ (static_cast<unsigned int>(island + 1) * 747796405u)
        ^ (static_cast<unsigned int>(tid    + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    if (tid < BLOCK_POP_SIZE)
        init_random_tour(pop_a + tid * stride, n, rng);
    __syncthreads();

    int* current = pop_a;
    int* next    = pop_b;

    for (int generation = 0; generation < generations; ++generation) {
        if (tid < BLOCK_POP_SIZE)
            lengths[tid] = tour_length_smem(s_dist, current + tid * stride, n);  // B4-smem
        __syncthreads();

        find_top2_warp(lengths, &order[0], &order[1]);
        __syncthreads();

        if (tid < elite_count) {
            const int ei = order[tid];
            for (int k = 0; k < n; ++k)
                next[tid * stride + k] = current[ei * stride + k];
        } else if (tid < BLOCK_POP_SIZE) {
            const int pa = tournament_select_device(lengths, rng);
            const int pb = tournament_select_device(lengths, rng);
            order_crossover_device(current + pa * stride, current + pb * stride,
                                   next    + tid * stride, n, rng);
            mutate_swap_device(next + tid * stride, n, mutation_rate, rng);
        }
        __syncthreads();
        int* tmp = current; current = next; next = tmp;
        __syncthreads();
    }

    if (tid < BLOCK_POP_SIZE)
        lengths[tid] = tour_length_smem(s_dist, current + tid * stride, n);
    __syncthreads();

    int best_idx = find_best_warp(lengths);
    if (tid == 0) best_lengths[island] = lengths[best_idx];
    for (int k = tid; k < n; k += BLOCK_POP_SIZE)
        best_tours[island * n + k] = current[best_idx * stride + k];
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
    if (cfg.elite_count < 1 || cfg.elite_count >= BLOCK_POP_SIZE)
        throw std::runtime_error("elite_count in [1, BLOCK_POP_SIZE)");
    return cfg;
}

static TourResult run_gpu_population_ga(const TspMatrixInstance& inst,
                                        const GaConfig& cfg) {
    const int n = inst.dimension;
    if (n < 2)                  throw std::runtime_error("dimension must be >= 2");
    if (n > MAX_CITIES)         throw std::runtime_error("dimension exceeds MAX_CITIES");
    if (n > SMEM_DIST_MAX_N) {
        throw std::runtime_error(
            "B4-smem requires n <= " + std::to_string(SMEM_DIST_MAX_N) +
            " (shared memory capacity). Use B4-global for n=" + std::to_string(n));
    }

    const unsigned int seed = cfg.seed == 0
        ? static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count())
        : cfg.seed;

    // Print shared memory breakdown for this n
    size_t pop_bytes  = sizeof(int) * (2 * BLOCK_POP_SIZE * (n + 1) + 2 * BLOCK_POP_SIZE);
    size_t dist_bytes = sizeof(int) * n * n;
    std::cout << "Shared memory breakdown for n=" << n << ":\n"
              << "  population: " << pop_bytes  << " bytes\n"
              << "  s_dist:     " << dist_bytes << " bytes\n"
              << "  total:      " << (pop_bytes + dist_bytes) << " bytes"
              << " (limit: 65536)\n";

    // Upload dist to global memory (kernel will load to smem)
    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(int) * inst.dist.size()));
    CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(),
                          sizeof(int) * inst.dist.size(), cudaMemcpyHostToDevice));

    int* d_best_tours   = nullptr;
    int* d_best_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_tours,
                          sizeof(int) * static_cast<size_t>(cfg.islands) * n));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, sizeof(int) * cfg.islands));

    // Shared memory: pop arrays + dist matrix
    const size_t shared_bytes =
        sizeof(int) * (2 * BLOCK_POP_SIZE * (n + 1) +
                       2 * BLOCK_POP_SIZE +
                       n * n);                   // ← includes s_dist

    // Report occupancy
    int max_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, ga_island_kernel, BLOCK_POP_SIZE, shared_bytes);
    std::cout << "Occupancy: " << max_active_blocks << " blocks/SM"
              << " (" << (100.0f * max_active_blocks * BLOCK_POP_SIZE / 32 / 64)
              << "%)\n";

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
        n, cfg.generations, cfg.mutation_rate, cfg.elite_count, seed,
        d_dist, d_best_tours, d_best_lengths);
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
        std::cerr << "NOTE: B4-smem requires n <= " << SMEM_DIST_MAX_N
                  << ". For n=128, use B4-global.\n";
        return 1;
    }
    try {
        GaConfig cfg = parse_config(argc, argv);
        TspMatrixInstance inst = load_tsplib_matrix(argv[1]);
        std::cout << "VERSION: GPU-Pop B1+B2+B3-shuffle+B4-smem\n";
        std::cout << "NAME: " << inst.name << "  DIMENSION: " << inst.dimension << "\n";
        std::cout << "Islands: " << cfg.islands << "  Generations: " << cfg.generations << "\n";
        TourResult best = run_gpu_population_ga(inst, cfg);
        std::cout << "\nBest tour length: " << best.length << "\n";
        std::cout << "Best tour (0-based indices):\n";
        for (int city : best.tour) {
            std::cout << city << " ";
        }
        std::cout << best.tour.front() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
