// CUDA-GA-B3-reduce.cu
// ============================================================
// Optimization B3 (Variant B): Classical Shared-Memory Tree Reduction
// Replaces thread-0 serialized O(P²) selection sort with a
// parallel binary tree reduction over shared memory.
// ============================================================
// Cumulative fixes applied: B1 (stride) + B2 (bitmask) + B3-reduce
//
// CONTRAST WITH B3-SHUFFLE (CUDA-GA-B3-shuffle.cu):
//   B3-shuffle:  uses __shfl_xor_sync — purely register-based, warp-synchronous,
//                no __syncthreads between reduction steps, no extra shared memory.
//   B3-reduce:   uses shared memory index array + __syncthreads per step.
//                Requires more synchronization but is architecturally more general:
//                works for any BLOCK_POP_SIZE (not just multiples of 32).
//
// THE CLASSICAL TREE REDUCTION ALGORITHM:
//
//   Goal: find the index of the minimum element in lengths[0..31].
//
//   Step 0: order[tid] = tid             (each thread "votes" for itself)
//
//   Step 1 (half=16): threads 0..15 compare:
//     order[tid] vs order[tid+16]
//     Keep the index with the smaller length.
//     __syncthreads() — wait until all comparisons are written.
//
//   Step 2 (half=8): threads 0..7 compare:
//     order[tid] vs order[tid+8]
//     ...
//
//   Step 3 (half=4), Step 4 (half=2), Step 5 (half=1):
//     After step 5: order[0] = index of global minimum.
//
//   Total: 5 reduction steps, each needing 1 __syncthreads().
//   Total comparisons: 16+8+4+2+1 = 31 (instead of 496 for selection sort).
//   All active threads (16, 8, 4, 2, 1) work in parallel.
//
//   Pass 1 finds elite0. Save elite0_idx.
//   Pass 2: reset order[], set order[elite0_idx]'s contribution to INT_MAX,
//            repeat reduction to find elite1.

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

__constant__ int c_dist[MAX_CITIES * MAX_CITIES];

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
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
__device__ int rand_bounded(unsigned int& state, int bound) {
    return static_cast<int>(xorshift32(state) % static_cast<unsigned int>(bound));
}
__device__ float rand_unit(unsigned int& state) {
    return static_cast<float>(xorshift32(state)) / 4294967295.0f;
}

__device__ int tour_length_const(const int* tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k)
        total += c_dist[tour[k] * n + tour[(k + 1) % n]];
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
        child[out] = gene;
        MARK(gene);
        out = (out + 1) % n;
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

// ─── B3-REDUCE: shared-memory binary tree reduction ──────────────────────────
//
// find_top2_reduce: all BLOCK_POP_SIZE threads call this.
//   Uses 'order' (shared memory, size BLOCK_POP_SIZE) as scratch index array.
//   Uses 'lengths' (shared memory) for comparison values.
//   Requires: BLOCK_POP_SIZE must be a power of 2.
//
// STEP-BY-STEP TRACE (BLOCK_POP_SIZE = 32):
//
//   Initial:  order = [0,1,2,...,31]
//   half=16:  threads 0..15 active
//     thread t: compare order[t] vs order[t+16]
//               if lengths[order[t+16]] < lengths[order[t]]: order[t] = order[t+16]
//     __syncthreads()
//     Active survivors: 16 candidates in order[0..15]
//
//   half=8:   threads 0..7 active
//     thread t: compare order[t] vs order[t+8]
//     __syncthreads()
//     Active survivors: 8 candidates in order[0..7]
//
//   half=4, half=2, half=1: similarly
//   After half=1: order[0] = index of global minimum
//   __syncthreads()
//
//   Total: 5 __syncthreads() calls. Comparisons: 16+8+4+2+1 = 31.
//
// This is CORRECT for any BLOCK_POP_SIZE that is a power of 2, regardless of
// warp size. It works even if BLOCK_POP_SIZE = 64 or 128 (multi-warp blocks).
//
__device__ void find_top2_reduce(const int* lengths, int* order,
                                  int* out_elite0, int* out_elite1) {
    const int tid = threadIdx.x;

    // ── Pass 1: find elite0 ───────────────────────────────────────────────
    order[tid] = tid;
    __syncthreads();

    // Binary tree reduction: halve the active range each step
    for (int half = BLOCK_POP_SIZE >> 1; half > 0; half >>= 1) {
        if (tid < half) {
            int ia = order[tid];
            int ib = order[tid + half];
            // Keep the index with smaller length; break ties by lower index
            if (lengths[ib] < lengths[ia] ||
                (lengths[ib] == lengths[ia] && ib < ia)) {
                order[tid] = ib;
            }
        }
        __syncthreads();
    }
    // order[0] now holds the index of the global minimum (elite0)
    int elite0_idx = order[0];
    __syncthreads();  // ensure all threads read elite0_idx before we overwrite order[]

    // ── Pass 2: find elite1 (exclude elite0) ─────────────────────────────
    order[tid] = tid;
    __syncthreads();

    for (int half = BLOCK_POP_SIZE >> 1; half > 0; half >>= 1) {
        if (tid < half) {
            int ia = order[tid];
            int ib = order[tid + half];
            // Treat elite0's length as INT_MAX so it cannot win
            int la = (ia == elite0_idx) ? INT_MAX : lengths[ia];
            int lb = (ib == elite0_idx) ? INT_MAX : lengths[ib];
            if (lb < la || (lb == la && ib < ia)) {
                order[tid] = ib;
            }
        }
        __syncthreads();
    }
    int elite1_idx = order[0];
    __syncthreads();

    // Store final elite indices back into order[] for the rest of the kernel
    if (tid == 0) {
        *out_elite0 = elite0_idx;
        *out_elite1 = elite1_idx;
    }
}

// Single-min classical reduction (for final output)
__device__ int find_best_reduce(const int* lengths, int* order) {
    const int tid = threadIdx.x;
    order[tid] = tid;
    __syncthreads();
    for (int half = BLOCK_POP_SIZE >> 1; half > 0; half >>= 1) {
        if (tid < half) {
            int ia = order[tid], ib = order[tid + half];
            if (lengths[ib] < lengths[ia]) order[tid] = ib;
        }
        __syncthreads();
    }
    return order[0];
}

// ─── main kernel (B1 + B2 + B3-reduce) ──────────────────────────────────────

__global__ void ga_island_kernel(int n, int generations, float mutation_rate,
                                  int elite_count, unsigned int seed,
                                  int* best_tours, int* best_lengths) {
    extern __shared__ int shared[];

    const int tid    = threadIdx.x;
    const int island = blockIdx.x;
    const int stride = n + 1;   // B1

    int* pop_a   = shared;
    int* pop_b   = pop_a  + BLOCK_POP_SIZE * stride;
    int* lengths = pop_b  + BLOCK_POP_SIZE * stride;
    int* order   = lengths + BLOCK_POP_SIZE;

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
            lengths[tid] = tour_length_const(current + tid * stride, n);
        __syncthreads();

        // ── B3-REDUCE: binary tree reduction in shared memory ─────────────
        // Threads 0..(BLOCK_POP_SIZE-1) all participate. Uses __syncthreads().
        // After the call: order[0] = elite0_idx, order[1] = elite1_idx.
        find_top2_reduce(lengths, order, &order[0], &order[1]);
        __syncthreads();  // ensure order[0..1] visible to all threads
        // ─────────────────────────────────────────────────────────────────

        if (tid < elite_count) {
            const int elite_idx = order[tid];
            for (int k = 0; k < n; ++k)
                next[tid * stride + k] = current[elite_idx * stride + k];
        } else if (tid < BLOCK_POP_SIZE) {
            const int pa = tournament_select_device(lengths, rng);
            const int pb = tournament_select_device(lengths, rng);
            order_crossover_device(current + pa * stride,
                                   current + pb * stride,
                                   next    + tid * stride,
                                   n, rng);
            mutate_swap_device(next + tid * stride, n, mutation_rate, rng);
        }
        __syncthreads();

        int* tmp = current; current = next; next = tmp;
        __syncthreads();
    }

    if (tid < BLOCK_POP_SIZE)
        lengths[tid] = tour_length_const(current + tid * stride, n);
    __syncthreads();

    // Final best: tree reduction
    int best_idx = find_best_reduce(lengths, order);
    // Cooperative output copy
    if (tid == 0)
        best_lengths[island] = lengths[best_idx];
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
        throw std::runtime_error("mutation_rate must be in [0,1]");
    if (cfg.elite_count < 1 || cfg.elite_count >= BLOCK_POP_SIZE)
        throw std::runtime_error("elite_count must be in [1, BLOCK_POP_SIZE)");
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

    CUDA_CHECK(cudaMemcpyToSymbol(c_dist, inst.dist.data(),
                                  sizeof(int) * inst.dist.size()));

    int* d_best_tours   = nullptr;
    int* d_best_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_tours,
                          sizeof(int) * static_cast<size_t>(cfg.islands) * n));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, sizeof(int) * cfg.islands));

    const size_t shared_ints =
        2 * static_cast<size_t>(BLOCK_POP_SIZE) * (n + 1) +
        2 * static_cast<size_t>(BLOCK_POP_SIZE);
    const size_t shared_bytes = sizeof(int) * shared_ints;

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
        n, cfg.generations, cfg.mutation_rate, cfg.elite_count, seed,
        d_best_tours, d_best_lengths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_best_tours(static_cast<size_t>(cfg.islands) * n);
    std::vector<int> h_best_lengths(cfg.islands);
    CUDA_CHECK(cudaMemcpy(h_best_tours.data(),   d_best_tours,
                          sizeof(int) * h_best_tours.size(),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_lengths.data(), d_best_lengths,
                          sizeof(int) * h_best_lengths.size(), cudaMemcpyDeviceToHost));
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
    if (checked != best.length)
        throw std::runtime_error("CPU cross-check mismatch");

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
        std::cout << "VERSION: GPU-Pop B1+B2+B3-reduce\n";
        std::cout << "NAME: " << inst.name << "\n";
        std::cout << "DIMENSION: " << inst.dimension << "\n";
        std::cout << "Islands: " << cfg.islands << "\n";
        std::cout << "Generations: " << cfg.generations << "\n";
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
