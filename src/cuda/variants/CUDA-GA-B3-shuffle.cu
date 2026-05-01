// CUDA-GA-B3-shuffle.cu
// ============================================================
// Optimization B3 (Variant A): Warp XOR-Shuffle Min-Reduction
// Replaces thread-0 serialized O(P²) selection sort with a
// parallel warp-shuffle tree that finds top-2 in 10 shuffle steps.
// ============================================================
// Cumulative fixes applied: B1 (stride) + B2 (bitmask) + B3-shuffle
//
// THE PROBLEM (B3):
//   Original natural selection phase:
//
//   if (tid == 0) {
//       // O(P^2) selection sort: (P-1)+(P-2)+...+1 = P(P-1)/2 = 496 comparisons
//       for (int i = 0; i < BLOCK_POP_SIZE - 1; ++i) { ... }
//   }
//   __syncthreads(); // 31 threads idle waiting here
//
//   Two problems compound:
//   (a) 31 of 32 warp lanes are idle for the entire sort duration.
//   (b) The 32-thread block has only 1 warp, so the scheduler cannot
//       hide this latency by switching to another warp — there is none.
//       Every cycle during the sort is a wasted slot.
//
//   Serial ops per full run:
//     496 comparisons × 128 islands × 1000 gens = 63,488,000 serial comparisons
//
// THE KEY INSIGHT:
//   We do NOT need a full sort. We only need elite_count=2 best indices.
//   Finding the minimum of 32 values takes 5 steps with warp shuffles (log2(32)=5).
//   Finding the top-2 takes 2 passes × 5 steps = 10 shuffle steps total.
//   All 32 lanes participate simultaneously, warp-synchronously (no __syncthreads).
//
// HOW __shfl_xor_sync WORKS:
//   __shfl_xor_sync(mask, val, laneMask):
//     Each lane i exchanges its value with lane (i XOR laneMask).
//     All lanes in mask participate simultaneously.
//     No memory involved — values travel through the warp's register file crossbar.
//     Latency: ~4 cycles (much less than a shared memory round-trip).
//
//   XOR-shuffle reduction pattern for min over 32 lanes:
//     mask=16: lane 0<->16, 1<->17, ..., 15<->31  (compare halves)
//     mask= 8: lane 0<-> 8, 1<-> 9, ...            (compare quarters)
//     mask= 4: ...
//     mask= 2: ...
//     mask= 1: lane 0<->1, 2<->3, ...
//   After 5 steps, lane 0 holds the global minimum. Then broadcast to all lanes.
//
// CORRECTNESS REQUIREMENT:
//   BLOCK_POP_SIZE must equal 32 (one warp). If BLOCK_POP_SIZE < 32, use a
//   mask covering only the active lanes. If BLOCK_POP_SIZE > 32, a two-level
//   reduction is required (warp-local -> shared memory -> warp of partials).

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

// ─── B3-SHUFFLE: parallel warp min-reduction ─────────────────────────────────
//
// find_top2_warp: all 32 threads call this simultaneously.
// No shared memory needed, no __syncthreads needed.
// Communicates entirely through warp-shuffle register exchanges.
//
// Pass 1: XOR-tree finds global minimum (best individual).
//   Each thread starts with:  my_len = lengths[tid], my_idx = tid
//   Step mask=16: thread i compares with thread (i XOR 16).
//     Both threads get each other's (len, idx) and keep the smaller.
//   After 5 steps: all threads agree on the global minimum's (len, idx).
//   We broadcast the winner's index to all lanes.
//
// Pass 2: Repeat with elite0 masked to INT_MAX.
//   Thread elite0_idx contributes INT_MAX so it cannot win again.
//   After 5 steps: all threads agree on the second-minimum's index.
//
// Result: *elite0_idx and *elite1_idx are written from thread 0.
//         Caller must __syncthreads() before reading from shared memory.
//
__device__ void find_top2_warp(const int* lengths,
                                int* out_elite0, int* out_elite1) {
    const unsigned int FULL_MASK = 0xFFFFFFFFu;

    // ── Pass 1: find elite0 ───────────────────────────────────────────────
    int my_len = lengths[threadIdx.x];
    int my_idx = threadIdx.x;

    // XOR-shuffle tree: 5 steps for 32 lanes
    for (int xor_mask = 16; xor_mask > 0; xor_mask >>= 1) {
        int other_len = __shfl_xor_sync(FULL_MASK, my_len, xor_mask);
        int other_idx = __shfl_xor_sync(FULL_MASK, my_idx, xor_mask);
        // Keep the smaller; break ties by lower index (deterministic)
        if (other_len < my_len || (other_len == my_len && other_idx < my_idx)) {
            my_len = other_len;
            my_idx = other_idx;
        }
    }
    // After the tree, all 32 lanes hold the global minimum's (len, idx).
    // Broadcast lane 0's winner index to all lanes (it's already identical,
    // but this makes the intent explicit and handles any tie-breaking).
    int elite0_idx = __shfl_sync(FULL_MASK, my_idx, 0);

    // ── Pass 2: find elite1 (exclude elite0) ─────────────────────────────
    // Mask out elite0 by setting its length to INT_MAX for this pass
    my_len = (threadIdx.x == elite0_idx) ? INT_MAX : lengths[threadIdx.x];
    my_idx = threadIdx.x;

    for (int xor_mask = 16; xor_mask > 0; xor_mask >>= 1) {
        int other_len = __shfl_xor_sync(FULL_MASK, my_len, xor_mask);
        int other_idx = __shfl_xor_sync(FULL_MASK, my_idx, xor_mask);
        if (other_len < my_len || (other_len == my_len && other_idx < my_idx)) {
            my_len = other_len;
            my_idx = other_idx;
        }
    }
    int elite1_idx = __shfl_sync(FULL_MASK, my_idx, 0);

    // Write results from thread 0 into shared memory order[] slots
    if (threadIdx.x == 0) {
        *out_elite0 = elite0_idx;
        *out_elite1 = elite1_idx;
    }
}

// Warp-shuffle reduction to find a single minimum (used in final output step)
__device__ int find_best_warp(const int* lengths) {
    const unsigned int FULL_MASK = 0xFFFFFFFFu;
    int my_len = lengths[threadIdx.x];
    int my_idx = threadIdx.x;
    for (int xor_mask = 16; xor_mask > 0; xor_mask >>= 1) {
        int o_len = __shfl_xor_sync(FULL_MASK, my_len, xor_mask);
        int o_idx = __shfl_xor_sync(FULL_MASK, my_idx, xor_mask);
        if (o_len < my_len || (o_len == my_len && o_idx < my_idx)) {
            my_len = o_len; my_idx = o_idx;
        }
    }
    return __shfl_sync(FULL_MASK, my_idx, 0);
}

// ─── main kernel (B1 + B2 + B3-shuffle) ──────────────────────────────────────

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
    int* order   = lengths + BLOCK_POP_SIZE;   // reused as elite index storage

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

        // Evaluate fitness (all threads, parallel)
        if (tid < BLOCK_POP_SIZE)
            lengths[tid] = tour_length_const(current + tid * stride, n);
        __syncthreads();

        // ── B3 FIX: warp-shuffle top-2 reduction ─────────────────────────
        // All 32 threads participate. No __syncthreads() needed between
        // shuffle steps (warp is implicitly synchronous).
        // find_top2_warp writes elite indices to order[0] and order[1].
        find_top2_warp(lengths, &order[0], &order[1]);
        __syncthreads();  // ensure order[] is visible before elite copy
        // ─────────────────────────────────────────────────────────────────

        // Elite copy + crossover (same as before — order[tid] still used)
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

    // Final evaluation
    if (tid < BLOCK_POP_SIZE)
        lengths[tid] = tour_length_const(current + tid * stride, n);
    __syncthreads();

    // Final best: warp-shuffle reduction (all threads find best_idx)
    int best_idx = find_best_warp(lengths);
    // Cooperative output: all threads write their portion of the tour
    // best_tours is global memory, stride n (packed, not n+1)
    if (tid == 0) {
        best_lengths[island] = lengths[best_idx];
    }
    // All threads cooperate on copying the tour to global memory
    for (int k = tid; k < n; k += BLOCK_POP_SIZE) {
        best_tours[island * n + k] = current[best_idx * stride + k];
    }
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
        std::cout << "VERSION: GPU-Pop B1+B2+B3-shuffle\n";
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
