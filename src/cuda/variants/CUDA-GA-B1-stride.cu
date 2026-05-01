// CUDA-GA-B1-stride.cu
// ============================================================
// Optimization B1: Shared Memory Bank Conflict Elimination
// via Stride Padding (stride = n + 1)
// ============================================================
// Cumulative fixes applied: B1
// Baseline: CUDA-GA-GPU-Pop.cu
//
// THE ONE CHANGE:
//   Before: int* pop_a = shared;
//           int* pop_b = pop_a + BLOCK_POP_SIZE * n;      // stride = n
//           access: pop_a[tid * n + k]
//
//   After:  const int stride = n + 1;
//           int* pop_a = shared;
//           int* pop_b = pop_a + BLOCK_POP_SIZE * stride; // stride = n + 1
//           access: pop_a[tid * stride + k]
//
// WHY IT WORKS:
//   P100 shared memory has 32 banks, 4 bytes each.
//   Bank of element at word-index e: bank(e) = e % 32
//   With stride = n = 128: bank(t, k) = (t*128 + k) % 32 = k % 32  <- identical for all t!
//   Every warp thread hits the same bank -> 32-way serialization.
//   With stride = n + 1 = 129: bank(t, k) = (t*129 + k) % 32 = (t + k) % 32
//   Thread 0: bank k, thread 1: bank k+1, ..., thread 31: bank k+31 -> all 32 distinct banks.
//
// EXPECTED PROFILER RESULT (nvprof):
//   shared_load_transactions_per_request:  ~32 (before) -> ~1 (after)
//   shared_store_transactions_per_request: ~32 (before) -> ~1 (after)

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

constexpr int MAX_CITIES    = 128;
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
    for (int k = 0; k < n; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        total += c_dist[a * n + b];
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

// B2 NOT yet applied: used[] still spills to local memory here.
__device__ void order_crossover_device(const int* parent_a,
                                       const int* parent_b,
                                       int* child, int n,
                                       unsigned int& rng) {
    int left  = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) { int tmp = left; left = right; right = tmp; }

    int used[MAX_CITIES];
    for (int i = 0; i < n; ++i) { child[i] = -1; used[i] = 0; }

    for (int i = left; i <= right; ++i) {
        child[i] = parent_a[i];
        used[parent_a[i]] = 1;
    }

    int out = (right + 1) % n;
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (used[gene]) continue;
        child[out] = gene;
        used[gene] = 1;
        out = (out + 1) % n;
    }
}

__device__ void mutate_swap_device(int* tour, int n, float mutation_rate,
                                   unsigned int& rng) {
    if (rand_unit(rng) > mutation_rate) return;
    int a = rand_bounded(rng, n);
    int b = rand_bounded(rng, n);
    while (b == a) b = rand_bounded(rng, n);
    int tmp = tour[a]; tour[a] = tour[b]; tour[b] = tmp;
}

// ─── main kernel ─────────────────────────────────────────────────────────────

__global__ void ga_island_kernel(int n, int generations, float mutation_rate,
                                  int elite_count, unsigned int seed,
                                  int* best_tours, int* best_lengths) {
    extern __shared__ int shared[];

    const int tid    = threadIdx.x;
    const int island = blockIdx.x;

    // ── B1 FIX: stride = n + 1 ────────────────────────────────────────────
    const int stride = n + 1;   // <-- KEY CHANGE

    int* pop_a   = shared;
    int* pop_b   = pop_a  + BLOCK_POP_SIZE * stride;  // was: * n
    int* lengths = pop_b  + BLOCK_POP_SIZE * stride;  // was: * n
    int* order   = lengths + BLOCK_POP_SIZE;
    // ──────────────────────────────────────────────────────────────────────

    unsigned int rng = seed
        ^ (static_cast<unsigned int>(island + 1) * 747796405u)
        ^ (static_cast<unsigned int>(tid    + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    if (tid < BLOCK_POP_SIZE)
        init_random_tour(pop_a + tid * stride, n, rng);  // was: tid * n
    __syncthreads();

    int* current = pop_a;
    int* next    = pop_b;

    for (int generation = 0; generation < generations; ++generation) {

        // Evaluate fitness
        if (tid < BLOCK_POP_SIZE)
            lengths[tid] = tour_length_const(current + tid * stride, n);  // was: tid * n
        __syncthreads();

        // Selection sort by thread 0 (B3 not yet fixed)
        if (tid == 0) {
            for (int i = 0; i < BLOCK_POP_SIZE; ++i) order[i] = i;
            for (int i = 0; i < BLOCK_POP_SIZE - 1; ++i) {
                int best = i;
                for (int j = i + 1; j < BLOCK_POP_SIZE; ++j)
                    if (lengths[order[j]] < lengths[order[best]]) best = j;
                int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
            }
        }
        __syncthreads();

        // Elite copy + crossover/mutation
        if (tid < elite_count) {
            const int elite_idx = order[tid];
            for (int k = 0; k < n; ++k)
                next[tid * stride + k] = current[elite_idx * stride + k];  // was: * n
        } else if (tid < BLOCK_POP_SIZE) {
            const int pa = tournament_select_device(lengths, rng);
            const int pb = tournament_select_device(lengths, rng);
            order_crossover_device(current + pa * stride,     // was: * n
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

    // Write best tour to global memory (thread 0 serially finds best + copies)
    if (tid == 0) {
        int best_idx = 0;
        for (int i = 1; i < BLOCK_POP_SIZE; ++i)
            if (lengths[i] < lengths[best_idx]) best_idx = i;

        best_lengths[island] = lengths[best_idx];
        // Output uses packed stride n, not n+1
        for (int k = 0; k < n; ++k)
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

    // ── B1 FIX: shared memory now uses stride = n + 1 ────────────────────
    const size_t shared_ints =
        2 * static_cast<size_t>(BLOCK_POP_SIZE) * (n + 1) +  // pop_a + pop_b (padded)
        2 * static_cast<size_t>(BLOCK_POP_SIZE);              // lengths + order
    const size_t shared_bytes = sizeof(int) * shared_ints;
    // ──────────────────────────────────────────────────────────────────────

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

        std::cout << "VERSION: GPU-Pop B1-stride\n";
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
