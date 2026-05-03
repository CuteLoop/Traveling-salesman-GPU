// CUDA-GA-B5-bigpop.cu
// ============================================================
// Optimization B5: 512-individual islands in global memory
// with GPU nearest-neighbour seeding and top-k elite reduction.
// ============================================================

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

constexpr int MAX_CITIES      = 128;
constexpr int BLOCK_POP_SIZE  = 512;
constexpr int TOURNAMENT_SIZE = 3;

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

__device__ int tour_length_global(const int* __restrict__ d_dist,
                                  const int* tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        total += d_dist[a * n + b];
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

__device__ void order_crossover_device(const int* parent_a,
                                       const int* parent_b,
                                       int* child,
                                       int n, unsigned int& rng) {
    int left  = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) {
        int tmp = left;
        left = right;
        right = tmp;
    }

    uint32_t used0 = 0u, used1 = 0u, used2 = 0u, used3 = 0u;
    for (int i = 0; i < n; ++i) child[i] = -1;
    for (int i = left; i <= right; ++i) {
        child[i] = parent_a[i];
        MARK(parent_a[i]);
    }
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

__device__ void mutate_swap_device(int* tour, int n,
                                   float mutation_rate, unsigned int& rng) {
    if (rand_unit(rng) > mutation_rate) return;
    int a = rand_bounded(rng, n);
    int b = rand_bounded(rng, n);
    while (b == a) b = rand_bounded(rng, n);
    int tmp = tour[a];
    tour[a] = tour[b];
    tour[b] = tmp;
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

__global__ void greedy_nn_kernel(const int* __restrict__ d_dist,
                                 int* d_nn_tours,
                                 int* d_nn_lengths,
                                 int n) {
    if (threadIdx.x != 0) return;

    const int start = blockIdx.x;
    int* tour = d_nn_tours + start * n;

    uint32_t vis0 = 0u, vis1 = 0u, vis2 = 0u, vis3 = 0u;
#define NN_MARK(c)  do { \
    if ((c) < 32) vis0 |= (1u << (c)); \
    else if ((c) < 64) vis1 |= (1u << ((c) - 32)); \
    else if ((c) < 96) vis2 |= (1u << ((c) - 64)); \
    else vis3 |= (1u << ((c) - 96)); \
} while (0)
#define NN_ISSET(c) ( \
    (c) < 32 ? (vis0 >> (c)) & 1u : \
    (c) < 64 ? (vis1 >> ((c) - 32)) & 1u : \
    (c) < 96 ? (vis2 >> ((c) - 64)) & 1u : \
               (vis3 >> ((c) - 96)) & 1u )

    int current = start;
    tour[0] = current;
    NN_MARK(current);
    int total = 0;

    for (int pos = 1; pos < n; ++pos) {
        const int* row = d_dist + current * n;
        int best_city = -1;
        int best_d = INT_MAX;
        for (int city = 0; city < n; ++city) {
            if (NN_ISSET(city)) continue;
            if (row[city] < best_d) {
                best_d = row[city];
                best_city = city;
            }
        }
        tour[pos] = best_city;
        NN_MARK(best_city);
        total += best_d;
        current = best_city;
    }
    total += d_dist[current * n + start];
    d_nn_lengths[start] = total;

#undef NN_MARK
#undef NN_ISSET
}

__global__ void ga_island_kernel(int n, int generations,
                                 float mutation_rate, int elite_count,
                                 unsigned int seed,
                                 int* g_pop_a,
                                 int* g_pop_b,
                                 const int* __restrict__ d_dist,
                                 const int* d_seed_tour,
                                 int* best_tours,
                                 int* best_lengths) {
    extern __shared__ int shared[];
    int* lengths = shared;
    int* order   = lengths + BLOCK_POP_SIZE;
    int* s_red   = order   + BLOCK_POP_SIZE;

    const int tid = threadIdx.x;
    const int island = blockIdx.x;
    const int base = island * BLOCK_POP_SIZE * n;
    int* current = g_pop_a + base;
    int* next = g_pop_b + base;

    unsigned int rng = seed
        ^ (static_cast<unsigned int>(island + 1) * 747796405u)
        ^ (static_cast<unsigned int>(tid + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    if (tid == 0) {
        for (int k = 0; k < n; ++k)
            current[k] = d_seed_tour[k];
        int n_swaps = 2 + (island % 8);
        for (int s = 0; s < n_swaps; ++s) {
            int a = rand_bounded(rng, n);
            int b = rand_bounded(rng, n);
            int tmp = current[a];
            current[a] = current[b];
            current[b] = tmp;
        }
    }
    if (tid > 0) {
        int* my_tour = current + tid * n;
        for (int i = 0; i < n; ++i) my_tour[i] = i;
        for (int i = n - 1; i > 0; --i) {
            int j = rand_bounded(rng, i + 1);
            int tmp = my_tour[i];
            my_tour[i] = my_tour[j];
            my_tour[j] = tmp;
        }
    }
    __syncthreads();

    for (int generation = 0; generation < generations; ++generation) {
        lengths[tid] = tour_length_global(d_dist, current + tid * n, n);
        __syncthreads();

        find_top_k_reduce(lengths, order, s_red, tid, elite_count);

        if (tid < elite_count) {
            const int elite_idx = order[tid];
            for (int k = 0; k < n; ++k)
                next[tid * n + k] = current[elite_idx * n + k];
        } else {
            int parent_a_idx = tournament_select_device(lengths, rng);
            int parent_b_idx = tournament_select_device(lengths, rng);
            int* parent_a = current + parent_a_idx * n;
            int* parent_b = current + parent_b_idx * n;
            int* child = next + tid * n;
            order_crossover_device(parent_a, parent_b, child, n, rng);
            mutate_swap_device(child, n, mutation_rate, rng);
        }
        __syncthreads();

        int* tmp = current;
        current = next;
        next = tmp;
        __syncthreads();
    }

    lengths[tid] = tour_length_global(d_dist, current + tid * n, n);
    __syncthreads();

    find_top_k_reduce(lengths, order, s_red, tid, elite_count);

    if (tid == 0) best_lengths[island] = lengths[order[0]];
    const int best_idx = order[0];
    for (int k = tid; k < n; k += BLOCK_POP_SIZE)
        best_tours[island * n + k] = current[best_idx * n + k];
    __syncthreads();
}

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

    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(int) * inst.dist.size()));
    CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(),
                          sizeof(int) * inst.dist.size(), cudaMemcpyHostToDevice));

    int* d_nn_tours = nullptr;
    int* d_nn_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nn_tours, sizeof(int) * n * n));
    CUDA_CHECK(cudaMalloc(&d_nn_lengths, sizeof(int) * n));

    greedy_nn_kernel<<<n, 1>>>(d_dist, d_nn_tours, d_nn_lengths, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_nn_lengths(n);
    CUDA_CHECK(cudaMemcpy(h_nn_lengths.data(), d_nn_lengths,
                          sizeof(int) * n, cudaMemcpyDeviceToHost));
    int best_start = static_cast<int>(std::min_element(h_nn_lengths.begin(),
                                                       h_nn_lengths.end())
                                      - h_nn_lengths.begin());
    int* d_seed_tour = d_nn_tours + best_start * n;

    CUDA_CHECK(cudaFree(d_nn_lengths));

    int* d_pop_a = nullptr;
    int* d_pop_b = nullptr;
    const size_t pop_bytes =
        sizeof(int) * static_cast<size_t>(cfg.islands) * BLOCK_POP_SIZE * n;
    CUDA_CHECK(cudaMalloc(&d_pop_a, pop_bytes));
    CUDA_CHECK(cudaMalloc(&d_pop_b, pop_bytes));

    int* d_best_tours = nullptr;
    int* d_best_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_tours,
                          sizeof(int) * static_cast<size_t>(cfg.islands) * n));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, sizeof(int) * cfg.islands));

    const size_t shared_bytes = sizeof(int) * 3 * BLOCK_POP_SIZE;

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
        n, cfg.generations, cfg.mutation_rate, cfg.elite_count, seed,
        d_pop_a, d_pop_b, d_dist, d_seed_tour,
        d_best_tours, d_best_lengths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_best_tours(static_cast<size_t>(cfg.islands) * n);
    std::vector<int> h_best_lengths(cfg.islands);
    CUDA_CHECK(cudaMemcpy(h_best_tours.data(), d_best_tours,
                          sizeof(int) * h_best_tours.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_lengths.data(), d_best_lengths,
                          sizeof(int) * h_best_lengths.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_nn_tours));
    CUDA_CHECK(cudaFree(d_pop_a));
    CUDA_CHECK(cudaFree(d_pop_b));
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
        std::cout << "VERSION: GPU-Pop B5-bigpop\n";
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