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

// P100-friendly first pass:
// - one CUDA block is one isolated GA island
// - each island stores two populations in shared memory
// - the full edge-weight matrix lives in constant memory
// Variant:
// - explicit array-of-structs population layout for layout comparison
constexpr int MAX_CITIES = 128;
constexpr int BLOCK_POP_SIZE = 32;
constexpr int TOURNAMENT_SIZE = 3;

__constant__ int c_dist[MAX_CITIES * MAX_CITIES];

struct TourStruct {
    int cities[MAX_CITIES];
};

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
    int islands = 128;
    int generations = 1000;
    float mutation_rate = 0.05f;
    int elite_count = 2;
    unsigned int seed = 0;
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
    for (int i = 0; i < n; ++i) {
        tour[i] = i;
    }

    for (int i = n - 1; i > 0; --i) {
        int j = rand_bounded(rng, i + 1);
        int tmp = tour[i];
        tour[i] = tour[j];
        tour[j] = tmp;
    }
}

__device__ int tournament_select_device(const int* lengths,
                                        unsigned int& rng) {
    int best = rand_bounded(rng, BLOCK_POP_SIZE);

    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        int candidate = rand_bounded(rng, BLOCK_POP_SIZE);
        if (lengths[candidate] < lengths[best]) {
            best = candidate;
        }
    }

    return best;
}

__device__ void order_crossover_device(const int* parent_a,
                                       const int* parent_b,
                                       int* child,
                                       int n,
                                       unsigned int& rng) {
    int left = rand_bounded(rng, n);
    int right = rand_bounded(rng, n);
    if (left > right) {
        int tmp = left;
        left = right;
        right = tmp;
    }

    int used[MAX_CITIES];
    for (int i = 0; i < n; ++i) {
        child[i] = -1;
        used[i] = 0;
    }

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

__device__ void mutate_swap_device(int* tour,
                                   int n,
                                   float mutation_rate,
                                   unsigned int& rng) {
    if (rand_unit(rng) > mutation_rate) return;

    int a = rand_bounded(rng, n);
    int b = rand_bounded(rng, n);
    while (b == a) {
        b = rand_bounded(rng, n);
    }

    int tmp = tour[a];
    tour[a] = tour[b];
    tour[b] = tmp;
}

__global__ void ga_island_kernel(int n,
                                 int generations,
                                 float mutation_rate,
                                 int elite_count,
                                 unsigned int seed,
                                 int* best_tours,
                                 int* best_lengths) {
    __shared__ TourStruct pop_a[BLOCK_POP_SIZE];
    __shared__ TourStruct pop_b[BLOCK_POP_SIZE];
    __shared__ int lengths[BLOCK_POP_SIZE];
    __shared__ int order[BLOCK_POP_SIZE];

    const int tid = threadIdx.x;
    const int island = blockIdx.x;

    unsigned int rng = seed ^
                       (static_cast<unsigned int>(island + 1) * 747796405u) ^
                       (static_cast<unsigned int>(tid + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    if (tid < BLOCK_POP_SIZE) {
        init_random_tour(pop_a[tid].cities, n, rng);
    }
    __syncthreads();

    TourStruct* current = pop_a;
    TourStruct* next = pop_b;

    for (int generation = 0; generation < generations; ++generation) {
        if (tid < BLOCK_POP_SIZE) {
            lengths[tid] = tour_length_const(current[tid].cities, n);
        }
        __syncthreads();

        if (tid == 0) {
            for (int i = 0; i < BLOCK_POP_SIZE; ++i) {
                order[i] = i;
            }

            for (int i = 0; i < BLOCK_POP_SIZE - 1; ++i) {
                int best = i;
                for (int j = i + 1; j < BLOCK_POP_SIZE; ++j) {
                    if (lengths[order[j]] < lengths[order[best]]) {
                        best = j;
                    }
                }

                int tmp = order[i];
                order[i] = order[best];
                order[best] = tmp;
            }
        }
        __syncthreads();

        if (tid < elite_count) {
            const int elite_idx = order[tid];
            for (int k = 0; k < n; ++k) {
                next[tid].cities[k] = current[elite_idx].cities[k];
            }
        } else if (tid < BLOCK_POP_SIZE) {
            const int parent_a_idx = tournament_select_device(lengths, rng);
            const int parent_b_idx = tournament_select_device(lengths, rng);

            const int* parent_a = current[parent_a_idx].cities;
            const int* parent_b = current[parent_b_idx].cities;
            int* child = next[tid].cities;

            order_crossover_device(parent_a, parent_b, child, n, rng);
            mutate_swap_device(child, n, mutation_rate, rng);
        }
        __syncthreads();

        TourStruct* tmp = current;
        current = next;
        next = tmp;
        __syncthreads();
    }

    if (tid < BLOCK_POP_SIZE) {
        lengths[tid] = tour_length_const(current[tid].cities, n);
    }
    __syncthreads();

    if (tid == 0) {
        int best_idx = 0;
        for (int i = 1; i < BLOCK_POP_SIZE; ++i) {
            if (lengths[i] < lengths[best_idx]) {
                best_idx = i;
            }
        }

        best_lengths[island] = lengths[best_idx];
        for (int k = 0; k < n; ++k) {
            best_tours[island * n + k] = current[best_idx].cities[k];
        }
    }
}

static int cpu_tour_length(const std::vector<int>& dist,
                           const std::vector<int>& tour,
                           int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        total += dist[tour[k] * n + tour[(k + 1) % n]];
    }
    return total;
}

static GaConfig parse_config(int argc, char* argv[]) {
    GaConfig cfg;
    if (argc > 2) cfg.islands = std::stoi(argv[2]);
    if (argc > 3) cfg.generations = std::stoi(argv[3]);
    if (argc > 4) cfg.mutation_rate = std::stof(argv[4]);
    if (argc > 5) cfg.elite_count = std::stoi(argv[5]);
    if (argc > 6) cfg.seed = static_cast<unsigned int>(std::stoul(argv[6]));

    if (cfg.islands < 1) {
        throw std::runtime_error("islands must be at least 1");
    }
    if (cfg.generations < 1) {
        throw std::runtime_error("generations must be at least 1");
    }
    if (cfg.mutation_rate < 0.0f || cfg.mutation_rate > 1.0f) {
        throw std::runtime_error("mutation_rate must be between 0 and 1");
    }
    if (cfg.elite_count < 1 || cfg.elite_count >= BLOCK_POP_SIZE) {
        throw std::runtime_error("elite_count must be in [1, BLOCK_POP_SIZE)");
    }

    return cfg;
}

static TourResult run_gpu_population_ga(const TspMatrixInstance& inst,
                                        const GaConfig& cfg) {
    const int n = inst.dimension;
    if (n < 2) {
        throw std::runtime_error("TSP dimension must be at least 2");
    }
    if (n > MAX_CITIES) {
        throw std::runtime_error("TSP dimension exceeds MAX_CITIES for constant memory");
    }

    const unsigned int seed = cfg.seed == 0
        ? static_cast<unsigned int>(
              std::chrono::high_resolution_clock::now().time_since_epoch().count())
        : cfg.seed;

    CUDA_CHECK(cudaMemcpyToSymbol(c_dist,
                                  inst.dist.data(),
                                  sizeof(int) * inst.dist.size()));

    int* d_best_tours = nullptr;
    int* d_best_lengths = nullptr;

    const size_t best_tours_bytes =
        sizeof(int) * static_cast<size_t>(cfg.islands) * n;
    const size_t best_lengths_bytes = sizeof(int) * cfg.islands;

    CUDA_CHECK(cudaMalloc(&d_best_tours, best_tours_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, best_lengths_bytes));

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE>>>(
        n,
        cfg.generations,
        cfg.mutation_rate,
        cfg.elite_count,
        seed,
        d_best_tours,
        d_best_lengths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_best_tours(static_cast<size_t>(cfg.islands) * n);
    std::vector<int> h_best_lengths(cfg.islands);

    CUDA_CHECK(cudaMemcpy(h_best_tours.data(),
                          d_best_tours,
                          best_tours_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_lengths.data(),
                          d_best_lengths,
                          best_lengths_bytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_best_tours));
    CUDA_CHECK(cudaFree(d_best_lengths));

    int best_island = 0;
    for (int island = 1; island < cfg.islands; ++island) {
        if (h_best_lengths[island] < h_best_lengths[best_island]) {
            best_island = island;
        }
    }

    TourResult best;
    best.length = h_best_lengths[best_island];
    best.tour.assign(h_best_tours.begin() + best_island * n,
                     h_best_tours.begin() + (best_island + 1) * n);

    const int checked_length = cpu_tour_length(inst.dist, best.tour, n);
    if (checked_length != best.length) {
        throw std::runtime_error("CPU cross-check did not match GPU best length");
    }

    return best;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <file.tsp> [islands=128] [generations=1000]"
                  << " [mutation_rate=0.05] [elite_count=2] [seed=auto]\n";
        std::cerr << "Limits: MAX_CITIES=" << MAX_CITIES
                  << ", BLOCK_POP_SIZE=" << BLOCK_POP_SIZE << "\n";
        return 1;
    }

    try {
        GaConfig cfg = parse_config(argc, argv);
        TspMatrixInstance inst = load_tsplib_matrix(argv[1]);

        std::cout << "VERSION: GA-GPU-POP-AoS\n";
        std::cout << "NAME: " << inst.name << "\n";
        std::cout << "TYPE: " << inst.type << "\n";
        std::cout << "DIMENSION: " << inst.dimension << "\n";
        std::cout << "Islands: " << cfg.islands << "\n";
        std::cout << "Island population: " << BLOCK_POP_SIZE << "\n";
        std::cout << "Total GPU population: " << cfg.islands * BLOCK_POP_SIZE << "\n";
        std::cout << "Generations: " << cfg.generations << "\n";
        std::cout << "Mutation rate: " << cfg.mutation_rate << "\n";
        std::cout << "Elite count per island: " << cfg.elite_count << "\n";

        TourResult best = run_gpu_population_ga(inst, cfg);

        std::cout << "\nBest GPU-population GA tour length: " << best.length << "\n";
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
