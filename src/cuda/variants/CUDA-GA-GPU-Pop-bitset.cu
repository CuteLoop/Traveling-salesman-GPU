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
// Variant 2:
// - replace per-thread used[MAX_CITIES] scratch with a compact bitset
constexpr int MAX_CITIES = 128;
constexpr int BLOCK_POP_SIZE = 32;
constexpr int TOURNAMENT_SIZE = 3;
constexpr int USED_WORDS = (MAX_CITIES + 31) / 32;

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
    int islands = 128;
    int generations = 1000;
    float mutation_rate = 0.05f;
    int elite_count = 2;
    unsigned int seed = 0;
    int target_length = -1;
};

struct GaRunResult {
    TourResult best;
    float elapsed_ms = 0.0f;
    int generations_run = 0;
    bool target_reached = false;
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

__device__ inline void bitset_clear(uint32_t* used) {
    #pragma unroll
    for (int i = 0; i < USED_WORDS; ++i) {
        used[i] = 0u;
    }
}

__device__ inline void bitset_set(uint32_t* used, int city) {
    used[city >> 5] |= (1u << (city & 31));
}

__device__ inline bool bitset_test(const uint32_t* used, int city) {
    return (used[city >> 5] & (1u << (city & 31))) != 0u;
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

    uint32_t used[USED_WORDS];
    bitset_clear(used);

    for (int i = 0; i < n; ++i) {
        child[i] = -1;
    }

    for (int i = left; i <= right; ++i) {
        child[i] = parent_a[i];
        bitset_set(used, parent_a[i]);
    }

    int out = (right + 1) % n;
    for (int offset = 1; offset <= n; ++offset) {
        int gene = parent_b[(right + offset) % n];
        if (bitset_test(used, gene)) continue;

        child[out] = gene;
        bitset_set(used, gene);
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
                                 int target_length,
                                 int* stop_flag,
                                 int* generations_run,
                                 int* best_tours,
                                 int* best_lengths) {
    extern __shared__ int shared[];

    const int tid = threadIdx.x;
    const int island = blockIdx.x;

    int* pop_a = shared;
    int* pop_b = pop_a + BLOCK_POP_SIZE * n;
    int* lengths = pop_b + BLOCK_POP_SIZE * n;
    int* order = lengths + BLOCK_POP_SIZE;

    unsigned int rng = seed ^
                       (static_cast<unsigned int>(island + 1) * 747796405u) ^
                       (static_cast<unsigned int>(tid + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    if (tid < BLOCK_POP_SIZE) {
        init_random_tour(pop_a + tid * n, n, rng);
    }
    __syncthreads();

    int* current = pop_a;
    int* next = pop_b;

    int generation = 0;

    while (true) {
        if (tid < BLOCK_POP_SIZE) {
            lengths[tid] = tour_length_const(current + tid * n, n);
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

            if (target_length >= 0 && lengths[order[0]] <= target_length) {
                atomicExch(stop_flag, 1);
            }
        }
        __syncthreads();

        if (target_length >= 0 && *stop_flag != 0) {
            break;
        }

        if (tid < elite_count) {
            const int elite_idx = order[tid];
            for (int k = 0; k < n; ++k) {
                next[tid * n + k] = current[elite_idx * n + k];
            }
        } else if (tid < BLOCK_POP_SIZE) {
            const int parent_a_idx = tournament_select_device(lengths, rng);
            const int parent_b_idx = tournament_select_device(lengths, rng);

            const int* parent_a = current + parent_a_idx * n;
            const int* parent_b = current + parent_b_idx * n;
            int* child = next + tid * n;

            order_crossover_device(parent_a, parent_b, child, n, rng);
            mutate_swap_device(child, n, mutation_rate, rng);
        }
        __syncthreads();

        int* tmp = current;
        current = next;
        next = tmp;
        __syncthreads();

        ++generation;
        if (target_length < 0 && generation >= generations) {
            break;
        }
    }

    if (tid < BLOCK_POP_SIZE) {
        lengths[tid] = tour_length_const(current + tid * n, n);
    }
    __syncthreads();

    if (tid == 0) {
        generations_run[island] = generation;

        int best_idx = 0;
        for (int i = 1; i < BLOCK_POP_SIZE; ++i) {
            if (lengths[i] < lengths[best_idx]) {
                best_idx = i;
            }
        }

        best_lengths[island] = lengths[best_idx];
        for (int k = 0; k < n; ++k) {
            best_tours[island * n + k] = current[best_idx * n + k];
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
    if (argc > 7) cfg.target_length = std::stoi(argv[7]);

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
    if (cfg.target_length != -1 && cfg.target_length <= 0) {
        throw std::runtime_error("target_length must be positive when provided");
    }

    return cfg;
}

static GaRunResult run_gpu_population_ga(const TspMatrixInstance& inst,
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
    int* d_generations_run = nullptr;
    int* d_stop_flag = nullptr;

    const size_t best_tours_bytes =
        sizeof(int) * static_cast<size_t>(cfg.islands) * n;
    const size_t best_lengths_bytes = sizeof(int) * cfg.islands;

    CUDA_CHECK(cudaMalloc(&d_best_tours, best_tours_bytes));
    CUDA_CHECK(cudaMalloc(&d_best_lengths, best_lengths_bytes));
    CUDA_CHECK(cudaMalloc(&d_generations_run, sizeof(int) * cfg.islands));
    CUDA_CHECK(cudaMalloc(&d_stop_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_generations_run, 0, sizeof(int) * cfg.islands));
    CUDA_CHECK(cudaMemset(d_stop_flag, 0, sizeof(int)));

    const size_t shared_ints =
        2 * static_cast<size_t>(BLOCK_POP_SIZE) * n +
        2 * static_cast<size_t>(BLOCK_POP_SIZE);
    const size_t shared_bytes = sizeof(int) * shared_ints;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
        n,
        cfg.generations,
        cfg.mutation_rate,
        cfg.elite_count,
        seed,
        cfg.target_length,
        d_stop_flag,
        d_generations_run,
        d_best_tours,
        d_best_lengths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

    std::vector<int> h_best_tours(static_cast<size_t>(cfg.islands) * n);
    std::vector<int> h_best_lengths(cfg.islands);
    std::vector<int> h_generations_run(cfg.islands, 0);
    int h_stop_flag = 0;

    CUDA_CHECK(cudaMemcpy(h_best_tours.data(),
                          d_best_tours,
                          best_tours_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_lengths.data(),
                          d_best_lengths,
                          best_lengths_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_generations_run.data(),
                          d_generations_run,
                          sizeof(int) * cfg.islands,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_stop_flag,
                          d_stop_flag,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_best_tours));
    CUDA_CHECK(cudaFree(d_best_lengths));
    CUDA_CHECK(cudaFree(d_generations_run));
    CUDA_CHECK(cudaFree(d_stop_flag));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    int best_island = 0;
    for (int island = 1; island < cfg.islands; ++island) {
        if (h_best_lengths[island] < h_best_lengths[best_island]) {
            best_island = island;
        }
    }

    GaRunResult result;
    result.best.length = h_best_lengths[best_island];
    result.best.tour.assign(h_best_tours.begin() + best_island * n,
                            h_best_tours.begin() + (best_island + 1) * n);
    result.elapsed_ms = elapsed_ms;
    result.target_reached = (h_stop_flag != 0);
    for (int island = 0; island < cfg.islands; ++island) {
        if (h_generations_run[island] > result.generations_run) {
            result.generations_run = h_generations_run[island];
        }
    }

    const int checked_length = cpu_tour_length(inst.dist, result.best.tour, n);
    if (checked_length != result.best.length) {
        throw std::runtime_error("CPU cross-check did not match GPU best length");
    }

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <file.tsp> [islands=128] [generations=1000]"
                  << " [mutation_rate=0.05] [elite_count=2] [seed=auto]"
                  << " [target_length=off]\n";
        std::cerr << "Limits: MAX_CITIES=" << MAX_CITIES
                  << ", BLOCK_POP_SIZE=" << BLOCK_POP_SIZE << "\n";
        return 1;
    }

    try {
        GaConfig cfg = parse_config(argc, argv);
        TspMatrixInstance inst = load_tsplib_matrix(argv[1]);

        std::cout << "VERSION: GPU-Pop Bitset Variant\n";
        std::cout << "NAME: " << inst.name << "\n";
        std::cout << "TYPE: " << inst.type << "\n";
        std::cout << "DIMENSION: " << inst.dimension << "\n";
        std::cout << "Islands: " << cfg.islands << "\n";
        std::cout << "Island population: " << BLOCK_POP_SIZE << "\n";
        std::cout << "Total GPU population: " << cfg.islands * BLOCK_POP_SIZE << "\n";
        std::cout << "Generations: " << cfg.generations << "\n";
        std::cout << "Mutation rate: " << cfg.mutation_rate << "\n";
        std::cout << "Elite count per island: " << cfg.elite_count << "\n";
        if (cfg.target_length >= 0) {
            std::cout << "Stop mode: target length\n";
            std::cout << "Target length: " << cfg.target_length << "\n";
        } else {
            std::cout << "Stop mode: fixed generations\n";
        }

        GaRunResult run = run_gpu_population_ga(inst, cfg);

        std::cout << "CUDA kernel elapsed ms: " << run.elapsed_ms << "\n";
        std::cout << "Generations completed: " << run.generations_run << "\n";
        std::cout << "Target reached: " << (run.target_reached ? "yes" : "no") << "\n";
        std::cout << "\nBest GPU-population GA tour length: " << run.best.length << "\n";
        std::cout << "Best tour (0-based indices):\n";
        for (int city : run.best.tour) {
            std::cout << city << " ";
        }
        std::cout << run.best.tour.front() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
