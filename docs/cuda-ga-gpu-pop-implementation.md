# CUDA GPU-Population GA Implementation Documentation

This document explains the implementation in [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu), including:

- File purpose and architecture
- Function-by-function behavior
- Inputs and outputs
- Relevant code snippets
- Full source listing for LLM context ingestion

## 1. Scope and Purpose

The file [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu) moves most GA work onto the GPU using an island model:

1. One CUDA block represents one independent GA island.
2. Each block evolves a fixed-size population in shared memory.
3. Distance matrix is stored in constant memory.
4. After all generations, each island writes only its best tour and length to global memory.
5. CPU selects the best island result and validates it.

Compared to [CUDA-GA.cu](CUDA-GA.cu), this version greatly reduces host-device synchronization per generation.

## 2. Files Involved

- [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu)
  - GPU-resident island GA implementation and executable entry point.
- [tsplib_parser.h](tsplib_parser.h)
  - Declares TspMatrixInstance and load_tsplib_matrix.
- [tsplib_parser.cpp](tsplib_parser.cpp)
  - TSPLIB parser implementation.

### Parser Contract

From [tsplib_parser.h](tsplib_parser.h):

    struct TspMatrixInstance {
        std::string name;
        std::string type;               // TSP, ATSP, etc.
        int dimension = 0;
        std::vector<int> dist;          // row-major N x N
    };

    TspMatrixInstance load_tsplib_matrix(const std::string& path);

## 3. Constants and Memory Model

Defined in [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu):

- MAX_CITIES = 128
- BLOCK_POP_SIZE = 32
- TOURNAMENT_SIZE = 3

Global constant memory:

- c_dist[MAX_CITIES * MAX_CITIES]
  - Whole edge-weight matrix copied once via cudaMemcpyToSymbol

Per-block shared memory layout in ga_island_kernel:

- pop_a: BLOCK_POP_SIZE x n ints
- pop_b: BLOCK_POP_SIZE x n ints
- lengths: BLOCK_POP_SIZE ints
- order: BLOCK_POP_SIZE ints

This layout supports in-kernel evolution without repeated global memory transfers.

## 4. High-Level Program Flow

In [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu):

1. Parse CLI config.
2. Load TSPLIB matrix.
3. Validate n range: 2 <= n <= MAX_CITIES.
4. Copy matrix into constant memory c_dist.
5. Allocate output arrays for one best tour and one best length per island.
6. Launch ga_island_kernel with:
   - grid = islands blocks
   - block = BLOCK_POP_SIZE threads
   - dynamic shared memory sized for two populations + metadata
7. Copy island-best arrays back to host.
8. Pick globally best island on CPU.
9. CPU cross-check best tour length.
10. Print results.

## 5. Inputs and Outputs

## 5.1 Executable Input

Command line:

    CUDA-GA-GPU-Pop <file.tsp> [islands=128] [generations=1000] [mutation_rate=0.05] [elite_count=2] [seed=auto]

Arguments:

- file.tsp: required TSPLIB matrix file
- islands: optional int, must be at least 1
- generations: optional int, must be at least 1
- mutation_rate: optional float in [0, 1]
- elite_count: optional int in [1, BLOCK_POP_SIZE)
- seed: optional unsigned int, 0 means auto-seed

Additional hard limit:

- n must not exceed MAX_CITIES (128)

## 5.2 Runtime Inputs

- inst.dimension (n)
- inst.dist (row-major n x n)
- kernel config: islands, generations, mutation_rate, elite_count, seed

## 5.3 Runtime Outputs

- Console metadata and config
- Best tour length found across all islands
- Best tour sequence (0-based city ids with return-to-start print)

No output files are created.

## 6. Function-by-Function Documentation

All functions are in [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu).

## 6.1 CUDA_CHECK macro

Purpose:
- Wrap CUDA runtime API calls and throw std::runtime_error on failure.

## 6.2 xorshift32

Signature:

    __device__ unsigned int xorshift32(unsigned int& state)

Purpose:
- Lightweight per-thread PRNG core for device-side randomness.

Input:
- mutable RNG state reference

Output:
- next pseudo-random unsigned int

## 6.3 rand_bounded

Signature:

    __device__ int rand_bounded(unsigned int& state, int bound)

Purpose:
- Uniform-ish integer in [0, bound).

Input:
- RNG state
- bound

Output:
- bounded random integer

## 6.4 rand_unit

Signature:

    __device__ float rand_unit(unsigned int& state)

Purpose:
- Random float in approximately [0, 1].

Output:
- float for mutation probability checks

## 6.5 tour_length_const

Signature:

    __device__ int tour_length_const(const int* tour, int n)

Purpose:
- Compute tour length using c_dist constant memory matrix.

Input:
- tour pointer
- n

Output:
- cycle length

## 6.6 init_random_tour

Signature:

    __device__ void init_random_tour(int* tour, int n, unsigned int& rng)

Purpose:
- Initialize one tour as [0..n-1], then apply Fisher-Yates shuffle on device.

Output:
- randomized permutation written in-place

## 6.7 tournament_select_device

Signature:

    __device__ int tournament_select_device(const int* lengths,
                                            unsigned int& rng)

Purpose:
- Device-side tournament selection over BLOCK_POP_SIZE individuals.

Output:
- selected parent index

## 6.8 order_crossover_device

Signature:

    __device__ void order_crossover_device(const int* parent_a,
                                           const int* parent_b,
                                           int* child,
                                           int n,
                                           unsigned int& rng)

Purpose:
- Device-side OX crossover.

Notes:
- Uses local array used[MAX_CITIES] to track genes.
- Requires n <= MAX_CITIES.

Output:
- child tour written in-place

## 6.9 mutate_swap_device

Signature:

    __device__ void mutate_swap_device(int* tour,
                                       int n,
                                       float mutation_rate,
                                       unsigned int& rng)

Purpose:
- With probability mutation_rate, swap two random cities.

Output:
- in-place mutation or unchanged tour

## 6.10 ga_island_kernel

Signature:

    __global__ void ga_island_kernel(int n,
                                     int generations,
                                     float mutation_rate,
                                     int elite_count,
                                     unsigned int seed,
                                     int* best_tours,
                                     int* best_lengths)

Purpose:
- Full in-block island evolution kernel.

Thread/block mapping:
- blockIdx.x = island id
- threadIdx.x = individual id inside island (0..BLOCK_POP_SIZE-1)

Kernel phases:

1. Setup shared-memory pointers and RNG seed per thread.
2. Initialize population in shared memory.
3. For each generation:
   - evaluate lengths
   - thread 0 builds sorted order array
   - elites copied directly
   - remaining threads create offspring via tournament + OX + mutation
   - swap current/next population pointers
4. Re-evaluate final population.
5. Thread 0 writes best individual of this island to global outputs.

Outputs:
- best_lengths[island]
- best_tours[island * n : island * n + n)

## 6.11 cpu_tour_length

Signature:

    static int cpu_tour_length(const std::vector<int>& dist,
                               const std::vector<int>& tour,
                               int n)

Purpose:
- Host-side validation of final best tour.

## 6.12 parse_config

Signature:

    static GaConfig parse_config(int argc, char* argv[])

Purpose:
- Parse CLI arguments and validate ranges.

Validation:
- islands >= 1
- generations >= 1
- mutation_rate in [0, 1]
- elite_count in [1, BLOCK_POP_SIZE)

## 6.13 run_gpu_population_ga

Signature:

    static TourResult run_gpu_population_ga(const TspMatrixInstance& inst,
                                            const GaConfig& cfg)

Purpose:
- Host orchestration wrapper around the island kernel.

Actions:
- Validate dimensions
- Copy matrix to constant memory
- Allocate device output arrays
- Launch kernel once
- Copy island-best results back
- Select best island
- CPU cross-check

Output:
- final best TourResult

## 6.14 main

Signature:

    int main(int argc, char* argv[])

Purpose:
- Entry point for CLI execution.

Output:
- return 0 on success, 1 on errors
- prints config + final best result

## 7. Relevant Kernel Snippets

RNG seeding per thread:

    unsigned int rng = seed ^
                       (static_cast<unsigned int>(island + 1) * 747796405u) ^
                       (static_cast<unsigned int>(tid + 1) * 2891336453u);
    if (rng == 0) rng = 1;

Shared memory partitioning:

    int* pop_a = shared;
    int* pop_b = pop_a + BLOCK_POP_SIZE * n;
    int* lengths = pop_b + BLOCK_POP_SIZE * n;
    int* order = lengths + BLOCK_POP_SIZE;

Kernel launch sizing:

    ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
        n,
        cfg.generations,
        cfg.mutation_rate,
        cfg.elite_count,
        seed,
        d_best_tours,
        d_best_lengths);

## 8. Design Characteristics and Limitations

- Strengths:
  - Evolution happens inside GPU blocks with minimal host interaction.
  - Constant memory can improve repeated distance lookups.
  - Shared-memory population buffers reduce global-memory traffic during evolution.

- Limitations:
  - Hard city cap at MAX_CITIES = 128.
  - Fixed per-island population = BLOCK_POP_SIZE = 32.
  - In-block sorting is simple O(P^2) selection sort by thread 0.
  - No migration between islands.
  - Constant-memory matrix may not scale for larger n.

## 9. LLM Context Guidance

For another LLM, include:

1. This document
2. [CUDA-GA-GPU-Pop.cu](CUDA-GA-GPU-Pop.cu) full source (included below)
3. [tsplib_parser.h](tsplib_parser.h) for parser and matrix contract

Good prompt seed:

    Preserve island-model semantics and CPU cross-check, then propose improvements such as island migration, warp-aware sorting, and larger-instance support beyond MAX_CITIES.

## 10. Full Source: CUDA-GA-GPU-Pop.cu

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
    constexpr int MAX_CITIES = 128;
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

        for (int generation = 0; generation < generations; ++generation) {
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
            }
            __syncthreads();

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
        }

        if (tid < BLOCK_POP_SIZE) {
            lengths[tid] = tour_length_const(current + tid * n, n);
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

        const size_t shared_ints =
            2 * static_cast<size_t>(BLOCK_POP_SIZE) * n +
            2 * static_cast<size_t>(BLOCK_POP_SIZE);
        const size_t shared_bytes = sizeof(int) * shared_ints;

        ga_island_kernel<<<cfg.islands, BLOCK_POP_SIZE, shared_bytes>>>(
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
