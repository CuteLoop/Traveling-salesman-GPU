# CUDA GA Implementation Documentation

This document explains the implementation in [CUDA-GA.cu](CUDA-GA.cu), including:

- File purpose and architecture
- Function-by-function behavior
- Inputs and outputs
- Relevant code snippets
- Full source listing for LLM context ingestion

## 1. Scope and Purpose

The file [CUDA-GA.cu](CUDA-GA.cu) implements a hybrid genetic algorithm for TSP:

1. CPU handles GA control flow (selection, crossover, mutation, elitism, generation loop).
2. GPU evaluates fitness (tour lengths) for the whole population each generation.
3. CPU sorts by fitness and creates the next generation.
4. CPU validates the final best result against a deterministic length recomputation.

This is a practical middle stage between a naive CUDA kernel demo and a fully GPU-resident GA.

## 2. Files Involved

- [CUDA-GA.cu](CUDA-GA.cu)
  - Main hybrid GA implementation and executable entry point.
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

## 3. High-Level Program Flow

In [CUDA-GA.cu](CUDA-GA.cu):

1. Parse command-line config.
2. Load TSPLIB matrix.
3. Build greedy nearest-neighbor seed tour (best start city).
4. Initialize population:
   - Individual 0 = greedy seed
   - Remaining individuals = random permutations
5. Allocate GPU buffers for matrix, population, and lengths.
6. Repeat for each generation:
   - Copy current population to GPU
   - Evaluate all tour lengths on GPU kernel
   - Copy lengths back to CPU
   - Sort indices by fitness
   - Update global best
   - Create next generation using elitism + tournament selection + OX crossover + swap mutation
7. Free GPU memory.
8. CPU cross-check of final best length.
9. Print best tour and length.

## 4. Inputs and Outputs

## 4.1 Executable Input

Command line:

    CUDA-GA <file.tsp> [population=512] [generations=1000] [mutation_rate=0.05] [elite_count=4] [seed=auto]

Arguments:

- file.tsp: required TSPLIB matrix file
- population: optional int, must be at least 2
- generations: optional int, must be at least 1
- mutation_rate: optional double in [0, 1]
- elite_count: optional int in [1, population)
- seed: optional unsigned int, 0 means auto-seed from clock

## 4.2 Core Runtime Inputs

- inst.dimension (n): number of cities
- inst.dist: row-major n x n integer matrix
- population array: flattened population_size x n tours

## 4.3 Runtime Outputs

- Console metadata and run config
- Per-generation progress logs at generation 1, each 50th generation, and final generation
- Best tour length and best tour sequence (0-based city ids with return-to-start print)

No output files are produced by this source.

## 5. Function-by-Function Documentation

All functions listed here are in [CUDA-GA.cu](CUDA-GA.cu).

## 5.1 CUDA_CHECK macro

Purpose:
- Wrap CUDA runtime API calls.
- Convert CUDA errors into std::runtime_error exceptions.

Typical usage:

    CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));
    CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(), dist_bytes, cudaMemcpyHostToDevice));

## 5.2 eval_tour_lengths_kernel

Signature:

    __global__ void eval_tour_lengths_kernel(const int* tours,
                                             const int* dist,
                                             int* lengths,
                                             int population_size,
                                             int n)

Purpose:
- Compute one fitness value per tour in parallel.
- Mapping: one thread handles one tour.

Input:
- tours: flattened population_size x n tours
- dist: flattened n x n matrix
- population_size
- n

Output:
- lengths[tid] = cycle length for tour tid

Key kernel pattern:

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size) return;

    int base = tid * n;
    int sum = 0;
    for (int k = 0; k < n; ++k) {
        int a = tours[base + k];
        int b = tours[base + ((k + 1) % n)];
        sum += dist[a * n + b];
    }
    lengths[tid] = sum;

## 5.3 cpu_tour_length

Signature:

    static int cpu_tour_length(const std::vector<int>& dist,
                               const std::vector<int>& tour,
                               int n)

Purpose:
- Deterministic CPU recomputation of tour length.
- Used for final integrity validation.

Input:
- dist, tour, n

Output:
- integer tour length

## 5.4 nearest_neighbor_best_start

Signature:

    static TourResult nearest_neighbor_best_start(const std::vector<int>& dist, int n)

Purpose:
- Build a greedy nearest-neighbor tour for every possible start node.
- Return the best greedy tour among starts.

Input:
- dist, n

Output:
- TourResult containing best greedy seed and its length

Failure mode:
- Throws if construction cannot pick a next city.

## 5.5 parse_config

Signature:

    static GaConfig parse_config(int argc, char* argv[])

Purpose:
- Parse optional CLI parameters into GaConfig.
- Validate numeric ranges.

Input:
- argc, argv

Output:
- GaConfig

Validation rules:
- population_size >= 2
- generations >= 1
- mutation_rate in [0, 1]
- elite_count in [1, population_size)

## 5.6 initialize_population

Signature:

    static void initialize_population(std::vector<int>& population,
                                      const TourResult& greedy,
                                      int population_size,
                                      int n,
                                      std::mt19937& rng)

Purpose:
- Create initial population.
- First individual uses greedy seed.
- Remaining individuals are random permutations.

Input:
- mutable population buffer
- greedy seed
- population_size, n, rng

Output:
- population vector filled in-place

## 5.7 tournament_select

Signature:

    static int tournament_select(const std::vector<int>& lengths,
                                 int population_size,
                                 std::mt19937& rng)

Purpose:
- Select a parent index via tournament selection.
- Tournament size is fixed to 3.

Input:
- lengths
- population_size
- rng

Output:
- index of selected parent

## 5.8 order_crossover

Signature:

    static void order_crossover(const int* parent_a,
                                const int* parent_b,
                                int* child,
                                int n,
                                std::mt19937& rng)

Purpose:
- Apply Order Crossover (OX).
- Keep a segment from parent A, then fill remaining positions by parent B order skipping duplicates.

Input:
- parent_a, parent_b, child, n, rng

Output:
- child tour written in-place

## 5.9 mutate_swap

Signature:

    static void mutate_swap(int* tour,
                            int n,
                            double mutation_rate,
                            std::mt19937& rng)

Purpose:
- With probability mutation_rate, swap two random positions in the tour.

Input:
- mutable tour
- n
- mutation_rate
- rng

Output:
- tour mutated in-place (or unchanged)

## 5.10 evaluate_population_cuda

Signature:

    static void evaluate_population_cuda(const std::vector<int>& population,
                                         const int* d_dist,
                                         int* d_population,
                                         int* d_lengths,
                                         std::vector<int>& lengths,
                                         int population_size,
                                         int n)

Purpose:
- Bridge between CPU GA loop and GPU fitness kernel.

Actions:
- Copy host population to device
- Launch eval_tour_lengths_kernel
- Synchronize
- Copy lengths back to host

Input:
- host population
- preallocated device buffers
- host lengths output vector
- population_size, n

Output:
- lengths filled with fitness values

## 5.11 run_cuda_ga

Signature:

    static TourResult run_cuda_ga(const TspMatrixInstance& inst, const GaConfig& cfg)

Purpose:
- Main GA engine.

Input:
- parsed instance
- validated config

Output:
- best tour and length found during evolution

Core loop behavior:
- evaluate on GPU
- sort indices by ascending length
- update global best
- carry elite_count elites
- produce remaining children via selection + crossover + mutation
- swap buffers

Safety checks:
- n must be at least 2
- final CPU cross-check must match best length

## 5.12 main

Signature:

    int main(int argc, char* argv[])

Purpose:
- Program entry point and CLI interface.

Input:
- CLI args

Output:
- returns 0 on success, 1 on error
- prints summary + best result

## 6. Data Layout and Memory Notes

- Distance matrix layout:
  - dist[i * n + j] = edge weight i -> j
- Population layout:
  - individual p starts at offset p * n
- Kernel sizing:
  - block_size = 256
  - grid_size = ceil(population_size / block_size)
- Memory strategy:
  - d_dist copied once
  - d_population and d_lengths reused every generation

## 7. Design Characteristics and Limitations

- Strengths:
  - Clear separation of concerns: GPU fitness, CPU evolution.
  - Deterministic validation step via CPU cross-check.
  - Good baseline for profiling and incremental GPU migration.

- Limitations:
  - Selection/crossover/mutation still on CPU.
  - Population copied host->device each generation.
  - Sorting performed on CPU.
  - No GPU-side random generation for genetic operators.

## 8. LLM Context Guidance

For another LLM, include:

1. This document
2. [CUDA-GA.cu](CUDA-GA.cu) full source (included below)
3. [tsplib_parser.h](tsplib_parser.h) for parser and matrix contract

Good prompt seed:

    Preserve behavior and correctness checks while proposing GPU-side acceleration for parent selection and offspring creation.

## 9. Full Source: CUDA-GA.cu

    #include "tsplib_parser.h"

    #include <cuda_runtime.h>

    #include <algorithm>
    #include <chrono>
    #include <iostream>
    #include <limits>
    #include <numeric>
    #include <random>
    #include <stdexcept>
    #include <string>
    #include <vector>

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
        int population_size = 512;
        int generations = 1000;
        double mutation_rate = 0.05;
        int elite_count = 4;
        unsigned int seed = 0;
    };

    __global__ void eval_tour_lengths_kernel(const int* tours,
                                             const int* dist,
                                             int* lengths,
                                             int population_size,
                                             int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= population_size) return;

        int base = tid * n;
        int sum = 0;

        for (int k = 0; k < n; ++k) {
            int a = tours[base + k];
            int b = tours[base + ((k + 1) % n)];
            sum += dist[a * n + b];
        }

        lengths[tid] = sum;
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

    static TourResult nearest_neighbor_best_start(const std::vector<int>& dist, int n) {
        TourResult best;

        for (int start = 0; start < n; ++start) {
            std::vector<int> tour(n);
            std::vector<char> visited(n, 0);

            int current = start;
            tour[0] = current;
            visited[current] = 1;
            int total = 0;

            for (int pos = 1; pos < n; ++pos) {
                int best_city = -1;
                int best_dist = std::numeric_limits<int>::max();

                for (int candidate = 0; candidate < n; ++candidate) {
                    if (visited[candidate]) continue;

                    int d = dist[current * n + candidate];
                    if (d < best_dist) {
                        best_dist = d;
                        best_city = candidate;
                    }
                }

                if (best_city == -1) {
                    throw std::runtime_error("Nearest-neighbor construction failed");
                }

                tour[pos] = best_city;
                visited[best_city] = 1;
                total += best_dist;
                current = best_city;
            }

            total += dist[current * n + start];
            if (total < best.length) {
                best.tour = tour;
                best.length = total;
            }
        }

        return best;
    }

    static GaConfig parse_config(int argc, char* argv[]) {
        GaConfig cfg;
        if (argc > 2) cfg.population_size = std::stoi(argv[2]);
        if (argc > 3) cfg.generations = std::stoi(argv[3]);
        if (argc > 4) cfg.mutation_rate = std::stod(argv[4]);
        if (argc > 5) cfg.elite_count = std::stoi(argv[5]);
        if (argc > 6) cfg.seed = static_cast<unsigned int>(std::stoul(argv[6]));

        if (cfg.population_size < 2) {
            throw std::runtime_error("population_size must be at least 2");
        }
        if (cfg.generations < 1) {
            throw std::runtime_error("generations must be at least 1");
        }
        if (cfg.mutation_rate < 0.0 || cfg.mutation_rate > 1.0) {
            throw std::runtime_error("mutation_rate must be between 0 and 1");
        }
        if (cfg.elite_count < 1 || cfg.elite_count >= cfg.population_size) {
            throw std::runtime_error("elite_count must be in [1, population_size)");
        }

        return cfg;
    }

    static void initialize_population(std::vector<int>& population,
                                      const TourResult& greedy,
                                      int population_size,
                                      int n,
                                      std::mt19937& rng) {
        std::vector<int> base(n);
        std::iota(base.begin(), base.end(), 0);

        std::copy(greedy.tour.begin(), greedy.tour.end(), population.begin());

        for (int i = 1; i < population_size; ++i) {
            std::shuffle(base.begin(), base.end(), rng);
            std::copy(base.begin(), base.end(), population.begin() + i * n);
        }
    }

    static int tournament_select(const std::vector<int>& lengths,
                                 int population_size,
                                 std::mt19937& rng) {
        constexpr int tournament_size = 3;
        std::uniform_int_distribution<int> pick(0, population_size - 1);

        int best = pick(rng);
        for (int i = 1; i < tournament_size; ++i) {
            int candidate = pick(rng);
            if (lengths[candidate] < lengths[best]) {
                best = candidate;
            }
        }

        return best;
    }

    static void order_crossover(const int* parent_a,
                                const int* parent_b,
                                int* child,
                                int n,
                                std::mt19937& rng) {
        std::uniform_int_distribution<int> cut_dist(0, n - 1);
        int left = cut_dist(rng);
        int right = cut_dist(rng);
        if (left > right) std::swap(left, right);

        std::fill(child, child + n, -1);
        std::vector<char> used(n, 0);

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

    static void mutate_swap(int* tour,
                            int n,
                            double mutation_rate,
                            std::mt19937& rng) {
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        if (chance(rng) > mutation_rate) return;

        std::uniform_int_distribution<int> city_dist(0, n - 1);
        int a = city_dist(rng);
        int b = city_dist(rng);
        while (b == a) {
            b = city_dist(rng);
        }
        std::swap(tour[a], tour[b]);
    }

    static void evaluate_population_cuda(const std::vector<int>& population,
                                         const int* d_dist,
                                         int* d_population,
                                         int* d_lengths,
                                         std::vector<int>& lengths,
                                         int population_size,
                                         int n) {
        const size_t population_bytes = sizeof(int) * population.size();
        const size_t lengths_bytes = sizeof(int) * lengths.size();

        CUDA_CHECK(cudaMemcpy(d_population,
                              population.data(),
                              population_bytes,
                              cudaMemcpyHostToDevice));

        const int block_size = 256;
        const int grid_size = (population_size + block_size - 1) / block_size;

        eval_tour_lengths_kernel<<<grid_size, block_size>>>(d_population,
                                                            d_dist,
                                                            d_lengths,
                                                            population_size,
                                                            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(lengths.data(),
                              d_lengths,
                              lengths_bytes,
                              cudaMemcpyDeviceToHost));
    }

    static TourResult run_cuda_ga(const TspMatrixInstance& inst, const GaConfig& cfg) {
        const int n = inst.dimension;
        if (n < 2) {
            throw std::runtime_error("TSP dimension must be at least 2");
        }

        const int population_size = cfg.population_size;
        const unsigned int seed = cfg.seed == 0
            ? static_cast<unsigned int>(
                  std::chrono::high_resolution_clock::now().time_since_epoch().count())
            : cfg.seed;

        std::mt19937 rng(seed);
        TourResult greedy = nearest_neighbor_best_start(inst.dist, n);

        std::vector<int> population(static_cast<size_t>(population_size) * n);
        std::vector<int> next_population(population.size());
        std::vector<int> lengths(population_size, 0);
        std::vector<int> order(population_size, 0);

        initialize_population(population, greedy, population_size, n, rng);

        int* d_dist = nullptr;
        int* d_population = nullptr;
        int* d_lengths = nullptr;

        const size_t dist_bytes = sizeof(int) * inst.dist.size();
        const size_t population_bytes = sizeof(int) * population.size();
        const size_t lengths_bytes = sizeof(int) * lengths.size();

        CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));
        CUDA_CHECK(cudaMalloc(&d_population, population_bytes));
        CUDA_CHECK(cudaMalloc(&d_lengths, lengths_bytes));
        CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(), dist_bytes, cudaMemcpyHostToDevice));

        TourResult global_best = greedy;

        for (int generation = 0; generation < cfg.generations; ++generation) {
            evaluate_population_cuda(population,
                                     d_dist,
                                     d_population,
                                     d_lengths,
                                     lengths,
                                     population_size,
                                     n);

            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return lengths[a] < lengths[b];
            });

            if (lengths[order[0]] < global_best.length) {
                global_best.length = lengths[order[0]];
                global_best.tour.assign(population.begin() + order[0] * n,
                                        population.begin() + (order[0] + 1) * n);
            }

            if (generation == 0 ||
                generation == cfg.generations - 1 ||
                (generation + 1) % 50 == 0) {
                std::cout << "Generation " << (generation + 1)
                          << " best = " << lengths[order[0]]
                          << ", global best = " << global_best.length << "\n";
            }

            int out = 0;
            for (; out < cfg.elite_count; ++out) {
                int elite_idx = order[out];
                std::copy(population.begin() + elite_idx * n,
                          population.begin() + (elite_idx + 1) * n,
                          next_population.begin() + out * n);
            }

            for (; out < population_size; ++out) {
                int parent_a_idx = tournament_select(lengths, population_size, rng);
                int parent_b_idx = tournament_select(lengths, population_size, rng);

                const int* parent_a = population.data() + parent_a_idx * n;
                const int* parent_b = population.data() + parent_b_idx * n;
                int* child = next_population.data() + out * n;

                order_crossover(parent_a, parent_b, child, n, rng);
                mutate_swap(child, n, cfg.mutation_rate, rng);
            }

            population.swap(next_population);
        }

        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_population));
        CUDA_CHECK(cudaFree(d_lengths));

        const int checked_length = cpu_tour_length(inst.dist, global_best.tour, n);
        if (checked_length != global_best.length) {
            throw std::runtime_error("CPU cross-check did not match GPU best length");
        }

        return global_best;
    }

    int main(int argc, char* argv[]) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " <file.tsp> [population=512] [generations=1000]"
                      << " [mutation_rate=0.05] [elite_count=4] [seed=auto]\n";
            return 1;
        }

        try {
            GaConfig cfg = parse_config(argc, argv);
            TspMatrixInstance inst = load_tsplib_matrix(argv[1]);

            std::cout << "NAME: " << inst.name << "\n";
            std::cout << "TYPE: " << inst.type << "\n";
            std::cout << "DIMENSION: " << inst.dimension << "\n";
            std::cout << "Population: " << cfg.population_size << "\n";
            std::cout << "Generations: " << cfg.generations << "\n";
            std::cout << "Mutation rate: " << cfg.mutation_rate << "\n";
            std::cout << "Elite count: " << cfg.elite_count << "\n";

            TourResult best = run_cuda_ga(inst, cfg);

            std::cout << "\nBest GA tour length: " << best.length << "\n";
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
