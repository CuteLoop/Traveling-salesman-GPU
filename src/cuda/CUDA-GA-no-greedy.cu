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
                                  int population_size,
                                  int n,
                                  std::mt19937& rng) {
    std::vector<int> base(n);
    std::iota(base.begin(), base.end(), 0);

    for (int i = 0; i < population_size; ++i) {
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

    std::vector<int> population(static_cast<size_t>(population_size) * n);
    std::vector<int> next_population(population.size());
    std::vector<int> lengths(population_size, 0);
    std::vector<int> order(population_size, 0);

    initialize_population(population, population_size, n, rng);

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

    TourResult global_best;

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

        std::cout << "VERSION: cuda_ga_no_greedy\n";
        std::cout << "NAME: " << inst.name << "\n";
        std::cout << "TYPE: " << inst.type << "\n";
        std::cout << "DIMENSION: " << inst.dimension << "\n";
        std::cout << "Population: " << cfg.population_size << "\n";
        std::cout << "Generations: " << cfg.generations << "\n";
        std::cout << "Mutation rate: " << cfg.mutation_rate << "\n";
        std::cout << "Elite count: " << cfg.elite_count << "\n";

        const auto started_at = std::chrono::high_resolution_clock::now();
        TourResult best = run_cuda_ga(inst, cfg);
        const auto finished_at = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_ms = finished_at - started_at;

        std::cout << "CUDA kernel elapsed ms: " << elapsed_ms.count() << "\n";
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
