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

//
// - One CUDA block is one independent GA island.
// - One CUDA thread inside the block owns one candidate tour.
// - The island population has BLOCK_POP_SIZE tours, so a 32-thread block maps
//   naturally to a single warp on NVIDIA GPUs.
// - Two populations, pop_a and pop_b, live in dynamic shared memory so that
//   most per-generation GA work avoids global-memory traffic.
// - The distance matrix lives in constant memory because it is read-only and
//   small enough for MAX_CITIES = 128.
// - The CPU only receives one best tour and one best length per island after
//   all generations complete, reducing host-device synchronization.
constexpr int MAX_CITIES = 128;
constexpr int BLOCK_POP_SIZE = 32;
constexpr int TOURNAMENT_SIZE = 3;

// Constant memory is a small device memory space cached for broadcast-style
// reads. It is a reasonable fit for the distance matrix in the <=128-city
// version because every thread repeatedly reads edge weights but never writes
// them. It is not ideal for every access pattern: if every thread in a warp
// reads a different address, accesses may serialize. Still, for this matrix
// size it gives a useful baseline to compare against global-memory variants.
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

// Minimal per-thread pseudo-random number generator.
//
// A full CUDA RNG library such as cuRAND would be more statistically robust,
// but this xorshift is cheap and easy to keep entirely in a thread-local
// register. Each thread receives a different seed derived from the island id,
// thread id, and user seed.
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

// Compute one complete TSP tour length.
//
// In this implementation one thread computes the length of one tour. That means
// the loop is serial inside the thread, but all tours in the island are
// evaluated in parallel across the 32 block threads.
//
// The tour array contains city indices. For every adjacent pair (a, b), we
// fetch the precomputed edge weight from c_dist. The final edge wraps from the
// last city back to the first city using (k + 1) % n.
__device__ int tour_length_const(const int* tour, int n) {
    int total = 0;
    for (int k = 0; k < n; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % n];
        total += c_dist[a * n + b];
    }
    return total;
}

// Initialize a valid random tour.
//
// A TSP tour must be a permutation: every city appears exactly once. We first
// write the identity tour [0, 1, 2, ... n-1], then apply a Fisher-Yates shuffle.
// Because all swaps are within the array, the result remains a valid
// permutation.
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

// Tournament selection.
//
// The thread randomly samples TOURNAMENT_SIZE candidate individuals and returns
// the one with the shortest tour length. This avoids sorting the whole
// population for parent selection and gives better tours a higher chance of
// reproducing without making the algorithm completely deterministic.
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

// Order crossover for permutation chromosomes.
//
// Standard one-point or two-point crossover can break a TSP tour by duplicating
// cities and omitting others. Order crossover preserves validity:
//
// 1. Pick a random segment [left, right].
// 2. Copy that segment from parent_a into the child.
// 3. Mark those cities as used.
// 4. Walk parent_b circularly and copy unused cities into the remaining child
//    slots.
//
// This keeps the child as a legal permutation while inheriting contiguous
// structure from one parent and relative ordering from the other.
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

    // This scratch array tracks which cities are already present in the child.
    // It is simple and readable, but it can be expensive: each thread has its
    // own 128-int array. The bitset variant reduces this storage pressure.
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

// Swap mutation.
//
// Mutation helps preserve diversity. With probability mutation_rate, this
// function swaps two distinct city positions in the tour. A swap preserves the
// permutation property because no city is created or removed.
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

// Main GPU island-model GA kernel.
//
// Grid/block mapping:
// - blockIdx.x identifies one island.
// - threadIdx.x identifies one individual inside that island.
// - BLOCK_POP_SIZE is 32, so each island is one warp-sized block.
//
// High-level loop:
// 1. Initialize one random tour per thread.
// 2. For each generation:
//    a. Evaluate fitness for each tour.
//    b. Sort/rank the population so elites can be copied.
//    c. Preserve elites unchanged.
//    d. Use tournament selection + crossover + mutation to fill the rest.
//    e. Swap current and next population buffers.
// 3. Evaluate the final generation and write one best result per island.
__global__ void ga_island_kernel(int n,
                                 int generations,
                                 float mutation_rate,
                                 int elite_count,
                                 unsigned int seed,
                                 int* best_tours,
                                 int* best_lengths) {
    // Dynamic shared memory is allocated by the kernel launch. We manually
    // carve it into four logical regions below:
    //
    //   pop_a   : current/next population buffer A
    //   pop_b   : current/next population buffer B
    //   lengths : one fitness value per individual
    //   order   : sorted index list used for elites
    //
    // The two population buffers allow the kernel to read parents from one
    // generation while writing children into the next generation without
    // overwriting data that other threads still need.
    extern __shared__ int shared[];

    const int tid = threadIdx.x;
    const int island = blockIdx.x;

    int* pop_a = shared;
    int* pop_b = pop_a + BLOCK_POP_SIZE * n;
    int* lengths = pop_b + BLOCK_POP_SIZE * n;
    int* order = lengths + BLOCK_POP_SIZE;

    // Derive a per-thread RNG seed. The constants are odd 32-bit mixing values
    // used to separate nearby island/thread ids so adjacent threads do not all
    // walk the same pseudo-random sequence.
    unsigned int rng = seed ^
                       (static_cast<unsigned int>(island + 1) * 747796405u) ^
                       (static_cast<unsigned int>(tid + 1) * 2891336453u);
    if (rng == 0) rng = 1;

    // Each thread initializes the tour that it owns. Since the block has exactly
    // BLOCK_POP_SIZE threads, this branch is always true for the current launch
    // configuration, but keeping it explicit makes the ownership clear.
    if (tid < BLOCK_POP_SIZE) {
        init_random_tour(pop_a + tid * n, n, rng);
    }
    // Synchronize so all initial tours are complete before any thread begins
    // reading the population in generation 0.
    __syncthreads();

    int* current = pop_a;
    int* next = pop_b;

    for (int generation = 0; generation < generations; ++generation) {
        // Fitness evaluation: each thread evaluates its own current tour.
        // This is the embarrassingly parallel part of the GA.
        if (tid < BLOCK_POP_SIZE) {
            lengths[tid] = tour_length_const(current + tid * n, n);
        }
        // All lengths must be ready before sorting, tournament selection, or
        // elite selection can read the lengths array.
        __syncthreads();

        // Simple baseline ranking step.
        //
        // Only thread 0 sorts order[] using selection sort. This is easy to
        // reason about and deterministic, but it serializes ranking work inside
        // the block. The ParallelSort variant replaces this section with a
        // cooperative bitonic sort.
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
        // All threads must wait until order[] is fully sorted before elite
        // threads read order[tid].
        __syncthreads();

        // Elitism copies the best elite_count tours unchanged into the next
        // generation. This protects the best discovered solutions from being
        // destroyed by crossover or mutation.
        if (tid < elite_count) {
            const int elite_idx = order[tid];
            for (int k = 0; k < n; ++k) {
                next[tid * n + k] = current[elite_idx * n + k];
            }
        } else if (tid < BLOCK_POP_SIZE) {
            // Non-elite threads create one child each. Parent selection only
            // reads lengths[], while crossover/mutation read parent tours and
            // write this thread's child tour.
            const int parent_a_idx = tournament_select_device(lengths, rng);
            const int parent_b_idx = tournament_select_device(lengths, rng);

            const int* parent_a = current + parent_a_idx * n;
            const int* parent_b = current + parent_b_idx * n;
            int* child = next + tid * n;

            order_crossover_device(parent_a, parent_b, child, n, rng);
            mutate_swap_device(child, n, mutation_rate, rng);
        }
        // Make sure all children and elites are written before swapping buffers.
        __syncthreads();

        // Swap population roles. The old next buffer becomes current for the
        // following generation; the old current buffer becomes scratch space for
        // future children.
        int* tmp = current;
        current = next;
        next = tmp;
        // This extra barrier keeps all threads aligned after the pointer swap.
        __syncthreads();
    }

    // After the final generation, evaluate the population one last time because
    // the loop swaps in a newly created generation without immediately ranking
    // it for output.
    if (tid < BLOCK_POP_SIZE) {
        lengths[tid] = tour_length_const(current + tid * n, n);
    }
    __syncthreads();

    if (tid == 0) {
        // Select the best individual in this island. Only one result per island
        // is written back to global memory, which keeps device-to-host transfer
        // size small.
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

// Host-side orchestration for the GPU island GA.
//
// This function owns memory allocation, input validation, kernel launch, result
// copying, and a CPU-side correctness check of the returned best tour length.
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

    // Copy the complete distance matrix into device constant memory once before
    // launching the kernel. The kernel then reads c_dist directly.
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

    // Dynamic shared-memory size for one block/island:
    //
    // - pop_a: BLOCK_POP_SIZE * n ints
    // - pop_b: BLOCK_POP_SIZE * n ints
    // - lengths: BLOCK_POP_SIZE ints
    // - order: BLOCK_POP_SIZE ints
    //
    // This is passed as the third kernel launch parameter.
    const size_t shared_ints =
        2 * static_cast<size_t>(BLOCK_POP_SIZE) * n +
        2 * static_cast<size_t>(BLOCK_POP_SIZE);
    const size_t shared_bytes = sizeof(int) * shared_ints;

    // Launch one block per island and one thread per individual.
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

    // Validate that the returned tour length matches an independent CPU
    // recomputation. This catches kernel bugs where the tour and length drift
    // out of agreement.
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

        std::cout << "VERSION: GA-GPU-POP-VerboseComments\n";
        std::cout << "NAME: " << inst.name << "\n";
        std::cout << "TYPE: " << inst.type << "\n";
        std::cout << "DIMENSION: " << inst.dimension << "\n";
        std::cout << "Islands: " << cfg.islands << "\n";
        std::cout << "Island population: " << BLOCK_POP_SIZE << "\n";
        std::cout << "Total GPU population: " << cfg.islands * BLOCK_POP_SIZE << "\n";
        std::cout << "Generations: " << cfg.generations << "\n";
        std::cout << "Mutation rate: " << cfg.mutation_rate << "\n";
        std::cout << "Elite count per island: " << cfg.elite_count << "\n";

        const auto started_at = std::chrono::high_resolution_clock::now();
        TourResult best = run_gpu_population_ga(inst, cfg);
        const auto finished_at = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_ms = finished_at - started_at;

        std::cout << "CUDA kernel elapsed ms: " << elapsed_ms.count() << "\n";
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
