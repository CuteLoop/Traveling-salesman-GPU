#include "tsplib_parser.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

//this implements a greedy nearest-neighbor algorithm in CPU
//the CUDA kernel implements tour evaluation with a tour per thread approach 

struct TourResult {
    std::vector<int> tour;
    int length = 0;
};

TourResult nearest_neighbor_tour(const std::vector<int>& dist, int N, int start) {
    if (start < 0 || start >= N) {
        throw std::runtime_error("Invalid start city");
    }

    TourResult result;
    result.tour.resize(N);

    std::vector<bool> visited(N, false);

    int current = start;
    result.tour[0] = current;
    visited[current] = true;
    int total = 0;

    for (int pos = 1; pos < N; ++pos) {
        int best_city = -1;
        int best_dist = std::numeric_limits<int>::max();

        for (int candidate = 0; candidate < N; ++candidate) {
            if (visited[candidate]) continue;

            int d = dist[current * N + candidate];
            if (d < best_dist) {
                best_dist = d;
                best_city = candidate;
            }
        }

        if (best_city == -1) {
            throw std::runtime_error("Failed to find next city");
        }

        result.tour[pos] = best_city;
        visited[best_city] = true;
        total += best_dist;
        current = best_city;
    }

    total += dist[current * N + start];
    result.length = total;

    return result;
}

TourResult nearest_neighbor_best_start(const std::vector<int>& dist, int N) {
    TourResult best;
    best.length = std::numeric_limits<int>::max();

    for (int start = 0; start < N; ++start) {
        TourResult candidate = nearest_neighbor_tour(dist, N, start);
        if (candidate.length < best.length) {
            best = candidate;
        }
    }

    return best;
}

int cpu_tour_length(const std::vector<int>& dist, const std::vector<int>& tour, int N) {
    if (static_cast<int>(tour.size()) != N) {
        throw std::runtime_error("Tour size does not match dimension");
    }

    int total = 0;
    for (int k = 0; k < N; ++k) {
        int a = tour[k];
        int b = tour[(k + 1) % N];
        total += dist[a * N + b];
    }
    return total;
}

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

__global__ void eval_tour_lengths_kernel(const int* tours,
                                         const int* dist,
                                         int* lengths,
                                         int num_tours,
                                         int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tours) return;

    int base = tid * N;
    int sum = 0;

    for (int k = 0; k < N; ++k) {
        int a = tours[base + k];
        int b = tours[base + ((k + 1) % N)];
        sum += dist[a * N + b];
    }

    lengths[tid] = sum;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.tsp>\n";
        return 1;
    }

    try {
        TspMatrixInstance inst = load_tsplib_matrix(argv[1]);

        std::cout << "NAME: " << inst.name << "\n";
        std::cout << "TYPE: " << inst.type << "\n";
        std::cout << "DIMENSION: " << inst.dimension << "\n";

        std::cout << "First 15 matrix elements:\n";
        const int count = std::min(15, static_cast<int>(inst.dist.size()));
        for (int i = 0; i < count; ++i) {
            std::cout << inst.dist[i] << " ";
        }
        std::cout << "\n";

        // CPU greedy baseline
        TourResult greedy = nearest_neighbor_best_start(inst.dist, inst.dimension);

        std::cout << "\nGreedy nearest-neighbor result:\n";
        std::cout << "Tour length: " << greedy.length << "\n";

        std::cout << "Tour (0-based indices):\n";
        for (int i = 0; i < inst.dimension; ++i) {
            std::cout << greedy.tour[i] << " ";
        }
        std::cout << greedy.tour[0] << "\n";

        // ------------------------------------------------------------
        // CUDA test: evaluate tour lengths on GPU
        // For now, build a tiny batch of tours from the greedy tour
        // ------------------------------------------------------------
        const int N = inst.dimension;
        const int num_tours = 4;

        std::vector<int> h_tours(num_tours * N);

        // tour 0 = greedy
        for (int k = 0; k < N; ++k) {
            h_tours[0 * N + k] = greedy.tour[k];
        }

        // tours 1..3 = rotated greedy tours, just for testing
        for (int t = 1; t < num_tours; ++t) {
            int shift = t % N;
            for (int k = 0; k < N; ++k) {
                h_tours[t * N + k] = greedy.tour[(k + shift) % N];
            }
        }

        std::vector<int> h_lengths(num_tours, 0);

        int* d_tours = nullptr;
        int* d_dist = nullptr;
        int* d_lengths = nullptr;

        size_t tours_bytes = sizeof(int) * h_tours.size();
        size_t dist_bytes = sizeof(int) * inst.dist.size();
        size_t lengths_bytes = sizeof(int) * h_lengths.size();

        CUDA_CHECK(cudaMalloc(&d_tours, tours_bytes));
        CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));
        CUDA_CHECK(cudaMalloc(&d_lengths, lengths_bytes));

        CUDA_CHECK(cudaMemcpy(d_tours, h_tours.data(), tours_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dist, inst.dist.data(), dist_bytes, cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (num_tours + block_size - 1) / block_size;

        eval_tour_lengths_kernel<<<grid_size, block_size>>>(d_tours,
                                                            d_dist,
                                                            d_lengths,
                                                            num_tours,
                                                            N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_lengths.data(), d_lengths, lengths_bytes, cudaMemcpyDeviceToHost));

        std::cout << "\nGPU tour length evaluation:\n";
        for (int t = 0; t < num_tours; ++t) {
            std::cout << "Tour " << t << " GPU length = " << h_lengths[t] << "\n";
        }

        std::cout << "\nCPU cross-check:\n";
        for (int t = 0; t < num_tours; ++t) {
            std::vector<int> temp_tour(N);
            for (int k = 0; k < N; ++k) {
                temp_tour[k] = h_tours[t * N + k];
            }

            int cpu_len = cpu_tour_length(inst.dist, temp_tour, N);
            std::cout << "Tour " << t
                      << " CPU length = " << cpu_len
                      << ", GPU length = " << h_lengths[t];

            if (cpu_len == h_lengths[t]) {
                std::cout << "  [MATCH]";
            } else {
                std::cout << "  [MISMATCH]";
            }
            std::cout << "\n";
        }

        CUDA_CHECK(cudaFree(d_tours));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_lengths));
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}