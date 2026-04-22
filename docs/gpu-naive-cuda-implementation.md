# Naive CUDA TSP Implementation Documentation

This document explains the naive CUDA implementation in [GPU-Naive.cu](GPU-Naive.cu), including:

- File-level purpose and dependencies
- Function-by-function behavior
- Inputs and outputs
- Relevant code snippets
- Full source listing (for LLM context ingestion)

## 1. Scope and Purpose

The file [GPU-Naive.cu](GPU-Naive.cu) is an early CUDA integration step for TSP. It does not run a full GPU genetic algorithm. Instead, it:

1. Loads a TSPLIB matrix instance.
2. Builds a CPU nearest-neighbor baseline tour.
3. Sends a small batch of tours to the GPU.
4. Runs one CUDA kernel that computes each tour length in parallel (one thread per tour).
5. Cross-checks GPU lengths against CPU lengths.

The implementation is intentionally simple and is best viewed as a correctness and plumbing milestone before larger GPU GA kernels.

## 2. Files Involved

- [GPU-Naive.cu](GPU-Naive.cu)
  - Main implementation and executable entry point.
- [tsplib_parser.h](tsplib_parser.h)
  - Declares `TspMatrixInstance` and `load_tsplib_matrix`.
- [tsplib_parser.cpp](tsplib_parser.cpp)
  - Provides parsing implementation used by `main`.

### Data Structure Dependency

From [tsplib_parser.h](tsplib_parser.h):

```cpp
struct TspMatrixInstance {
    std::string name;
    std::string type;               // TSP, ATSP, etc.
    int dimension = 0;
    std::vector<int> dist;          // row-major N x N
};

TspMatrixInstance load_tsplib_matrix(const std::string& path);
```

## 3. High-Level Program Flow

In [GPU-Naive.cu](GPU-Naive.cu):

1. Parse CLI argument: TSPLIB file path.
2. Load matrix with `load_tsplib_matrix`.
3. Compute best nearest-neighbor tour across all start cities.
4. Construct a mini batch of 4 tours:
   - Tour 0: greedy baseline
   - Tours 1-3: cyclic rotations of the greedy tour
5. Copy tours and full matrix to GPU memory.
6. Launch `eval_tour_lengths_kernel`.
7. Copy lengths back.
8. Verify each GPU length against CPU result.
9. Print match or mismatch and exit.

## 4. Inputs and Outputs

## 4.1 Executable Input

Command line:

```text
GPU-Naive <file.tsp>
```

- Required:
  - `<file.tsp>`: path to TSPLIB matrix file.

## 4.2 Runtime Inputs (Core)

- `inst.dimension` (`N`): number of cities.
- `inst.dist`: row-major distance matrix (`N * N`).
- `h_tours`: host-side flattened list of candidate tours (`num_tours * N`).

## 4.3 Runtime Outputs

- Printed metadata: name/type/dimension and sample matrix values.
- CPU greedy baseline tour and length.
- GPU-computed tour lengths for batch.
- CPU vs GPU cross-check report for each tour.

No file outputs are written by this program.

## 5. Function-by-Function Documentation

All functions are in [GPU-Naive.cu](GPU-Naive.cu).

## 5.1 `nearest_neighbor_tour`

Signature:

```cpp
TourResult nearest_neighbor_tour(const std::vector<int>& dist, int N, int start)
```

Purpose:
- Build one greedy tour from a specific start city.

Inputs:
- `dist`: row-major `N x N` distance matrix.
- `N`: city count.
- `start`: starting city index.

Output:
- `TourResult` with:
  - `tour`: permutation of `[0..N-1]`
  - `length`: full cycle length including return to start

Throws:
- `std::runtime_error` for invalid start or failed city selection.

Key logic snippet:

```cpp
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

    result.tour[pos] = best_city;
    visited[best_city] = true;
    total += best_dist;
    current = best_city;
}
total += dist[current * N + start];
```

## 5.2 `nearest_neighbor_best_start`

Signature:

```cpp
TourResult nearest_neighbor_best_start(const std::vector<int>& dist, int N)
```

Purpose:
- Run nearest-neighbor from every possible start city and keep the best result.

Inputs:
- `dist`, `N`.

Output:
- Best `TourResult` across all start nodes.

Key logic snippet:

```cpp
for (int start = 0; start < N; ++start) {
    TourResult candidate = nearest_neighbor_tour(dist, N, start);
    if (candidate.length < best.length) {
        best = candidate;
    }
}
```

## 5.3 `cpu_tour_length`

Signature:

```cpp
int cpu_tour_length(const std::vector<int>& dist, const std::vector<int>& tour, int N)
```

Purpose:
- Deterministically compute a tour length on CPU for verification.

Inputs:
- `dist`, `tour`, `N`.

Output:
- Total cycle length.

Throws:
- `std::runtime_error` if `tour.size() != N`.

Key logic snippet:

```cpp
for (int k = 0; k < N; ++k) {
    int a = tour[k];
    int b = tour[(k + 1) % N];
    total += dist[a * N + b];
}
```

## 5.4 `CUDA_CHECK` macro

Purpose:
- Wrap CUDA runtime calls and convert errors into C++ exceptions.

Usage example:

```cpp
CUDA_CHECK(cudaMalloc(&d_tours, tours_bytes));
CUDA_CHECK(cudaMemcpy(d_tours, h_tours.data(), tours_bytes, cudaMemcpyHostToDevice));
```

Output:
- No return value; throws `std::runtime_error` on CUDA failure.

## 5.5 `eval_tour_lengths_kernel`

Signature:

```cpp
__global__ void eval_tour_lengths_kernel(const int* tours,
                                         const int* dist,
                                         int* lengths,
                                         int num_tours,
                                         int N)
```

Purpose:
- GPU kernel that assigns one thread per candidate tour and computes cycle length.

Inputs (device pointers):
- `tours`: flattened tours array of size `num_tours * N`.
- `dist`: flattened `N * N` matrix.
- `num_tours`: number of tours.
- `N`: dimension.

Output (device pointer):
- `lengths[tid]`: length for tour `tid`.

Thread mapping:
- `tid = blockIdx.x * blockDim.x + threadIdx.x`
- Guard returns if `tid >= num_tours`.

Key logic snippet:

```cpp
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
```

## 5.6 `main`

Signature:

```cpp
int main(int argc, char* argv[])
```

Purpose:
- Program entry point. Executes end-to-end flow from parsing to CPU/GPU validation.

Input:
- `argv[1]`: TSPLIB matrix file.

Output:
- Prints diagnostics and validation results.
- Returns `0` on success, `1` on failure.

Important local constants:
- `num_tours = 4`.
- `block_size = 256`.
- `grid_size = (num_tours + block_size - 1) / block_size`.

Key GPU launch snippet:

```cpp
eval_tour_lengths_kernel<<<grid_size, block_size>>>(d_tours,
                                                    d_dist,
                                                    d_lengths,
                                                    num_tours,
                                                    N);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## 6. Memory and Data Layout Notes

- Distance matrix is flattened row-major:
  - `dist[i * N + j]` is edge cost from city `i` to city `j`.
- Tour batch is flattened:
  - Tour `t` starts at offset `t * N` in `h_tours` / `d_tours`.
- Kernel writes one integer length per tour.

## 7. Current Limitations (Intentional for Naive Stage)

- No GA population evolution on GPU.
- Very small fixed tour batch (`num_tours = 4`) for testing.
- Full distance matrix stored in global memory each launch.
- No shared memory tiling or coalescing optimizations beyond straightforward access.
- No streams, overlap, or asynchronous pipeline.

## 8. Suggested LLM Context Usage

When giving this to another LLM, include:

1. This document.
2. [GPU-Naive.cu](GPU-Naive.cu) full source (included below).
3. [tsplib_parser.h](tsplib_parser.h) contract for `TspMatrixInstance`.

Prompt suggestion:

```text
Use the naive CUDA tour-evaluation implementation as a correctness baseline.
Preserve the data layout and CPU/GPU cross-check behavior when proposing optimizations.
```

## 9. Full Source: GPU-Naive.cu

```cpp
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
```
