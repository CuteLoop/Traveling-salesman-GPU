	#include "tsplib_parser.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

//This cpp implements a greedy closest edge algorithm for testing the tsplib parsing as below
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

    // close the tour
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

        // Greedy nearest-neighbor baseline
        TourResult greedy = nearest_neighbor_best_start(inst.dist, inst.dimension);

        std::cout << "\nGreedy nearest-neighbor result:\n";
        std::cout << "Tour length: " << greedy.length << "\n";

        std::cout << "Tour (0-based indices):\n";
        for (int i = 0; i < inst.dimension; ++i) {
            std::cout << greedy.tour[i] << " ";
        }
        std::cout << greedy.tour[0] << "\n"; // show return to start

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}