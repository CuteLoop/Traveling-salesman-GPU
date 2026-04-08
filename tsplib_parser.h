#ifndef TSP_PARSER_H
#define TSP_PARSER_H

#include <string>
#include <vector>
#include <cstdint>

enum class EdgeWeightType {
    EUC_2D,
    EUC_3D,
    UNKNOWN
};

struct TspInstance {
    std::string name;
    int dimension = 0;
    EdgeWeightType ewt = EdgeWeightType::UNKNOWN;

    // CUDA-friendly SoA layout
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z; // only used for 3D

    bool is3D() const { return !z.empty(); }
};

// Parser
TspInstance load_tsplib_tsp(const std::string& path);

// Distance helpers
int dist_euc_2d(const TspInstance& inst, int i, int j);
int dist_euc_3d(const TspInstance& inst, int i, int j);

// CPU validation helper
std::int64_t tour_length_cpu(const TspInstance& inst, const std::vector<int>& tour);

#endif