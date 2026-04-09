#ifndef TSPLIB_PARSER_H
#define TSPLIB_PARSER_H

#include <string>
#include <vector>

struct TspMatrixInstance {
    std::string name;
    std::string type;               // TSP, ATSP, etc.
    int dimension = 0;
    std::vector<int> dist;          // row-major N x N
};

TspMatrixInstance load_tsplib_matrix(const std::string& path);

#endif