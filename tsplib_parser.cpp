#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tsplib_parser.h"

static inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static inline std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

enum class EdgeWeightType {
    EUC_2D,
    EUC_3D,
    // Add more later: ATT, GEO, EXPLICIT, ...
    UNKNOWN
};

struct TspInstance {
    std::string name;
    int dimension = 0;
    EdgeWeightType ewt = EdgeWeightType::UNKNOWN;

    // CUDA-friendly SoA coordinates
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z; // empty unless EUC_3D

    bool is3D() const { return !z.empty(); }
};

// TSPLIB-like rounding: "round to nearest integer"
static inline int tsp_round(double v) {
    return static_cast<int>(std::llround(v));
}

// EUC_2D distance definition commonly used in TSPLIB (rounded)
static inline int dist_euc_2d(const TspInstance& inst, int i, int j) {
    double dx = static_cast<double>(inst.x[i]) - inst.x[j];
    double dy = static_cast<double>(inst.y[i]) - inst.y[j];
    return tsp_round(std::sqrt(dx * dx + dy * dy));
}

// Optional EUC_3D
static inline int dist_euc_3d(const TspInstance& inst, int i, int j) {
    double dx = static_cast<double>(inst.x[i]) - inst.x[j];
    double dy = static_cast<double>(inst.y[i]) - inst.y[j];
    double dz = static_cast<double>(inst.z[i]) - inst.z[j];
    return tsp_round(std::sqrt(dx * dx + dy * dy + dz * dz));
}

static inline EdgeWeightType parse_ewt(const std::string& v) {
    std::string u = upper(trim(v));
    if (u == "EUC_2D") return EdgeWeightType::EUC_2D;
    if (u == "EUC_3D") return EdgeWeightType::EUC_3D;
    return EdgeWeightType::UNKNOWN;
}

// Splits "KEY: VALUE" or "KEY : VALUE"
static inline bool parse_key_value(const std::string& line, std::string& key, std::string& value) {
    auto pos = line.find(':');
    if (pos == std::string::npos) return false;
    key = upper(trim(line.substr(0, pos)));
    value = trim(line.substr(pos + 1));
    return true;
}

TspInstance load_tsplib_tsp(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open file: " + path);

    TspInstance inst;

    bool in_node_coord_section = false;

    // TSPLIB node ids are typically 1..N but not guaranteed to be ordered;
    // we'll map them into 0..N-1 based on read order (most files are ordered).
    // If you want strict id mapping, we can add an id->index map.
    int coords_read = 0;

    std::string raw;
    while (std::getline(f, raw)) {
        std::string line = trim(raw);
        if (line.empty()) continue;

        std::string uline = upper(line);

        if (uline == "EOF") break;

        if (!in_node_coord_section) {
            // Section headers
            if (uline == "NODE_COORD_SECTION") {
                if (inst.dimension <= 0) {
                    throw std::runtime_error("DIMENSION must appear before NODE_COORD_SECTION");
                }
                if (inst.ewt == EdgeWeightType::UNKNOWN) {
                    throw std::runtime_error("EDGE_WEIGHT_TYPE missing or unsupported (need EUC_2D/EUC_3D)");
                }

                inst.x.assign(inst.dimension, 0.0f);
                inst.y.assign(inst.dimension, 0.0f);
                if (inst.ewt == EdgeWeightType::EUC_3D) inst.z.assign(inst.dimension, 0.0f);

                in_node_coord_section = true;
                continue;
            }

            // Header key: value lines
            std::string key, value;
            if (parse_key_value(line, key, value)) {
                if (key == "NAME") inst.name = value;
                else if (key == "DIMENSION") inst.dimension = std::stoi(value);
                else if (key == "EDGE_WEIGHT_TYPE") inst.ewt = parse_ewt(value);
                // ignore others for now: TYPE, COMMENT, etc.
                continue;
            }

            // Ignore unknown header lines
            continue;
        } else {
            // Parsing coords lines: "id x y" (and maybe z)
            // Some files may include extra spaces/tabs.
            std::istringstream iss(line);

            int id = 0;
            double xd = 0.0, yd = 0.0, zd = 0.0;

            if (inst.ewt == EdgeWeightType::EUC_3D) {
                if (!(iss >> id >> xd >> yd >> zd)) {
                    // Some TSPLIB files end sections with another keyword; handle it:
                    if (upper(line) == "EOF") break;
                    throw std::runtime_error("Bad NODE_COORD_SECTION line: '" + line + "'");
                }
            } else {
                if (!(iss >> id >> xd >> yd)) {
                    if (upper(line) == "EOF") break;
                    throw std::runtime_error("Bad NODE_COORD_SECTION line: '" + line + "'");
                }
            }

            if (coords_read >= inst.dimension) {
                // Some files might have trailing lines; treat as error for safety
                throw std::runtime_error("Read more coordinates than DIMENSION");
            }

            // Store in read order (0..N-1)
            inst.x[coords_read] = static_cast<float>(xd);
            inst.y[coords_read] = static_cast<float>(yd);
            if (inst.ewt == EdgeWeightType::EUC_3D) inst.z[coords_read] = static_cast<float>(zd);

            coords_read++;
            if (coords_read == inst.dimension) {
                // We can stop early (some files still have EOF)
                // but keep reading to allow EOF without issue
                in_node_coord_section = false;
            }
        }
    }

    if (inst.dimension <= 0) throw std::runtime_error("Missing/invalid DIMENSION");
    if (inst.ewt == EdgeWeightType::UNKNOWN) throw std::runtime_error("Missing/unsupported EDGE_WEIGHT_TYPE");
    if (static_cast<int>(inst.x.size()) != inst.dimension) {
        throw std::runtime_error("Did not read NODE_COORD_SECTION (or incomplete)");
    }

    return inst;
}

// Example usage: compute a tour length on CPU for validation
std::int64_t tour_length_cpu(const TspInstance& inst, const std::vector<int>& tour) {
    if (static_cast<int>(tour.size()) != inst.dimension) throw std::runtime_error("tour size != dimension");

    std::int64_t sum = 0;
    for (int k = 0; k < inst.dimension; ++k) {
        int i = tour[k];
        int j = tour[(k + 1) % inst.dimension];

        if (inst.ewt == EdgeWeightType::EUC_2D) sum += dist_euc_2d(inst, i, j);
        else if (inst.ewt == EdgeWeightType::EUC_3D) sum += dist_euc_3d(inst, i, j);
        else throw std::runtime_error("distance type unsupported in tour_length_cpu");
    }
    return sum;
}