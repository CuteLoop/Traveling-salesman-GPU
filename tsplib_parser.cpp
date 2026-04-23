#include "tsplib_parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class EdgeWeightType {
    EUC_2D,
    EUC_3D,
    MAX_2D,
    MAX_3D,
    MAN_2D,
    MAN_3D,
    CEIL_2D,
    GEO,
    ATT,
    EXPLICIT,
    UNKNOWN
};

enum class EdgeWeightFormat {
    FULL_MATRIX,
    UPPER_ROW,
    LOWER_ROW,
    UPPER_DIAG_ROW,
    LOWER_DIAG_ROW,
    UPPER_COL,
    LOWER_COL,
    UPPER_DIAG_COL,
    LOWER_DIAG_COL,
    FUNCTION,
    UNKNOWN
};

struct CoordNode {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

constexpr double TSPLIB_PI = 3.14159265358979323846;

static inline std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static inline std::string upper(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

static inline bool parse_key_value(const std::string& line,
                                   std::string& key,
                                   std::string& value) {
    size_t pos = line.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    key = upper(trim(line.substr(0, pos)));
    value = trim(line.substr(pos + 1));
    return true;
}

static inline int nint(double x) {
    return static_cast<int>(std::llround(x));
}

static inline EdgeWeightType parse_ewt(const std::string& s) {
    const std::string u = upper(trim(s));
    if (u == "EUC_2D") return EdgeWeightType::EUC_2D;
    if (u == "EUC_3D") return EdgeWeightType::EUC_3D;
    if (u == "MAX_2D") return EdgeWeightType::MAX_2D;
    if (u == "MAX_3D") return EdgeWeightType::MAX_3D;
    if (u == "MAN_2D") return EdgeWeightType::MAN_2D;
    if (u == "MAN_3D") return EdgeWeightType::MAN_3D;
    if (u == "CEIL_2D") return EdgeWeightType::CEIL_2D;
    if (u == "GEO") return EdgeWeightType::GEO;
    if (u == "ATT") return EdgeWeightType::ATT;
    if (u == "EXPLICIT") return EdgeWeightType::EXPLICIT;
    return EdgeWeightType::UNKNOWN;
}

static inline EdgeWeightFormat parse_ewf(const std::string& s) {
    const std::string u = upper(trim(s));
    if (u == "FULL_MATRIX") return EdgeWeightFormat::FULL_MATRIX;
    if (u == "UPPER_ROW") return EdgeWeightFormat::UPPER_ROW;
    if (u == "LOWER_ROW") return EdgeWeightFormat::LOWER_ROW;
    if (u == "UPPER_DIAG_ROW") return EdgeWeightFormat::UPPER_DIAG_ROW;
    if (u == "LOWER_DIAG_ROW") return EdgeWeightFormat::LOWER_DIAG_ROW;
    if (u == "UPPER_COL") return EdgeWeightFormat::UPPER_COL;
    if (u == "LOWER_COL") return EdgeWeightFormat::LOWER_COL;
    if (u == "UPPER_DIAG_COL") return EdgeWeightFormat::UPPER_DIAG_COL;
    if (u == "LOWER_DIAG_COL") return EdgeWeightFormat::LOWER_DIAG_COL;
    if (u == "FUNCTION") return EdgeWeightFormat::FUNCTION;
    return EdgeWeightFormat::UNKNOWN;
}

static inline double geo_to_radians(double x) {
    // TSPLIB GEO convention: DDD.MM where MM is minutes
    const int deg = static_cast<int>(x);
    const double min = x - deg;
    return TSPLIB_PI * (deg + 5.0 * min / 3.0) / 180.0;
}

static inline int dist_euc_2d(const CoordNode& a, const CoordNode& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return nint(std::sqrt(dx * dx + dy * dy));
}

static inline int dist_euc_3d(const CoordNode& a, const CoordNode& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return nint(std::sqrt(dx * dx + dy * dy + dz * dz));
}

static inline int dist_max_2d(const CoordNode& a, const CoordNode& b) {
    return static_cast<int>(std::max(std::fabs(a.x - b.x), std::fabs(a.y - b.y)));
}

static inline int dist_max_3d(const CoordNode& a, const CoordNode& b) {
    return static_cast<int>(std::max({std::fabs(a.x - b.x), std::fabs(a.y - b.y), std::fabs(a.z - b.z)}));
}

static inline int dist_man_2d(const CoordNode& a, const CoordNode& b) {
    return nint(std::fabs(a.x - b.x) + std::fabs(a.y - b.y));
}

static inline int dist_man_3d(const CoordNode& a, const CoordNode& b) {
    return nint(std::fabs(a.x - b.x) + std::fabs(a.y - b.y) + std::fabs(a.z - b.z));
}

static inline int dist_ceil_2d(const CoordNode& a, const CoordNode& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return static_cast<int>(std::ceil(std::sqrt(dx * dx + dy * dy)));
}

static inline int dist_att(const CoordNode& a, const CoordNode& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double rij = std::sqrt((dx * dx + dy * dy) / 10.0);
    const int tij = nint(rij);
    return (tij < rij) ? (tij + 1) : tij;
}

static inline int dist_geo(const CoordNode& a, const CoordNode& b) {
    constexpr double RRR = 6378.388;
    const double lati = geo_to_radians(a.x);
    const double longi = geo_to_radians(a.y);
    const double latj = geo_to_radians(b.x);
    const double longj = geo_to_radians(b.y);

    const double q1 = std::cos(longi - longj);
    const double q2 = std::cos(lati - latj);
    const double q3 = std::cos(lati + latj);

    const double dij = RRR * std::acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0;
    return static_cast<int>(dij);
}

static std::vector<double> read_all_numbers(std::istream& in) {
    std::vector<double> nums;
    std::string raw;
    while (std::getline(in, raw)) {
        const std::string line = trim(raw);
        if (line.empty()) continue;
        if (upper(line) == "EOF") break;

        std::istringstream iss(line);
        double v;
        while (iss >> v) {
            nums.push_back(v);
        }
    }
    return nums;
}

static void fill_symmetric(std::vector<int>& dist, int n, int i, int j, int v) {
    dist[i * n + j] = v;
    dist[j * n + i] = v;
}

static std::vector<int> build_from_explicit(int n,
                                            const std::vector<double>& vals,
                                            EdgeWeightFormat fmt,
                                            bool asymmetric) {
    std::vector<int> dist(n * n, 0);
    size_t idx = 0;

    auto next_val = [&]() -> int {
        if (idx >= vals.size()) {
            throw std::runtime_error("EDGE_WEIGHT_SECTION ended too early");
        }
        return static_cast<int>(std::llround(vals[idx++]));
    };

    switch (fmt) {
        case EdgeWeightFormat::FULL_MATRIX: {
            if (vals.size() < static_cast<size_t>(n) * static_cast<size_t>(n)) {
                throw std::runtime_error("FULL_MATRIX does not contain enough values");
            }
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    dist[i * n + j] = next_val();
                }
            }
            break;
        }

        case EdgeWeightFormat::UPPER_ROW: {
            for (int i = 0; i < n; ++i) dist[i * n + i] = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    fill_symmetric(dist, n, i, j, next_val());
                }
            }
            break;
        }

        case EdgeWeightFormat::LOWER_ROW: {
            for (int i = 0; i < n; ++i) dist[i * n + i] = 0;
            for (int i = 1; i < n; ++i) {
                for (int j = 0; j < i; ++j) {
                    fill_symmetric(dist, n, i, j, next_val());
                }
            }
            break;
        }

        case EdgeWeightFormat::UPPER_DIAG_ROW: {
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    const int v = next_val();
                    fill_symmetric(dist, n, i, j, v);
                }
            }
            break;
        }

        case EdgeWeightFormat::LOWER_DIAG_ROW: {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    const int v = next_val();
                    fill_symmetric(dist, n, i, j, v);
                }
            }
            break;
        }

        case EdgeWeightFormat::UPPER_COL: {
            for (int i = 0; i < n; ++i) dist[i * n + i] = 0;
            for (int j = 1; j < n; ++j) {
                for (int i = 0; i < j; ++i) {
                    fill_symmetric(dist, n, i, j, next_val());
                }
            }
            break;
        }

        case EdgeWeightFormat::LOWER_COL: {
            for (int i = 0; i < n; ++i) dist[i * n + i] = 0;
            for (int j = 0; j < n - 1; ++j) {
                for (int i = j + 1; i < n; ++i) {
                    fill_symmetric(dist, n, i, j, next_val());
                }
            }
            break;
        }

        case EdgeWeightFormat::UPPER_DIAG_COL: {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i <= j; ++i) {
                    const int v = next_val();
                    fill_symmetric(dist, n, i, j, v);
                }
            }
            break;
        }

        case EdgeWeightFormat::LOWER_DIAG_COL: {
            for (int j = 0; j < n; ++j) {
                for (int i = j; i < n; ++i) {
                    const int v = next_val();
                    fill_symmetric(dist, n, i, j, v);
                }
            }
            break;
        }

        case EdgeWeightFormat::FUNCTION:
            throw std::runtime_error("EDGE_WEIGHT_FORMAT FUNCTION is not implemented");

        default:
            if (asymmetric) {
                throw std::runtime_error("ATSP EXPLICIT instance requires a supported EDGE_WEIGHT_FORMAT");
            }
            throw std::runtime_error("Unsupported EDGE_WEIGHT_FORMAT for EXPLICIT");
    }

    if (idx != vals.size()) {
        // tolerate trailing whitespace-only reads? read_all_numbers already strips that.
        // leftover values usually mean malformed file or wrong format.
        // We keep this strict.
        throw std::runtime_error("EDGE_WEIGHT_SECTION contains extra values");
    }

    return dist;
}

static std::vector<int> build_from_coords(const std::vector<CoordNode>& coords,
                                          EdgeWeightType ewt) {
    const int n = static_cast<int>(coords.size());
    std::vector<int> dist(n * n, 0);

    for (int i = 0; i < n; ++i) {
        dist[i * n + i] = 0;
        for (int j = i + 1; j < n; ++j) {
            int d = 0;
            switch (ewt) {
                case EdgeWeightType::EUC_2D:  d = dist_euc_2d(coords[i], coords[j]); break;
                case EdgeWeightType::EUC_3D:  d = dist_euc_3d(coords[i], coords[j]); break;
                case EdgeWeightType::MAX_2D:  d = dist_max_2d(coords[i], coords[j]); break;
                case EdgeWeightType::MAX_3D:  d = dist_max_3d(coords[i], coords[j]); break;
                case EdgeWeightType::MAN_2D:  d = dist_man_2d(coords[i], coords[j]); break;
                case EdgeWeightType::MAN_3D:  d = dist_man_3d(coords[i], coords[j]); break;
                case EdgeWeightType::CEIL_2D: d = dist_ceil_2d(coords[i], coords[j]); break;
                case EdgeWeightType::ATT:     d = dist_att(coords[i], coords[j]); break;
                case EdgeWeightType::GEO:     d = dist_geo(coords[i], coords[j]); break;
                default:
                    throw std::runtime_error("Unsupported coordinate-based EDGE_WEIGHT_TYPE");
            }
            fill_symmetric(dist, n, i, j, d);
        }
    }
    return dist;
}

} // namespace

TspMatrixInstance load_tsplib_matrix(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    TspMatrixInstance out;

    EdgeWeightType ewt = EdgeWeightType::UNKNOWN;
    EdgeWeightFormat ewf = EdgeWeightFormat::UNKNOWN;

    bool in_node_coord_section = false;
    bool in_edge_weight_section = false;

    std::vector<CoordNode> coords;
    std::vector<double> explicit_vals;

    std::string raw;
    while (std::getline(f, raw)) {
        std::string line = trim(raw);
        if (line.empty()) continue;

        const std::string uline = upper(line);
        if (uline == "EOF") break;

        if (in_node_coord_section) {
            // A section header means the section ended unexpectedly
            if (uline == "EDGE_WEIGHT_SECTION" || uline == "DISPLAY_DATA_SECTION") {
                in_node_coord_section = false;
            } else {
                std::istringstream iss(line);
                int id = 0;
                CoordNode c;

                if (ewt == EdgeWeightType::EUC_3D ||
                    ewt == EdgeWeightType::MAX_3D ||
                    ewt == EdgeWeightType::MAN_3D) {
                    if (!(iss >> id >> c.x >> c.y >> c.z)) {
                        throw std::runtime_error("Bad NODE_COORD_SECTION line: " + line);
                    }
                } else {
                    if (!(iss >> id >> c.x >> c.y)) {
                        throw std::runtime_error("Bad NODE_COORD_SECTION line: " + line);
                    }
                }

                coords.push_back(c);
                continue;
            }
        }

        if (in_edge_weight_section) {
            // read current line plus rest of stream as numeric values
            std::istringstream iss(line);
            double v;
            while (iss >> v) {
                explicit_vals.push_back(v);
            }
            const std::vector<double> more = read_all_numbers(f);
            explicit_vals.insert(explicit_vals.end(), more.begin(), more.end());
            break;
        }

        if (uline == "NODE_COORD_SECTION") {
            if (out.dimension <= 0) {
                throw std::runtime_error("DIMENSION must be set before NODE_COORD_SECTION");
            }
            coords.reserve(out.dimension);
            in_node_coord_section = true;
            continue;
        }

        if (uline == "EDGE_WEIGHT_SECTION") {
            if (out.dimension <= 0) {
                throw std::runtime_error("DIMENSION must be set before EDGE_WEIGHT_SECTION");
            }
            in_edge_weight_section = true;
            continue;
        }

        std::string key, value;
        if (parse_key_value(line, key, value)) {
            if (key == "NAME") out.name = value;
            else if (key == "TYPE") out.type = value;
            else if (key == "DIMENSION") out.dimension = std::stoi(value);
            else if (key == "EDGE_WEIGHT_TYPE") ewt = parse_ewt(value);
            else if (key == "EDGE_WEIGHT_FORMAT") ewf = parse_ewf(value);
            continue;
        }
    }

    if (out.dimension <= 0) {
        throw std::runtime_error("Missing or invalid DIMENSION");
    }

    const bool asymmetric = (upper(out.type) == "ATSP");

    if (ewt == EdgeWeightType::EXPLICIT) {
        if (explicit_vals.empty()) {
            throw std::runtime_error("EXPLICIT instance missing EDGE_WEIGHT_SECTION");
        }
        out.dist = build_from_explicit(out.dimension, explicit_vals, ewf, asymmetric);
        return out;
    }

    if (coords.empty()) {
        throw std::runtime_error("Coordinate-based instance missing NODE_COORD_SECTION");
    }

    if (static_cast<int>(coords.size()) != out.dimension) {
        throw std::runtime_error("NODE_COORD_SECTION count does not match DIMENSION");
    }

    out.dist = build_from_coords(coords, ewt);
    return out;
}
