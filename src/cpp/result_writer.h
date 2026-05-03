// result_writer.h
// Single-row CSV append helper for TSP-GPU experiment binaries.
//
// Each binary calls write_result_row(...) once, just before it exits, with
// whatever fields it actually measured. Unset fields are written as NA.
//
// Two environment variables drive the I/O:
//   RESULT_CSV  Absolute or repo-relative path to the per-variant CSV.
//               If unset or empty, write_result_row is a silent no-op so the
//               binary can still be run interactively without side effects.
//   RUN_ID      Integer (or any string) used to identify the run. NA if unset.
//
// Concurrency: each binary writes to its own per-variant file (the SLURM
// runner sets RESULT_CSV per binary), so no inter-process locking is needed
// in the current sequential sweep flow. If parallel execution is added later,
// switch to flock-protected appends in this helper.
//
// CSV schema (one header line per file, written by the runner before any
// binary is invoked):
//   version,run_id,dataset,seed,islands,population,generations_requested,
//   mutation_rate,elite_count,best_length,kernel_ms,total_ms,
//   generations_completed,target_reached,best_tour
//
// Field semantics:
//   kernel_ms    - GPU kernel-only time, captured via cudaEvent. NA if the
//                  variant only measures end-to-end with chrono.
//   total_ms     - End-to-end wall time around the GA call (chrono). Always
//                  set if the variant timed anything at all.
//   target_reached - "yes" / "no" / "NA". Only the gpu_pop family with a
//                    target-length stop condition fills this.
//   best_tour    - Space-separated 0-based indices, including the start city
//                  repeated at the end (matching the tour print format).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

struct ResultRow {
    std::string version;             // e.g. "cuda_ga_b1_stride"
    std::string dataset;             // e.g. "data/berlin52.tsp"
    long long   seed                  = -1;  // <0 => NA
    int         islands               = -1;
    int         population            = -1;  // total per-thread or per-host pop
    int         generations_requested = -1;
    float       mutation_rate         = -1.0f;
    int         elite_count           = -1;
    long long   best_length           = -1;
    double      kernel_ms             = -1.0;  // <0 => NA
    double      total_ms              = -1.0;
    int         generations_completed = -1;
    int         target_reached        = -1;     // -1 NA, 0 no, 1 yes
    std::vector<int> best_tour;                 // empty => NA
};

namespace result_writer_detail {

inline std::string ll_or_na(long long v) {
    if (v < 0) return "NA";
    return std::to_string(v);
}

inline std::string int_or_na(int v) {
    if (v < 0) return "NA";
    return std::to_string(v);
}

inline std::string double_or_na(double v) {
    if (v < 0.0) return "NA";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return std::string(buf);
}

inline std::string float_or_na(float v) {
    if (v < 0.0f) return "NA";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return std::string(buf);
}

inline std::string target_reached_str(int v) {
    if (v < 0) return "NA";
    return v ? "yes" : "no";
}

inline std::string format_tour(const std::vector<int>& tour) {
    if (tour.empty()) return "NA";
    std::string out;
    out.reserve(tour.size() * 4 + 4);
    for (size_t i = 0; i < tour.size(); ++i) {
        if (i) out.push_back(' ');
        out += std::to_string(tour[i]);
    }
    // Mimic stdout convention: repeat the start city at the end.
    out.push_back(' ');
    out += std::to_string(tour.front());
    return out;
}

}  // namespace result_writer_detail

// Append a single CSV row to whatever file RESULT_CSV points at.
// Silently no-ops if RESULT_CSV is unset, so binaries are still safe to run
// interactively for debugging.
inline void write_result_row(const ResultRow& r) {
    const char* csv_path = std::getenv("RESULT_CSV");
    if (!csv_path || !*csv_path) return;

    const char* run_id_env = std::getenv("RUN_ID");
    const std::string run_id = (run_id_env && *run_id_env) ? run_id_env : "NA";

    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        std::fprintf(stderr,
                     "[result_writer] WARN: could not open RESULT_CSV='%s' for append\n",
                     csv_path);
        return;
    }

    using namespace result_writer_detail;

    out << r.version << ','
        << run_id << ','
        << r.dataset << ','
        << ll_or_na(r.seed) << ','
        << int_or_na(r.islands) << ','
        << int_or_na(r.population) << ','
        << int_or_na(r.generations_requested) << ','
        << float_or_na(r.mutation_rate) << ','
        << int_or_na(r.elite_count) << ','
        << ll_or_na(r.best_length) << ','
        << double_or_na(r.kernel_ms) << ','
        << double_or_na(r.total_ms) << ','
        << int_or_na(r.generations_completed) << ','
        << target_reached_str(r.target_reached) << ','
        << format_tour(r.best_tour) << '\n';
}

// Schema string the SLURM runner uses when it pre-creates an empty per-variant
// CSV. Centralized here so the binary and the runner cannot drift.
inline const char* result_csv_header() {
    return "version,run_id,dataset,seed,islands,population,generations_requested,"
           "mutation_rate,elite_count,best_length,kernel_ms,total_ms,"
           "generations_completed,target_reached,best_tour";
}
