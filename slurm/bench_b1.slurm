#!/bin/bash
# bench_b1.slurm
# Benchmark: B1 Stride Padding — shared memory bank conflict elimination
# Compares CUDA-GA-GPU-Pop (V0 baseline) vs CUDA-GA-B1-stride.
#
# What this measures:
#   - shared_load_transactions_per_request  (should drop 32 -> ~1)
#   - shared_store_transactions_per_request (should drop 32 -> ~1)
#   - Wall-clock kernel time (should decrease significantly)
#
# Usage: sbatch bench_b1.slurm
# Output: results/bench_b1_<jobid>.txt

#SBATCH --job-name=bench_b1_stride
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:30:00
#SBATCH -o results/bench_b1_%j.out
#SBATCH -e results/bench_b1_%j.error

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
    module load cuda11/11.8
else
    module load cuda11/11.0
fi

PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="/tmp/bench_b1_${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD_DIR" "${PROJ_DIR}/results"

TSP="${PROJ_DIR}/sequential/tests/fixtures/smoke_20.tsp"
# NOTE: For n=128 (worst-case 32-way conflict), use a 128-city TSP:
#   TSP="${PROJ_DIR}/data/benchmark_128.tsp"
# n=20 has at most an 8-way conflict; real gains are most visible at n=128.

ISLANDS=128
GENS=100        # Fewer gens for nvprof runs (reduces profiling time)
GENS_TIME=500   # More gens for wall-clock timing (more stable measurement)
SEED=42

OUT="${PROJ_DIR}/results/bench_b1_${SLURM_JOB_ID:-local}.txt"
exec > >(tee -a "$OUT") 2>&1

echo "================================================================"
echo "  B1 Benchmark: Stride Padding (bank conflict elimination)"
echo "  Job: ${SLURM_JOB_ID:-local}  Date: $(date)"
echo "  Host: $(hostname)  TSP: $TSP"
echo "================================================================"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "--- Building V0 (baseline) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_v0" \
    "${PROJ_DIR}/src/cuda/CUDA-GA-GPU-Pop.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo "--- Building B1 (stride = n+1) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b1" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B1-stride.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo ""

# ── Static analysis: local memory check ──────────────────────────────────────
echo "================================================================"
echo "  STATIC ANALYSIS: ptxas register / smem / lmem report"
echo "================================================================"
echo "--- V0 ---"
nvcc -arch=sm_60 -O3 --ptxas-options=-v \
     -o /dev/null \
    "${PROJ_DIR}/src/cuda/CUDA-GA-GPU-Pop.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp" 2>&1 | grep -E "lmem|smem|registers" || true

echo "--- B1 ---"
nvcc -arch=sm_60 -O3 --ptxas-options=-v \
     -o /dev/null \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B1-stride.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp" 2>&1 | grep -E "lmem|smem|registers" || true

echo ""
# Expected: smem increases by 256 bytes (padding), lmem unchanged (B2 not yet applied).

# ── nvprof: shared memory transaction counts ──────────────────────────────────
echo "================================================================"
echo "  METRIC 1: shared_load_transactions_per_request"
echo "  TARGET:   ~32 (V0) -> ~1 (B1)"
echo "================================================================"

echo "--- V0 shared load transactions ---"
nvprof --metrics shared_load_transactions_per_request \
       "${BUILD_DIR}/ga_v0" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "shared_load_transactions_per_request|Kernel|Error" || true

echo ""
echo "--- B1 shared load transactions ---"
nvprof --metrics shared_load_transactions_per_request \
       "${BUILD_DIR}/ga_b1" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "shared_load_transactions_per_request|Kernel|Error" || true

echo ""
echo "================================================================"
echo "  METRIC 2: shared_store_transactions_per_request"
echo "  TARGET:   ~32 (V0) -> ~1 (B1)"
echo "================================================================"

echo "--- V0 shared store transactions ---"
nvprof --metrics shared_store_transactions_per_request \
       "${BUILD_DIR}/ga_v0" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "shared_store_transactions_per_request|Kernel|Error" || true

echo ""
echo "--- B1 shared store transactions ---"
nvprof --metrics shared_store_transactions_per_request \
       "${BUILD_DIR}/ga_b1" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "shared_store_transactions_per_request|Kernel|Error" || true

echo ""

# ── Wall-clock timing: 5 runs each ───────────────────────────────────────────
echo "================================================================"
echo "  WALL-CLOCK TIMING (${GENS_TIME} generations, 5 runs each)"
echo "================================================================"

echo "--- V0 timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_v0" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B1 timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b1" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""

# ── nvprof trace: kernel execution time ──────────────────────────────────────
echo "================================================================"
echo "  KERNEL EXECUTION TIME (GPU trace)"
echo "================================================================"

echo "--- V0 kernel time ---"
nvprof --print-gpu-trace \
       "${BUILD_DIR}/ga_v0" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "ga_island_kernel|Duration" || true

echo ""
echo "--- B1 kernel time ---"
nvprof --print-gpu-trace \
       "${BUILD_DIR}/ga_b1" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "ga_island_kernel|Duration" || true

echo ""
echo "================================================================"
echo "  SUMMARY TABLE (fill in from output above)"
echo "================================================================"
echo "  Metric                                     V0      B1"
echo "  shared_load_transactions_per_request       ??      ??"
echo "  shared_store_transactions_per_request      ??      ??"
echo "  ga_island_kernel duration (ms, 100 gens)   ??      ??"
echo "  Wall time  (s, 500 gens)                   ??      ??"
echo "  Best tour length                           ??      ??"
echo "  (lengths should be ~equal: same RNG, same algorithm)"
echo ""
echo "Done. Results written to: $OUT"
