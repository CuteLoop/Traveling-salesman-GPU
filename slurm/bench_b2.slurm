#!/bin/bash
# bench_b2.slurm
# Benchmark: B2 Bitmask — eliminate local memory spill from OX crossover.
# Compares B1 (baseline for this test) vs B1+B2 (bitmask).
#
# What this measures:
#   - lmem in ptxas: 512 bytes (B1) -> 0 bytes (B2)
#   - local_load_transactions:  > 0 (B1) -> 0 (B2)
#   - local_store_transactions: > 0 (B1) -> 0 (B2)
#   - Wall-clock improvement from eliminating ~3.93 GB of hidden DRAM traffic

#SBATCH --job-name=bench_b2_bitmask
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:30:00
#SBATCH -o results/bench_b2_%j.out
#SBATCH -e results/bench_b2_%j.error

set -euo pipefail

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
    module load cuda11/11.8
else
    module load cuda11/11.0
fi

PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="/tmp/bench_b2_${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD_DIR" "${PROJ_DIR}/results"

TSP="${PROJ_DIR}/sequential/tests/fixtures/smoke_20.tsp"
ISLANDS=128
GENS=100
GENS_TIME=500
SEED=42

OUT="${PROJ_DIR}/results/bench_b2_${SLURM_JOB_ID:-local}.txt"
exec > >(tee -a "$OUT") 2>&1

echo "================================================================"
echo "  B2 Benchmark: Bitmask OX (local memory spill elimination)"
echo "  Job: ${SLURM_JOB_ID:-local}  Date: $(date)"
echo "================================================================"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "--- Building B1 (baseline for B2 comparison) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b1" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B1-stride.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo "--- Building B2 (B1 + bitmask OX) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b2" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B2-bitmask.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo ""

# ── Static analysis: lmem is the key signal ───────────────────────────────────
echo "================================================================"
echo "  STATIC ANALYSIS: ptxas lmem report"
echo "  KEY METRIC: lmem = local memory bytes per thread"
echo "  TARGET: lmem=512 (B1) -> lmem=0 (B2)"
echo "================================================================"

echo "--- B1 ptxas ---"
nvcc -arch=sm_60 -O3 --ptxas-options=-v \
     -o /dev/null \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B1-stride.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp" 2>&1 \
    | grep -E "ga_island_kernel|lmem|smem|registers" || true

echo ""
echo "--- B2 ptxas ---"
nvcc -arch=sm_60 -O3 --ptxas-options=-v \
     -o /dev/null \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B2-bitmask.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp" 2>&1 \
    | grep -E "ga_island_kernel|lmem|smem|registers" || true

echo ""
echo "Interpretation:"
echo "  lmem=512 means: 128 ints * 4 bytes = 512 bytes of DRAM traffic per thread per crossover"
echo "  lmem=0   means: all 'used' tracking lives in registers (no DRAM)"
echo ""

# ── nvprof: local memory transaction counts ───────────────────────────────────
echo "================================================================"
echo "  RUNTIME METRIC: local_load / local_store transactions"
echo "  TARGET: both -> 0 after bitmask fix"
echo "================================================================"

echo "--- B1 local memory transactions ---"
nvprof --metrics local_load_transactions,local_store_transactions \
       "${BUILD_DIR}/ga_b1" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "local_load_transactions|local_store_transactions|Kernel|Error" || true

echo ""
echo "--- B2 local memory transactions ---"
nvprof --metrics local_load_transactions,local_store_transactions \
       "${BUILD_DIR}/ga_b2" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "local_load_transactions|local_store_transactions|Kernel|Error" || true

echo ""

# ── Also check: shared bank conflicts still gone (B1 carried over) ────────────
echo "================================================================"
echo "  REGRESSION CHECK: shared bank conflicts still eliminated?"
echo "================================================================"
echo "--- B2 shared transactions ---"
nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request \
       "${BUILD_DIR}/ga_b2" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
    | grep -E "shared_.*_transactions|Kernel|Error" || true

echo ""

# ── Wall-clock timing ─────────────────────────────────────────────────────────
echo "================================================================"
echo "  WALL-CLOCK TIMING (${GENS_TIME} generations, 5 runs each)"
echo "================================================================"

echo "--- B1 timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b1" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B2 timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b2" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "================================================================"
echo "  CUMULATIVE SUMMARY: V0 -> B1 -> B2"
echo "================================================================"
echo "  Fix       Bottleneck removed          Expected lmem  Expected bank conflicts"
echo "  V0        (none)                      512 bytes      32-way"
echo "  B1        32-way bank conflicts        512 bytes       1-way (none)"
echo "  B2        local memory spill             0 bytes       1-way (none)"
echo ""
echo "Done. Results: $OUT"
