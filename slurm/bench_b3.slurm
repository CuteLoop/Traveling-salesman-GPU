#!/bin/bash
# bench_b3.slurm
# Benchmark: B3 — Thread-0 sort replacement.
# Compares THREE versions:
#   B2 (baseline):     O(P^2) selection sort by thread 0, 31 threads idle
#   B3-shuffle:        XOR-shuffle warp reduction (register-only, warp-synchronous)
#   B3-reduce:         Classical shared-memory binary tree reduction (__syncthreads)
#
# What this measures:
#   - smsp__warp_issue_stalled_barrier_per_warp_active.pct
#       High in B2 (31 threads stalled at __syncthreads after tid==0 sort)
#       Should drop in B3-shuffle and B3-reduce.
#   - Wall-clock time improvement from eliminating 63,488,000 serial comparisons.
#   - Comparison BETWEEN B3-shuffle and B3-reduce to understand __syncthreads cost.

#SBATCH --job-name=bench_b3_reduction
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:45:00
#SBATCH -o results/bench_b3_%j.out
#SBATCH -e results/bench_b3_%j.error

set -euo pipefail

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
    module load cuda11/11.8
else
    module load cuda11/11.0
fi

PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="/tmp/bench_b3_${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD_DIR" "${PROJ_DIR}/results"

TSP="${PROJ_DIR}/sequential/tests/fixtures/smoke_20.tsp"
ISLANDS=128
GENS=100
GENS_TIME=500
SEED=42

OUT="${PROJ_DIR}/results/bench_b3_${SLURM_JOB_ID:-local}.txt"
exec > >(tee -a "$OUT") 2>&1

echo "================================================================"
echo "  B3 Benchmark: Thread-0 Sort Elimination"
echo "  Comparing: B2 (baseline) vs B3-shuffle vs B3-reduce"
echo "  Job: ${SLURM_JOB_ID:-local}  Date: $(date)"
echo "================================================================"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "--- Building B2 (baseline for B3 comparison) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b2" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B2-bitmask.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo "--- Building B3-shuffle (XOR warp reduction) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b3_shuffle" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B3-shuffle.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo "--- Building B3-reduce (classical tree reduction) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b3_reduce" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B3-reduce.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo ""

# ── Static analysis: register count (shuffles use more regs) ─────────────────
echo "================================================================"
echo "  STATIC ANALYSIS: ptxas register counts"
echo "  Note: B3-shuffle may use slightly more registers due to shuffle"
echo "        operand pairs (my_len+my_idx = 2 extra regs per reduction)"
echo "================================================================"

for label in b2 b3_shuffle b3_reduce; do
    file="${BUILD_DIR}/ga_${label}"
    echo "--- ${label} ---"
    # Recompile with ptxas-verbose to capture register/smem report
    case "$label" in
        b2)         src="${PROJ_DIR}/src/cuda/variants/CUDA-GA-B2-bitmask.cu" ;;
        b3_shuffle) src="${PROJ_DIR}/src/cuda/variants/CUDA-GA-B3-shuffle.cu" ;;
        b3_reduce)  src="${PROJ_DIR}/src/cuda/variants/CUDA-GA-B3-reduce.cu" ;;
    esac
    nvcc -arch=sm_60 -O3 --ptxas-options=-v \
         -o /dev/null "$src" "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" -I"${PROJ_DIR}/src/cpp" 2>&1 \
        | grep -E "ga_island_kernel|lmem|smem|registers" || true
    echo ""
done

# ── nvprof: barrier stall percentage ─────────────────────────────────────────
echo "================================================================"
echo "  KEY METRIC: warp barrier stall percentage"
echo "  In B2, 31 threads stall at __syncthreads() while thread 0 sorts."
echo "  TARGET: stall % drops significantly in B3-shuffle and B3-reduce."
echo "  NOTE: ncu metrics below — may require SLURM GPU profiling permissions."
echo "================================================================"

for label in b2 b3_shuffle b3_reduce; do
    echo "--- ${label} barrier stall ---"
    nvprof --metrics warp_execution_efficiency \
           "${BUILD_DIR}/ga_${label}" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
        | grep -E "warp_execution_efficiency|Kernel|Error" || true
    echo ""
done

# ── nvprof: branch efficiency ─────────────────────────────────────────────────
echo "================================================================"
echo "  BRANCH EFFICIENCY"
echo "  B3-reduce introduces more branching (if tid < half) than B3-shuffle."
echo "================================================================"

for label in b2 b3_shuffle b3_reduce; do
    echo "--- ${label} branch efficiency ---"
    nvprof --metrics branch_efficiency \
           "${BUILD_DIR}/ga_${label}" "$TSP" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
        | grep -E "branch_efficiency|Kernel|Error" || true
    echo ""
done

# ── Wall-clock timing: all three versions ────────────────────────────────────
echo "================================================================"
echo "  WALL-CLOCK TIMING (${GENS_TIME} generations, 5 runs each)"
echo "  B2: baseline (tid==0 sort)"
echo "  B3-shuffle: XOR shuffle (no __syncthreads between steps)"
echo "  B3-reduce:  tree reduction (5 __syncthreads calls per pass)"
echo "================================================================"

echo "--- B2 timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b2" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B3-shuffle timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b3_shuffle" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B3-reduce timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b3_reduce" "$TSP" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""

# ── Correctness check: tour lengths should be similar ────────────────────────
echo "================================================================"
echo "  CORRECTNESS CHECK: single run with fixed seed"
echo "  Tour lengths will differ (parallel selection finds same quality elites"
echo "  but execution order within ties may vary), but should be in same range."
echo "================================================================"

echo "--- B2 single run ---"
"${BUILD_DIR}/ga_b2"          "$TSP" $ISLANDS 200 0.05 2 $SEED | grep "tour length"

echo "--- B3-shuffle single run ---"
"${BUILD_DIR}/ga_b3_shuffle"  "$TSP" $ISLANDS 200 0.05 2 $SEED | grep "tour length"

echo "--- B3-reduce single run ---"
"${BUILD_DIR}/ga_b3_reduce"   "$TSP" $ISLANDS 200 0.05 2 $SEED | grep "tour length"

echo ""
echo "================================================================"
echo "  SUMMARY TABLE (fill in from output above)"
echo "================================================================"
echo ""
echo "  Version       Registers  lmem  Sort ops   __syncthreads/gen  Wall time (s)"
echo "  B2            ??         0     496 serial  1                  ??"
echo "  B3-shuffle    ??         0     10 steps    1 (post-shuffle)   ??"
echo "  B3-reduce     ??         0     31 parallel 1+10 (5+5 passes)  ??"
echo ""
echo "  Expected: B3-shuffle <= B3-reduce <= B2 (wall time)"
echo "  Reasoning: shuffle has fewer __syncthreads and uses register file crossbar"
echo "             rather than shared memory for inter-thread communication."
echo ""
echo "Done. Results: $OUT"
