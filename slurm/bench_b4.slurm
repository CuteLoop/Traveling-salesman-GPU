#!/bin/bash
# bench_b4.slurm
# Benchmark: B4 — Distance matrix memory placement.
# Compares THREE versions:
#   B3-shuffle (baseline): dist in __constant__ memory (8 KB broadcast cache)
#   B4-global:             dist in global memory, __restrict__ pointer (L2-cached)
#   B4-smem:               dist in shared memory; n <= 99 only (fastest possible)
#
# What this measures:
#   - l2_read_hit_rate
#       B4-global should show a high L2 hit rate after warm-up tours.
#       B3-shuffle may show a high constant cache hit rate but serialized delivery.
#   - l2_read_transactions
#       B4-global issues parallel requests; B3-shuffle serializes 32 per warp step.
#   - Wall-clock time: the bottom-line improvement from eliminating constant-cache
#       serialization and (for B4-smem) all off-chip dist traffic entirely.
#
# Constraint reminder:
#   B4-smem is only valid for n <= SMEM_DIST_MAX_N = 99.
#   For n=128 (the production case), only B3-shuffle and B4-global are comparable.
#   This script runs n=20 (smoke) for all three and n=128 for B3-shuffle vs B4-global.

#SBATCH --job-name=bench_b4_distmem
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:50:00
#SBATCH -o results/bench_b4_%j.out
#SBATCH -e results/bench_b4_%j.error

set -euo pipefail

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
    module load cuda11/11.8
else
    module load cuda11/11.0
fi

PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="/tmp/bench_b4_${SLURM_JOB_ID:-local}"
mkdir -p "$BUILD_DIR" "${PROJ_DIR}/results"

# ── Fixtures ──────────────────────────────────────────────────────────────────
TSP_SMOKE="${PROJ_DIR}/sequential/tests/fixtures/smoke_20.tsp"
# For worst-case n=128 measurement swap the line below:
# TSP_128="${PROJ_DIR}/sequential/tests/fixtures/128city.tsp"
TSP_128="$TSP_SMOKE"   # replace with your 128-city file on the cluster

ISLANDS=128
GENS=100
GENS_TIME=500
SEED=42

OUT="${PROJ_DIR}/results/bench_b4_${SLURM_JOB_ID:-local}.txt"
exec > >(tee -a "$OUT") 2>&1

echo "================================================================"
echo "  B4 Benchmark: Distance Matrix Memory Placement"
echo "  Comparing: B3-shuffle (const mem) vs B4-global vs B4-smem"
echo "  Job: ${SLURM_JOB_ID:-local}  Date: $(date)"
echo "================================================================"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "--- Building B3-shuffle (baseline: __constant__ memory scatter) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b3_shuffle" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B3-shuffle.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo "--- Building B4-global (global memory + __restrict__, L2-cached) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b4_global" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B4-global.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo "--- Building B4-smem (shared memory dist; enforces n <= 99) ---"
nvcc -arch=sm_60 -O3 -lineinfo \
     -o "${BUILD_DIR}/ga_b4_smem" \
    "${PROJ_DIR}/src/cuda/variants/CUDA-GA-B4-smem.cu" \
    "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" \
    -I"${PROJ_DIR}/src/cpp"

echo ""

# ── Static analysis: shared memory and register footprint ────────────────────
echo "================================================================"
echo "  STATIC ANALYSIS: ptxas resource report"
echo "  B4-smem should show higher smem usage (dist added to shared memory)."
echo "  B4-global should show identical smem to B3-shuffle."
echo "================================================================"

for label in b3_shuffle b4_global b4_smem; do
    case "$label" in
        b3_shuffle) src="${PROJ_DIR}/src/cuda/variants/CUDA-GA-B3-shuffle.cu" ;;
        b4_global)  src="${PROJ_DIR}/src/cuda/variants/CUDA-GA-B4-global.cu" ;;
        b4_smem)    src="${PROJ_DIR}/src/cuda/variants/CUDA-GA-B4-smem.cu" ;;
    esac
    echo "--- ${label} ---"
    nvcc -arch=sm_60 -O3 --ptxas-options=-v \
         -o /dev/null "$src" "${PROJ_DIR}/src/cpp/tsplib_parser.cpp" -I"${PROJ_DIR}/src/cpp" 2>&1 \
        | grep -E "ga_island_kernel|lmem|smem|registers" || true
    echo ""
done

# ── nvprof: L2 read hit rate ──────────────────────────────────────────────────
echo "================================================================"
echo "  KEY METRIC: L2 read hit rate"
echo "  B4-global: dist (64 KB) fits easily in L2 (4 MB on P100)."
echo "  After ~5-10 warm-up tours, all subsequent dist reads should be L2 hits."
echo "  B3-shuffle: constant cache is 8 KB per SM. dist = 64 KB."
echo "             Only 1/8 of the matrix fits; evictions cause DRAM traffic."
echo "  B4-smem: no L2 traffic for dist at all (shared memory is on-chip)."
echo "================================================================"

for label in b3_shuffle b4_global b4_smem; do
    echo "--- ${label} L2 read hit rate ---"
    nvprof --metrics l2_read_hit_rate \
           "${BUILD_DIR}/ga_${label}" "$TSP_SMOKE" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
        | grep -E "l2_read_hit_rate|Kernel|Error" || true
    echo ""
done

# ── nvprof: L2 read transactions ──────────────────────────────────────────────
echo "================================================================"
echo "  L2 READ TRANSACTIONS"
echo "  Constant cache misses -> L2 transactions (serialized)."
echo "  Global memory with __restrict__ -> parallel in-flight L2 requests."
echo "  Shared memory -> zero L2 transactions for dist reads."
echo "================================================================"

for label in b3_shuffle b4_global b4_smem; do
    echo "--- ${label} L2 read transactions ---"
    nvprof --metrics l2_read_transactions \
           "${BUILD_DIR}/ga_${label}" "$TSP_SMOKE" $ISLANDS $GENS 0.05 2 $SEED 2>&1 \
        | grep -E "l2_read_transactions|Kernel|Error" || true
    echo ""
done

# ── Wall-clock timing: smoke (n=20), all three versions ──────────────────────
echo "================================================================"
echo "  WALL-CLOCK TIMING — smoke fixture n=20 (${GENS_TIME} generations, 5 runs)"
echo "  All three variants valid for n=20."
echo "================================================================"

echo "--- B3-shuffle (const mem) timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b3_shuffle" "$TSP_SMOKE" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B4-global timing ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b4_global" "$TSP_SMOKE" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B4-smem timing (n=20 fits in shared memory; SMEM_DIST_MAX_N=99) ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b4_smem" "$TSP_SMOKE" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""

# ── Wall-clock timing: production n=128, B3-shuffle vs B4-global only ─────────
echo "================================================================"
echo "  WALL-CLOCK TIMING — n=128 (production) ONLY for B3-shuffle vs B4-global"
echo "  B4-smem EXCLUDED: 128-city matrix does not fit in shared memory."
echo "  (swap TSP_128 above to point to your 128-city TSP fixture)"
echo "================================================================"

echo "--- B3-shuffle (const mem), n=128 ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b3_shuffle" "$TSP_128" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""
echo "--- B4-global, n=128 ---"
for run in 1 2 3 4 5; do
    /usr/bin/time -f "run ${run}: %e sec wall" \
        "${BUILD_DIR}/ga_b4_global" "$TSP_128" $ISLANDS $GENS_TIME 0.05 2 $SEED \
        2>&1 | grep "run\|tour length"
done

echo ""

# ── Correctness check ─────────────────────────────────────────────────────────
echo "================================================================"
echo "  CORRECTNESS CHECK: single run, fixed seed"
echo "  All three must agree on tour quality (same algorithm, same RNG seed)."
echo "  Exact tour lengths may differ by a small margin due to shared memory"
echo "  layout differences affecting tie-breaking in selection, but should"
echo "  fall within the same order of magnitude."
echo "================================================================"

echo "--- B3-shuffle single run ---"
"${BUILD_DIR}/ga_b3_shuffle" "$TSP_SMOKE" $ISLANDS 200 0.05 2 $SEED | grep "tour length"

echo "--- B4-global single run ---"
"${BUILD_DIR}/ga_b4_global"  "$TSP_SMOKE" $ISLANDS 200 0.05 2 $SEED | grep "tour length"

echo "--- B4-smem single run ---"
"${BUILD_DIR}/ga_b4_smem"    "$TSP_SMOKE" $ISLANDS 200 0.05 2 $SEED | grep "tour length"

echo ""
echo "================================================================"
echo "  SUMMARY TABLE (fill in from output above)"
echo "================================================================"
echo ""
echo "  Version        smem dist    L2 hit rate   L2 read txns   Wall time (s)"
echo "  B3-shuffle     const cache  ??%           ??             ??"
echo "  B4-global      L2 (4 MB)   ??%           ??             ??"
echo "  B4-smem        on-chip      N/A           ~0             ??"
echo ""
echo "  Expected ranking (n=128): B4-global < B3-shuffle (wall time)"
echo "  Expected ranking (n<=99): B4-smem < B4-global < B3-shuffle (wall time)"
echo "  Reasoning: serialized const cache < parallel L2 < on-chip shared memory"
echo ""
echo "Done. Results: $OUT"
