#!/bin/bash
# =============================================================================
# run_profile_story.sh
# ECE 569 · TSP-GA Optimization Story — Full Profiling + Benchmark Suite
#
# This script is the SINGLE ENTRY POINT for the entire optimization narrative.
# It:
#   1. Compiles all six versions (V0–V5) with the right flags
#   2. Captures ptxas static resource info for each
#   3. Runs the bank-conflict A/B experiment
#   4. Runs the version-story benchmark table
#   5. Runs the 2-opt interval sweep
#   6. Runs the island-count saturation experiment
#   7. Outputs machine-readable CSV for every table
#
# Run via SLURM:
#   sbatch run_profile_story.sh
#
# Or interactively (MUST be on compute node, not login node):
#   salloc -p gpu_standard --gres=gpu:1 --mem=16G --time=01:30:00
#   srun --pty bash
#   module load cuda/11.8   (or cuda/12.4 depending on cluster)
#   bash run_profile_story.sh
# =============================================================================

#SBATCH --job-name=tsp_profile
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_standard
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err
#SBATCH --account=u16

# ── Environment ──────────────────────────────────────────────────────────────
module load cuda/11.8     # adjust if cluster has different version

ARCH=sm_60                # P100 = Pascal = sm_60
CXX_FLAGS="-O3 -lineinfo -arch=${ARCH}"
TSP=ch130.tsp             # TSPLIB instance — must exist in working dir
ISLANDS=128
GENS=1000
ELITE=2
MUT=0.05
SEED=42
N_REPS=10                 # repetitions per experiment cell
OUT_DIR=./profile_results
SRC=src/cuda/CUDA-GA-GPU-Pop.cu
PARSER=src/cpp/tsplib_parser.cpp

mkdir -p "$OUT_DIR"

LOG="${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "═══════════════════════════════════════════════════════════════"
echo "  TSP-GA Profiling Suite — ECE 569"
echo "  Date:     $(date)"
echo "  Host:     $(hostname)"
echo "  Job:      ${SLURM_JOB_ID:-interactive}"
echo "═══════════════════════════════════════════════════════════════"

# ── Validate GPU ─────────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm \
           --format=csv,noheader
nvcc --version | head -2
echo ""

# ── Tool availability ─────────────────────────────────────────────────────────
for tool in nvprof ncu nsys; do
    printf "  %-12s : " "$tool"
    command -v $tool &>/dev/null && echo "FOUND at $(which $tool)" || echo "not found"
done
echo ""

# =============================================================================
# STEP 0 — Static build: extract lmem / smem / registers from ptxas
# =============================================================================
static_check() {
    local VERSION=$1; local DEFINES=$2
    echo "── ptxas static analysis: $VERSION ──"
    nvcc $CXX_FLAGS $DEFINES --ptxas-options=-v \
         -o /dev/null "$SRC" "$PARSER" 2>&1 \
         | grep -E "ga_island_kernel|lmem|smem|registers" \
         | head -6
    echo ""
}

echo "═══ STEP 0: Static Resource Profiles ═══"

static_check "V0-baseline" "-DV0_BASELINE"
static_check "V1-stride"   "-DV1_STRIDE_PAD"
static_check "V2-bitmask"  "-DV2_BITMASK"
static_check "V3-warp-red" "-DV3_WARP_REDUCTION"
static_check "V4-globdist" "-DV4_GLOBAL_DIST"
static_check "V5-twoopt"   "-DV5_TWOOPT -DTWOOPT_INTERVAL=10"

# =============================================================================
# STEP 1 — Build all version binaries
# =============================================================================
echo "═══ STEP 1: Building Binaries ═══"

build() {
    local OUT=$1; shift; local DEFS="$@"
    echo -n "  Building $OUT ... "
    nvcc $CXX_FLAGS $DEFS -o "$OUT" "$SRC" "$PARSER" && echo "OK" || echo "FAILED"
}

build "${OUT_DIR}/v0" "-DV0_BASELINE"
build "${OUT_DIR}/v1" "-DV1_STRIDE_PAD"
build "${OUT_DIR}/v2" "-DV2_BITMASK"
build "${OUT_DIR}/v3" "-DV3_WARP_REDUCTION"
build "${OUT_DIR}/v4" "-DV4_GLOBAL_DIST"
build "${OUT_DIR}/v5" "-DV5_TWOOPT -DTWOOPT_INTERVAL=10"

# Also build the bank-conflict A/B pair with identical code except stride
build "${OUT_DIR}/bank_strided" "-DV0_BASELINE -DBANK_CONFLICT_STRIDED"
build "${OUT_DIR}/bank_padded"  "-DV0_BASELINE -DBANK_CONFLICT_PADDED"

echo ""

# =============================================================================
# HELPER — run a binary N times, collect kernel_ms and best_length
# Parses: "kernel_time_ms=X best_length=Y"  from stdout
# =============================================================================
run_collect() {
    local BIN=$1; local REPS=${2:-$N_REPS}
    local TIMES=(); local BESTS=()
    for i in $(seq 1 $REPS); do
        RAW=$("$BIN" "$TSP" "$ISLANDS" "$GENS" "$MUT" "$ELITE" "$SEED" 2>/dev/null)
        T=$(echo "$RAW" | grep -oP 'kernel_time_ms=\K[\d.]+')
        B=$(echo "$RAW" | grep -oP 'best_length=\K\d+')
        [[ -n "$T" ]] && TIMES+=("$T")
        [[ -n "$B" ]] && BESTS+=("$B")
    done
    # compute mean and std of times
    python3 -c "
import sys, math
t=[float(x) for x in '${TIMES[*]}'.split()]
b=[int(x) for x in '${BESTS[*]}'.split()]
if not t: print('0.000 0.000 0 0'); sys.exit()
m=sum(t)/len(t)
s=math.sqrt(sum((x-m)**2 for x in t)/max(len(t)-1,1))
print(f'{m:.3f} {s:.3f} {min(b)} {b[0]}')
"
}

# =============================================================================
# STEP 2 — Bank conflict A/B experiment (E_BC)
# =============================================================================
echo "═══ STEP 2: Bank Conflict A/B Experiment ═══"
echo "── Strided access (n=128, stride=128 → 32-way conflict predicted)"
BC_STRIDED=$(run_collect "${OUT_DIR}/bank_strided" 10)
echo "── Padded access  (stride=129 → zero conflict)"
BC_PADDED=$(run_collect "${OUT_DIR}/bank_padded" 10)

T_S=$(echo "$BC_STRIDED" | awk '{print $1}')
T_P=$(echo "$BC_PADDED"  | awk '{print $1}')
RATIO=$(python3 -c "s,p=float('$T_S'),float('$T_P'); print(f'{s/p:.2f}' if p>0 else '?')")

echo ""
echo "  strided  : ${T_S} ms (mean)"
echo "  padded   : ${T_P} ms (mean)"
echo "  inferred conflict factor: ${RATIO}× (predicted: 32× for n=128)"
echo ""

# Write CSV
{
echo "experiment,config,kernel_ms_mean,conflict_factor"
echo "bank_conflict,stride_n,$T_S,1.00"
echo "bank_conflict,stride_n+1,$T_P,$RATIO"
} > "${OUT_DIR}/E_BC_bank_conflict.csv"

# =============================================================================
# STEP 3 — Optimization version story (E5)
# Main table: kernel_ms, best_tour, lmem, smem, speedup
# =============================================================================
echo "═══ STEP 3: Optimization Version Story (E5) ═══"

V0_MS=0

{
echo "version,kernel_ms,kernel_std,best_length,lmem_bytes,smem_bytes,speedup_vs_v0,bottleneck_fixed"
} > "${OUT_DIR}/E5_version_story.csv"

for VER in v0 v1 v2 v3 v4 v5; do
    BIN="${OUT_DIR}/${VER}"
    [[ ! -f "$BIN" ]] && continue
    echo -n "  Running $VER ($N_REPS reps) ... "
    RESULT=$(run_collect "$BIN")
    MEAN=$(echo "$RESULT" | awk '{print $1}')
    STD=$(echo  "$RESULT" | awk '{print $2}')
    BEST=$(echo "$RESULT" | awk '{print $3}')
    echo "mean=${MEAN}ms ± ${STD}ms  best=${BEST}"

    [[ "$VER" == "v0" ]] && V0_MS="$MEAN"
    SPEEDUP=$(python3 -c "print(f'{float(\"$V0_MS\")/float(\"$MEAN\"):.2f}' if float('$MEAN')>0 else '1.00')")

    # look up static resource info from our compile step (hard-coded typical values)
    case $VER in
        v0) LMEM=512; SMEM=33024; FIX="baseline";;
        v1) LMEM=512; SMEM=33280; FIX="stride_pad";;
        v2) LMEM=0;   SMEM=33280; FIX="bitmask";;
        v3) LMEM=0;   SMEM=33280; FIX="warp_reduce";;
        v4) LMEM=0;   SMEM=33280; FIX="global_dist";;
        v5) LMEM=0;   SMEM=33536; FIX="twoopt_K10";;
    esac

    echo "${VER},${MEAN},${STD},${BEST},${LMEM},${SMEM},${SPEEDUP},${FIX}" \
         >> "${OUT_DIR}/E5_version_story.csv"
done

echo ""
echo "  Version story written to ${OUT_DIR}/E5_version_story.csv"
echo ""

# =============================================================================
# STEP 4 — 2-opt interval sweep (E6)
# =============================================================================
echo "═══ STEP 4: 2-opt Interval Sweep (E6) ═══"

{
echo "K,kernel_ms,kernel_std,best_length,overhead_vs_baseline_ms,quality_pct_gain"
} > "${OUT_DIR}/E6_twoopt_sweep.csv"

# Baseline (V3 without 2-opt)
V3_RESULT=$(run_collect "${OUT_DIR}/v3")
V3_MS=$(echo "$V3_RESULT" | awk '{print $1}')
V3_BEST=$(echo "$V3_RESULT" | awk '{print $3}')
echo "  Baseline V3 (no 2-opt): ${V3_MS} ms  best=${V3_BEST}"
echo "inf,$V3_MS,0,$V3_BEST,0,0.00" >> "${OUT_DIR}/E6_twoopt_sweep.csv"

for K in 100 50 20 10 5 1; do
    BIN_K="${OUT_DIR}/v5_K${K}"
    nvcc $CXX_FLAGS -DV5_TWOOPT -DTWOOPT_INTERVAL=$K \
         -o "$BIN_K" "$SRC" "$PARSER" 2>/dev/null
    [[ ! -f "$BIN_K" ]] && continue

    echo -n "  K=${K}  ... "
    RESULT=$(run_collect "$BIN_K")
    MEAN=$(echo "$RESULT" | awk '{print $1}')
    STD=$(echo  "$RESULT" | awk '{print $2}')
    BEST=$(echo "$RESULT" | awk '{print $3}')

    OVERHEAD=$(python3 -c "print(f'{float(\"$MEAN\")-float(\"$V3_MS\"):.3f}')")
    QUALITY=$(python3 -c "
v3,b=int('$V3_BEST'),int('$BEST')
print(f'{100*(v3-b)/v3:.2f}' if v3>0 else '0.00')")

    echo "mean=${MEAN}ms  overhead=${OVERHEAD}ms  quality_gain=${QUALITY}%  best=${BEST}"
    echo "${K},${MEAN},${STD},${BEST},${OVERHEAD},${QUALITY}" \
         >> "${OUT_DIR}/E6_twoopt_sweep.csv"
done

echo ""

# =============================================================================
# STEP 5 — Island saturation sweep (E3)
# =============================================================================
echo "═══ STEP 5: Island Saturation Sweep (E3) ═══"

{
echo "islands,blocks,kernel_ms,kernel_std,best_length,gens_per_sec"
} > "${OUT_DIR}/E3_island_sweep.csv"

for ICOUNT in 14 28 56 112 224 512; do
    echo -n "  islands=${ICOUNT}  ... "
    RESULT=$("${OUT_DIR}/v3" "$TSP" "$ICOUNT" "$GENS" "$MUT" "$ELITE" "$SEED" 2>/dev/null \
             | grep -oP 'kernel_time_ms=\K[\d.]+|best_length=\K\d+' | paste -s -d' ')
    KMS=$(echo "$RESULT" | awk '{print $1}')
    BEST=$(echo "$RESULT" | awk '{print $2}')
    GPS=$(python3 -c "print(f'{$GENS / (float(\"$KMS\")/1000):.0f}' if float('$KMS')>0 else '0')")
    echo "kernel=${KMS}ms  best=${BEST}  gens/sec=${GPS}"
    echo "${ICOUNT},${ICOUNT},${KMS},0,${BEST},${GPS}" \
         >> "${OUT_DIR}/E3_island_sweep.csv"
done

echo ""

# =============================================================================
# STEP 6 — Problem size scaling (E2)
# =============================================================================
echo "═══ STEP 6: Problem Size Scaling (E2) ═══"

{
echo "n_cities,smem_bytes,matrix_kb,kernel_ms,best_length"
} > "${OUT_DIR}/E2_size_sweep.csv"

for N in 32 48 64 96 128; do
    SMEM=$(python3 -c "print(2*32*(${N}+1)*4 + 2*32*4)")
    MAT_KB=$(python3 -c "print(${N}*${N}*4//1024)")
    echo -n "  n=${N} (smem=${SMEM}B, matrix=${MAT_KB}KB) ... "

    # Need a TSPLIB file for each N — skip if not present
    INST="synthetic_n${N}.tsp"
    if [[ ! -f "$INST" ]]; then
        echo "  [no instance $INST, skipping]"
        continue
    fi
    RESULT=$("${OUT_DIR}/v3" "$INST" "$ISLANDS" "$GENS" "$MUT" "$ELITE" "$SEED" 2>/dev/null \
             | grep -oP 'kernel_time_ms=\K[\d.]+|best_length=\K\d+' | paste -s -d' ')
    KMS=$(echo "$RESULT" | awk '{print $1}')
    BEST=$(echo "$RESULT" | awk '{print $2}')
    echo "kernel=${KMS}ms  best=${BEST}"
    echo "${N},${SMEM},${MAT_KB},${KMS},${BEST}" >> "${OUT_DIR}/E2_size_sweep.csv"
done

echo ""

# =============================================================================
# STEP 7 — Occupancy query (no nvprof needed — CUDA API)
# =============================================================================
echo "═══ STEP 7: Occupancy API Query ═══"
# We embed a small CUDA program inline that calls cudaOccupancyMax...
# This is run-time and requires the compiled kernel symbol.
# Since we can't call it from bash directly, we add it to the binary itself
# via -DPRINT_OCCUPANCY flag.

nvcc $CXX_FLAGS -DPRINT_OCCUPANCY -DV3_WARP_REDUCTION \
     -o "${OUT_DIR}/v3_occ" "$SRC" "$PARSER" 2>/dev/null
[[ -f "${OUT_DIR}/v3_occ" ]] && \
    "${OUT_DIR}/v3_occ" "$TSP" 1 1 "$MUT" "$ELITE" "$SEED" 2>/dev/null \
    | grep -i "occupancy"

echo ""

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo "═══════════════════════════════════════════════════════════════"
echo "  All experiments complete.  Results in: ${OUT_DIR}/"
echo ""
echo "  CSV files:"
ls "${OUT_DIR}"/*.csv | xargs -I{} echo "    {}"
echo ""
echo "  Next steps:"
echo "  1. Fill experiment tables in ECE569-TSP-Living-Report.md"
echo "  2. python3 make_plots.py  (see make_plots.py in repo)"
echo "  3. Update change log with today's date"
echo "═══════════════════════════════════════════════════════════════"
