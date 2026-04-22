#!/usr/bin/env bash
# ===========================================================================
# benchmark.sh — Phase 10: Fixed-seed benchmark runner
#
# Compiles in release mode (-O3 -march=native -DNDEBUG) and runs ga-tsp
# across a fixed set of seeds, appending results to results/current.csv.
#
# Usage:
#   bash scripts/benchmark.sh [instance_file]
#
# Default instance: tests/fixtures/smoke_20.tsp
# ===========================================================================

set -euo pipefail

# ---- Configuration -------------------------------------------------------
SEEDS=(42 123 999 5555 9876)
POP=500
GEN=1000
ELITES=5
TK=2
PC=0.9
PM=0.1

INSTANCE="${1:-tests/fixtures/smoke_20.tsp}"
RESULTS_DIR="results"
OUTPUT_CSV="${RESULTS_DIR}/current.csv"
INSTANCE_NAME="$(basename "$INSTANCE")"

# ---- Build in release mode -----------------------------------------------
echo "=== Building in release mode ==="
make clean && make all BUILD=release
echo ""

# ---- Prepare output -------------------------------------------------------
mkdir -p "$RESULTS_DIR"
echo "instance,seed,population,generations,best_distance,elapsed_sec" > "$OUTPUT_CSV"

# ---- Run benchmarks -------------------------------------------------------
echo "=== Running benchmarks on ${INSTANCE_NAME} ==="
echo "  Seeds: ${SEEDS[*]}"
echo "  Config: pop=${POP} gen=${GEN} elites=${ELITES} tk=${TK} pc=${PC} pm=${PM}"
echo ""

for SEED in "${SEEDS[@]}"; do
    CSV_TMP=$(mktemp)

    START=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

    ./bin/ga-tsp --instance "$INSTANCE" \
                 --pop "$POP" --gen "$GEN" --seed "$SEED" \
                 --elites "$ELITES" --tk "$TK" --pc "$PC" --pm "$PM" \
                 --csv "$CSV_TMP" 2>&1 | tee /dev/stderr | grep -oP 'Best distance: \K[0-9.]+' > /tmp/_best_dist.txt

    END=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

    BEST_DIST=$(cat /tmp/_best_dist.txt)
    ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)

    echo "${INSTANCE_NAME},${SEED},${POP},${GEN},${BEST_DIST},${ELAPSED}" >> "$OUTPUT_CSV"
    echo "  Seed ${SEED}: best_distance=${BEST_DIST}  elapsed=${ELAPSED}s"

    rm -f "$CSV_TMP"
done

echo ""
echo "=== Results written to ${OUTPUT_CSV} ==="
cat "$OUTPUT_CSV"
