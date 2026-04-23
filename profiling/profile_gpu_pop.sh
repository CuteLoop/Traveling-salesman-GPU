#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: $0 <file.tsp> [islands] [generations] [mutation_rate] [elite_count] [seed]}"
ISLANDS="${2:-128}"
GENERATIONS="${3:-1000}"
MUTATION_RATE="${4:-0.05}"
ELITE_COUNT="${5:-2}"
SEED="${6:-12345}"
EXE="${EXE:-./tsp_gpu_pop}"
OUT_DIR="${OUT_DIR:-profiling/runs}"

mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${OUT_DIR}/gpu_pop_${STAMP}"

echo "Executable: ${EXE}"
echo "Input: ${INPUT}"
echo "Parameters: islands=${ISLANDS} generations=${GENERATIONS} mutation_rate=${MUTATION_RATE} elite_count=${ELITE_COUNT} seed=${SEED}"

echo
echo "== Plain timing run =="
/usr/bin/time -v "${EXE}" "${INPUT}" "${ISLANDS}" "${GENERATIONS}" "${MUTATION_RATE}" "${ELITE_COUNT}" "${SEED}" \
  2>&1 | tee "${BASE}_time.txt"

if command -v nsys >/dev/null 2>&1; then
  echo
  echo "== Nsight Systems =="
  nsys profile \
    --stats=true \
    --force-overwrite=true \
    --output="${BASE}_nsys" \
    "${EXE}" "${INPUT}" "${ISLANDS}" "${GENERATIONS}" "${MUTATION_RATE}" "${ELITE_COUNT}" "${SEED}" \
    2>&1 | tee "${BASE}_nsys.txt"
else
  echo
  echo "Skipping Nsight Systems: nsys not found"
fi

if command -v ncu >/dev/null 2>&1; then
  echo
  echo "== Nsight Compute basic =="
  ncu \
    --set basic \
    --target-processes all \
    --force-overwrite \
    --export "${BASE}_ncu" \
    "${EXE}" "${INPUT}" "${ISLANDS}" "${GENERATIONS}" "${MUTATION_RATE}" "${ELITE_COUNT}" "${SEED}" \
    2>&1 | tee "${BASE}_ncu.txt"
elif command -v nvprof >/dev/null 2>&1; then
  echo
  echo "== nvprof fallback =="
  nvprof "${EXE}" "${INPUT}" "${ISLANDS}" "${GENERATIONS}" "${MUTATION_RATE}" "${ELITE_COUNT}" "${SEED}" \
    2>&1 | tee "${BASE}_nvprof.txt"
else
  echo
  echo "Skipping kernel profiler: ncu and nvprof not found"
fi

echo
echo "Profile outputs written under ${OUT_DIR}"

