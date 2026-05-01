#!/bin/bash
#SBATCH --job-name=tsp_cuda_all
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:45:00
#SBATCH -o run_cuda_all_variants_%j.out
#SBATCH -e run_cuda_all_variants_%j.error

set -euo pipefail

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
  module load cuda11/11.8
else
  echo "Warning: module cuda11/11.8 not found, falling back to cuda11/11.0"
  module load cuda11/11.0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/Makefile" ]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
fi
cd "$REPO_DIR"

DATASET="${DATASET:-sequential/tests/fixtures/smoke_20.tsp}"
if [ ! -f "$DATASET" ]; then
  echo "Error: dataset '$DATASET' not found from repo root: $REPO_DIR"
  exit 1
fi

ISLANDS="${ISLANDS:-128}"
GENERATIONS="${GENERATIONS:-1000}"
MUTATION="${MUTATION:-0.05}"
ELITE_POP="${ELITE_POP:-2}"
ELITE_HYBRID="${ELITE_HYBRID:-4}"
SEED="${SEED:-42}"
RUNS="${RUNS:-5}"

mkdir -p results

if [ ! -x ./build/CUDA-GA-B4-smem ]; then
  echo "Building all CUDA versions..."
  make all_cuda_versions
fi

DATASET_TAG="$(basename "$DATASET" .tsp)"
SUMMARY_CSV="results/cuda_all_variants_summary_${DATASET_TAG}_${SLURM_JOB_ID}.csv"
AGGREGATE_CSV="results/cuda_all_variants_aggregate_${DATASET_TAG}_${SLURM_JOB_ID}.csv"

csv_escape() {
  local s="$1"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

write_summary_header() {
  cat > "$SUMMARY_CSV" <<EOF
job_id,timestamp,dataset,implementation,repeat_id,seed,executable,command,status,exit_code,runtime_sec,best_length,output_txt,tour_csv
EOF
}

write_aggregate_header() {
  cat > "$AGGREGATE_CSV" <<EOF
job_id,dataset,implementation,runs,successful_runs,mean_runtime_sec,stddev_runtime_sec,mean_best_length,stddev_best_length,min_best_length,max_best_length,best_run_repeat_id,best_run_output_txt,best_run_tour_csv
EOF
}

extract_best_length() {
  local out_file="$1"
  grep -E 'Best (GA tour length|GPU-population GA tour length|tour length)|Tour length' "$out_file" \
    | tail -n 1 \
    | grep -Eo '[0-9]+' \
    | tail -n 1 || true
}

extract_tour_line() {
  local out_file="$1"
  awk '
    /Best tour \(0-based indices\):|Tour \(0-based indices\):/ { capture=1; next }
    capture {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      if ($0 ~ /^[0-9][0-9 ]*$/) {
        print $0
        exit
      }
    }
  ' "$out_file"
}

write_tour_csv() {
  local tour_line="$1"
  local tour_csv="$2"
  local index=0

  echo "order,city" > "$tour_csv"
  for city in $tour_line; do
    echo "$index,$city" >> "$tour_csv"
    index=$((index + 1))
  done
}

append_summary_row() {
  local implementation="$1"
  local repeat_id="$2"
  local seed="$3"
  local executable="$4"
  local command="$5"
  local status="$6"
  local exit_code="$7"
  local runtime_sec="$8"
  local best_length="$9"
  local output_txt="${10}"
  local tour_csv="${11}"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  {
    csv_escape "${SLURM_JOB_ID:-local}"
    printf ','
    csv_escape "$timestamp"
    printf ','
    csv_escape "$DATASET"
    printf ','
    csv_escape "$implementation"
    printf ','
    csv_escape "$repeat_id"
    printf ','
    csv_escape "$seed"
    printf ','
    csv_escape "$executable"
    printf ','
    csv_escape "$command"
    printf ','
    csv_escape "$status"
    printf ','
    csv_escape "$exit_code"
    printf ','
    csv_escape "$runtime_sec"
    printf ','
    csv_escape "$best_length"
    printf ','
    csv_escape "$output_txt"
    printf ','
    csv_escape "$tour_csv"
    printf '\n'
  } >> "$SUMMARY_CSV"
}

append_aggregate_row() {
  local implementation="$1"
  local runs="$2"
  local successful_runs="$3"
  local mean_runtime_sec="$4"
  local stddev_runtime_sec="$5"
  local mean_best_length="$6"
  local stddev_best_length="$7"
  local min_best_length="$8"
  local max_best_length="$9"
  local best_run_repeat_id="${10}"
  local best_run_output_txt="${11}"
  local best_run_tour_csv="${12}"

  {
    csv_escape "${SLURM_JOB_ID:-local}"
    printf ','
    csv_escape "$DATASET"
    printf ','
    csv_escape "$implementation"
    printf ','
    csv_escape "$runs"
    printf ','
    csv_escape "$successful_runs"
    printf ','
    csv_escape "$mean_runtime_sec"
    printf ','
    csv_escape "$stddev_runtime_sec"
    printf ','
    csv_escape "$mean_best_length"
    printf ','
    csv_escape "$stddev_best_length"
    printf ','
    csv_escape "$min_best_length"
    printf ','
    csv_escape "$max_best_length"
    printf ','
    csv_escape "$best_run_repeat_id"
    printf ','
    csv_escape "$best_run_output_txt"
    printf ','
    csv_escape "$best_run_tour_csv"
    printf '\n'
  } >> "$AGGREGATE_CSV"
}

aggregate_variant_rows() {
  local implementation="$1"
  python3 - "$SUMMARY_CSV" "$implementation" <<'PY'
import csv
import math
import statistics
import sys

summary_csv = sys.argv[1]
implementation = sys.argv[2]

with open(summary_csv, newline='', encoding='utf-8') as handle:
    rows = [row for row in csv.DictReader(handle) if row['implementation'] == implementation]

if not rows:
    raise SystemExit(1)

success_rows = [
    row for row in rows
    if row['status'] == 'ok' and row['exit_code'] == '0' and row['runtime_sec'] and row['best_length']
]

def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

mean_runtime = None
std_runtime = None
mean_best = None
std_best = None
min_best = None
max_best = None
best_repeat = ''
best_output = ''
best_tour = ''

if success_rows:
    runtimes = [float(row['runtime_sec']) for row in success_rows]
    bests = [float(row['best_length']) for row in success_rows]
    mean_runtime = statistics.mean(runtimes)
    std_runtime = statistics.pstdev(runtimes)
    mean_best = statistics.mean(bests)
    std_best = statistics.pstdev(bests)
    min_best = min(bests)
    max_best = max(bests)
    best_row = min(success_rows, key=lambda row: (float(row['best_length']), int(row['repeat_id'])))
    best_repeat = best_row['repeat_id']
    best_output = best_row['output_txt']
    best_tour = best_row['tour_csv']

print("\t".join([
    str(len(rows)),
    str(len(success_rows)),
    fmt(mean_runtime),
    fmt(std_runtime),
    fmt(mean_best),
    fmt(std_best),
    fmt(min_best),
    fmt(max_best),
    best_repeat,
    best_output,
    best_tour,
]))
PY
}

write_summary_header
write_aggregate_header

run_variant() {
  local name="$1"
  shift
  local executable="$1"
  local command="$*"
  local repeat_id
  local out
  local runtime_file
  local tour_csv
  local runtime_sec
  local best_length
  local tour_line
  local status
  local exit_code
  local aggregate
  local runs
  local successful_runs
  local mean_runtime_sec
  local stddev_runtime_sec
  local mean_best_length
  local stddev_best_length
  local min_best_length
  local max_best_length
  local best_run_repeat_id
  local best_run_output_txt
  local best_run_tour_csv

  echo "== Running ${name} on ${DATASET} for ${RUNS} run(s) =="
  for repeat_id in $(seq 1 "$RUNS"); do
    out="results/${name}_${DATASET_TAG}_${SLURM_JOB_ID}_r${repeat_id}.txt"
    runtime_file="results/${name}_${DATASET_TAG}_${SLURM_JOB_ID}_r${repeat_id}.runtime"
    tour_csv="results/${name}_${DATASET_TAG}_${SLURM_JOB_ID}_r${repeat_id}_best_tour.csv"
    runtime_sec=""
    best_length=""
    tour_line=""
    status="ok"
    exit_code=0

    echo "-- ${name} repeat ${repeat_id}/${RUNS} --"
    set +e
    /usr/bin/time -f "%e" -o "$runtime_file" "$@" > "$out" 2>&1
    exit_code=$?
    set -e

    if [ "$exit_code" -ne 0 ]; then
      status="failed"
    fi

    if [ -f "$runtime_file" ]; then
      runtime_sec="$(tr -d '[:space:]' < "$runtime_file")"
      rm -f "$runtime_file"
    fi

    best_length="$(extract_best_length "$out")"
    tour_line="$(extract_tour_line "$out")"
    if [ -n "$tour_line" ]; then
      write_tour_csv "$tour_line" "$tour_csv"
    else
      tour_csv=""
    fi

    append_summary_row "$name" "$repeat_id" "$SEED" "$executable" "$command" "$status" "$exit_code" "$runtime_sec" "$best_length" "$out" "$tour_csv"

    echo "Wrote ${out}"
    if [ -n "$tour_csv" ]; then
      echo "Wrote ${tour_csv}"
    else
      echo "No tour exported for ${name} repeat ${repeat_id}; could not parse a tour line from output."
    fi

    if [ "$exit_code" -ne 0 ]; then
      echo "Warning: ${name} repeat ${repeat_id} exited with code ${exit_code}."
    fi
  done

  aggregate="$(aggregate_variant_rows "$name")"
  IFS=$'\t' read -r runs successful_runs mean_runtime_sec stddev_runtime_sec mean_best_length stddev_best_length min_best_length max_best_length best_run_repeat_id best_run_output_txt best_run_tour_csv <<< "$aggregate"
  append_aggregate_row "$name" "$runs" "$successful_runs" "$mean_runtime_sec" "$stddev_runtime_sec" "$mean_best_length" "$stddev_best_length" "$min_best_length" "$max_best_length" "$best_run_repeat_id" "$best_run_output_txt" "$best_run_tour_csv"
}

run_variant gpu_naive ./build/GPU-Naive "$DATASET"
run_variant cuda_ga ./build/CUDA-GA "$DATASET" 512 "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" "$SEED"
run_variant cuda_ga_gpu_pop ./build/CUDA-GA-GPU-Pop "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
run_variant cuda_ga_gpu_pop_bankconflict ./build/CUDA-GA-GPU-Pop-bankconflict "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
run_variant cuda_ga_gpu_pop_bitset ./build/CUDA-GA-GPU-Pop-bitset "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"

run_variant cuda_ga_b1_stride ./build/CUDA-GA-B1-stride "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
run_variant cuda_ga_b2_bitmask ./build/CUDA-GA-B2-bitmask "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
run_variant cuda_ga_b3_reduce ./build/CUDA-GA-B3-reduce "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
run_variant cuda_ga_b3_shuffle ./build/CUDA-GA-B3-shuffle "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
run_variant cuda_ga_b4_global ./build/CUDA-GA-B4-global "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"

# B4-smem only supports n <= 99. Berlin52 and smoke_20 are safe.
if [[ "$DATASET" == *"berlin52"* || "$DATASET" == *"smoke_20"* ]]; then
  run_variant cuda_ga_b4_smem ./build/CUDA-GA-B4-smem "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
else
  echo "Skipping CUDA-GA-B4-smem for dataset '$DATASET' (requires n <= 99)."
fi

echo "Done. Outputs are in results/ with suffix _${DATASET_TAG}_${SLURM_JOB_ID}.txt"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Aggregate CSV: ${AGGREGATE_CSV}"