#!/bin/bash
# Runs all current CUDA variants multiple times and writes per-run and average CSVs.
#
# Usage:
#   sbatch --export=ALL,DATASET=sequential/tests/fixtures/smoke_20.tsp,RUNS=10 slurm/run_cuda_all_variants_csv.slurm
#
# Notes:
# - Each run uses `seed = SEED_BASE + run_index - 1` so stochastic quality can be compared.
# - CUDA timing is parsed from program output when available.
# - Assumes `make all_cuda_versions` can build the CUDA binaries from the repo root.

#SBATCH --job-name=tsp_cuda_csv
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=02:30:00
#SBATCH -o run_cuda_all_variants_csv_%j.out
#SBATCH -e run_cuda_all_variants_csv_%j.error

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
RUNS="${RUNS:-10}"
ISLANDS="${ISLANDS:-128}"
GENERATIONS="${GENERATIONS:-1000}"
MUTATION="${MUTATION:-0.05}"
ELITE_POP="${ELITE_POP:-2}"
ELITE_HYBRID="${ELITE_HYBRID:-4}"
SEED_BASE="${SEED_BASE:-42}"
POP_HYBRID="${POP_HYBRID:-512}"

if [ ! -f "$DATASET" ]; then
  echo "Error: dataset '$DATASET' not found from repo root: $REPO_DIR"
  exit 1
fi

mkdir -p results build

JOB_TAG="${SLURM_JOB_ID:-manual}"
DATASET_TAG="$(basename "$DATASET" .tsp)"
RUN_CSV="results/cuda_all_variants_runs_${DATASET_TAG}_${JOB_TAG}.csv"
AVG_CSV="results/cuda_all_variants_avg_${DATASET_TAG}_${JOB_TAG}.csv"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tsp_cuda_csv_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "variant,binary,version,dataset,run_index,seed,population,islands,generations,mutation_rate,elite_count,reported_length,cuda_elapsed_ms,generations_completed,target_reached,best_tour_path,status,notes" > "$RUN_CSV"

get_dimension() {
  awk -F: '
    /^[[:space:]]*DIMENSION[[:space:]]*:/ {
      gsub(/[[:space:]]/, "", $2);
      print $2;
      exit
    }
  ' "$1"
}

DATASET_DIM="$(get_dimension "$DATASET")"
if [ -z "$DATASET_DIM" ]; then
  DATASET_DIM="unknown"
fi

run_and_record() {
  local variant="$1"
  local binary="$2"
  local run_index="$3"
  local seed="$4"
  local population="$5"
  local islands="$6"
  local generations="$7"
  local mutation="$8"
  local elite="$9"
  shift 9

  local stdout_file="$TMP_DIR/${variant}_run${run_index}.out"
  local rc=0

  "$@" > "$stdout_file" 2>&1 || rc=$?

  local version=""
  version="$(awk -F': ' '/^VERSION:/ {print $2; exit}' "$stdout_file")"
  if [ -z "$version" ]; then
    version="$variant"
  fi

  local length=""
  length="$(awk -F': ' '
    /Best GPU-population GA tour length/ {val=$2}
    /Best GA tour length/ {val=$2}
    /Best tour length/ {val=$2}
    /^Tour length:/ {if (val == "") val=$2}
    END {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val);
      print val
    }
  ' "$stdout_file")"
  if [ -z "$length" ]; then
    length="NA"
  fi

  local elapsed_ms=""
  elapsed_ms="$(awk -F': ' '/^CUDA kernel elapsed ms:/ {print $2; exit}' "$stdout_file")"
  if [ -z "$elapsed_ms" ]; then
    elapsed_ms="NA"
  fi

  local generations_completed=""
  generations_completed="$(awk -F': ' '/^Generations completed:/ {print $2; exit}' "$stdout_file")"
  if [ -z "$generations_completed" ]; then
    generations_completed="NA"
  fi

  local target_reached=""
  target_reached="$(awk -F': ' '/^Target reached:/ {print $2; exit}' "$stdout_file")"
  if [ -z "$target_reached" ]; then
    target_reached="NA"
  fi

  local best_tour_path=""
  best_tour_path="$(awk '
    found {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0);
      print $0;
      exit
    }
    /^Best tour \(0-based indices\):$/ { found=1; next }
    /^Tour \(0-based indices\):$/ { found=1; next }
  ' "$stdout_file")"
  if [ -z "$best_tour_path" ]; then
    best_tour_path="NA"
  fi

  local status="ok"
  local notes=""
  if [ "$rc" -ne 0 ]; then
    status="failed_${rc}"
    notes="$(tail -n 1 "$stdout_file" | tr ',' ';' | tr '\r\n' '  ')"
  fi

  echo "${variant},${binary},${version},${DATASET},${run_index},${seed},${population},${islands},${generations},${mutation},${elite},${length},${elapsed_ms},${generations_completed},${target_reached},${best_tour_path},${status},${notes}" >> "$RUN_CSV"
}

if [ ! -x ./build/CUDA-GA-C5-bigpop ]; then
  echo "Building CUDA variants with make all_cuda_versions..."
  make all_cuda_versions
fi

for run_index in $(seq 1 "$RUNS"); do
  seed=$((SEED_BASE + run_index - 1))

  echo "Run ${run_index}/${RUNS} with seed ${seed}"

  run_and_record "gpu_naive" "build/GPU-Naive" "$run_index" "$seed" "NA" "NA" "NA" "NA" "NA" \
    ./build/GPU-Naive "$DATASET"

  run_and_record "cuda_ga" "build/CUDA-GA" "$run_index" "$seed" "$POP_HYBRID" "NA" "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" \
    ./build/CUDA-GA "$DATASET" "$POP_HYBRID" "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" "$seed"

  run_and_record "cuda_ga_no_greedy" "build/CUDA-GA-no-greedy" "$run_index" "$seed" "$POP_HYBRID" "NA" "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" \
    ./build/CUDA-GA-no-greedy "$DATASET" "$POP_HYBRID" "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" "$seed"

  run_and_record "cuda_ga_gpu_pop" "build/CUDA-GA-GPU-Pop" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-GPU-Pop "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_gpu_pop_bankconflict" "build/CUDA-GA-GPU-Pop-bankconflict" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-GPU-Pop-bankconflict "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_gpu_pop_bitset" "build/CUDA-GA-GPU-Pop-bitset" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-GPU-Pop-bitset "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_gpu_pop_parallel_sort" "build/GA-GPU-POP-ParallelSort" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/GA-GPU-POP-ParallelSort "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_gpu_pop_aos" "build/GA-GPU-POP-AoS" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/GA-GPU-POP-AoS "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_gpu_pop_global_dist" "build/GA-GPU-POP-GlobalDist" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/GA-GPU-POP-GlobalDist "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_gpu_pop_verbose_comments" "build/GA-GPU-POP-VerboseComments" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/GA-GPU-POP-VerboseComments "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_b1_stride" "build/CUDA-GA-B1-stride" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-B1-stride "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_b2_bitmask" "build/CUDA-GA-B2-bitmask" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-B2-bitmask "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_b3_reduce" "build/CUDA-GA-B3-reduce" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-B3-reduce "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_b3_shuffle" "build/CUDA-GA-B3-shuffle" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-B3-shuffle "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_b4_global" "build/CUDA-GA-B4-global" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-B4-global "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_b5_bigpop" "build/CUDA-GA-B5-bigpop" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-B5-bigpop "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_c1_stride" "build/CUDA-GA-C1-stride" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-C1-stride "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_c2_bitmask" "build/CUDA-GA-C2-bitmask" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-C2-bitmask "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_c3_reduce" "build/CUDA-GA-C3-reduce" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-C3-reduce "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_c4_global" "build/CUDA-GA-C4-global" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-C4-global "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  run_and_record "cuda_ga_c5_bigpop" "build/CUDA-GA-C5-bigpop" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
    ./build/CUDA-GA-C5-bigpop "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"

  if [ "$DATASET_DIM" != "unknown" ] && [ "$DATASET_DIM" -le 99 ]; then
    run_and_record "cuda_ga_b4_smem" "build/CUDA-GA-B4-smem" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
      ./build/CUDA-GA-B4-smem "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"
  else
    echo "cuda_ga_b4_smem,build/CUDA-GA-B4-smem,cuda_ga_b4_smem,${DATASET},${run_index},${seed},NA,${ISLANDS},${GENERATIONS},${MUTATION},${ELITE_POP},NA,NA,NA,NA,NA,skipped,dataset dimension ${DATASET_DIM} exceeds B4-smem limit 99" >> "$RUN_CSV"
  fi
done

awk -F, '
BEGIN {
  OFS=",";
  print "variant,version,dataset,runs,ok_runs,target_hits,best_reported_length,best_cuda_elapsed_ms,best_tour_path,mean_reported_length,stddev_reported_length,mean_cuda_elapsed_ms,stddev_cuda_elapsed_ms,mean_generations_completed";
}
NR == 1 { next }
{
  variant = $1;
  version[variant] = $3;
  dataset[variant] = $4;
  runs[variant]++;

  if ($17 == "ok") {
    ok[variant]++;
    if ($15 == "yes") {
      hit_count[variant]++;
    }
    if ($12 != "NA") {
      current_len = $12 + 0;
      if (!(variant in best_len) || current_len < best_len[variant]) {
        best_len[variant] = current_len;
        best_time[variant] = $13;
        best_tour[variant] = $16;
      }
      len_sum[variant] += $12;
      len_sq_sum[variant] += ($12 * $12);
      len_count[variant]++;
    }
    if ($13 != "NA") {
      time_sum[variant] += $13;
      time_sq_sum[variant] += ($13 * $13);
      time_count[variant]++;
    }
    if ($14 != "NA") {
      gen_sum[variant] += $14;
      gen_count[variant]++;
    }
  }
}
END {
  for (variant in runs) {
    mean_len = (len_count[variant] > 0) ? len_sum[variant] / len_count[variant] : "NA";
    mean_time = (time_count[variant] > 0) ? time_sum[variant] / time_count[variant] : "NA";
    mean_gen = (gen_count[variant] > 0) ? gen_sum[variant] / gen_count[variant] : "NA";

    if (len_count[variant] > 1) {
      len_var = (len_sq_sum[variant] - ((len_sum[variant] * len_sum[variant]) / len_count[variant])) / (len_count[variant] - 1);
      if (len_var < 0) len_var = 0;
      len_stddev = sqrt(len_var);
    } else {
      len_stddev = "NA";
    }

    if (time_count[variant] > 1) {
      time_var = (time_sq_sum[variant] - ((time_sum[variant] * time_sum[variant]) / time_count[variant])) / (time_count[variant] - 1);
      if (time_var < 0) time_var = 0;
      time_stddev = sqrt(time_var);
    } else {
      time_stddev = "NA";
    }

    best_len_out = (variant in best_len) ? best_len[variant] : "NA";
    best_time_out = (variant in best_time) ? best_time[variant] : "NA";
    best_tour_out = (variant in best_tour) ? best_tour[variant] : "NA";

    print variant, version[variant], dataset[variant], runs[variant], ok[variant] + 0, hit_count[variant] + 0, best_len_out, best_time_out, best_tour_out, mean_len, len_stddev, mean_time, time_stddev, mean_gen;
  }
}
' "$RUN_CSV" > "$AVG_CSV"

echo "Per-run CSV: $RUN_CSV"
echo "Average CSV: $AVG_CSV"
