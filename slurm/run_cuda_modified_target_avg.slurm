#!/bin/bash
# Runs the currently modified GPU-pop variants against a target tour length,
# repeats each variant multiple times, and writes both per-run and averaged CSVs.
#
# First-use example:
#   sbatch --export=ALL,DATASET=data/berlin52.tsp,TARGET_LENGTH=7542,RUNS=10 slurm/run_cuda_modified_target_avg.slurm
#
# Modified variants covered by this script:
#   - CUDA-GA-GPU-Pop
#   - CUDA-GA-GPU-Pop-bankconflict
#   - CUDA-GA-GPU-Pop-bitset
#   - GA-GPU-POP-GlobalDist

#SBATCH --job-name=tsp_cuda_target_avg
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=02:00:00
#SBATCH -o run_cuda_modified_target_avg_%j.out
#SBATCH -e run_cuda_modified_target_avg_%j.error

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

DATASET="${DATASET:-data/berlin52.tsp}"
TARGET_LENGTH="${TARGET_LENGTH:?TARGET_LENGTH must be set, e.g. --export=ALL,DATASET=data/berlin52.tsp,TARGET_LENGTH=7542,RUNS=10}"
RUNS="${RUNS:-10}"
ISLANDS="${ISLANDS:-128}"
GENERATIONS="${GENERATIONS:-1000}"
MUTATION="${MUTATION:-0.05}"
ELITE_POP="${ELITE_POP:-2}"
SEED_BASE="${SEED_BASE:-42}"

if [ ! -f "$DATASET" ]; then
  echo "Error: dataset '$DATASET' not found from repo root: $REPO_DIR"
  exit 1
fi

mkdir -p results build

JOB_TAG="${SLURM_JOB_ID:-manual}"
DATASET_TAG="$(basename "$DATASET" .tsp)"
RUN_CSV="results/cuda_modified_target_runs_${DATASET_TAG}_${JOB_TAG}.csv"
AVG_CSV="results/cuda_modified_target_avg_${DATASET_TAG}_${JOB_TAG}.csv"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tsp_cuda_target_avg_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "variant,binary,version,dataset,target_length,run_index,seed,islands,generations_arg,mutation_rate,elite_count,reported_length,cuda_elapsed_ms,generations_completed,target_reached,status,notes" > "$RUN_CSV"

compile_variant() {
  local binary="$1"
  local source="$2"

  if [ -x "$binary" ]; then
    return 0
  fi

  echo "Building $(basename "$binary") from $source"
  nvcc -O3 -std=c++11 -Xcompiler -std=gnu++11 -arch=sm_60 -lineinfo \
    -Isrc/cpp -o "$binary" "$source" src/cpp/tsplib_parser.cpp
}

run_and_record() {
  local variant="$1"
  local binary="$2"
  local run_index="$3"
  local seed="$4"
  local stdout_file="$TMP_DIR/${variant}_run${run_index}.out"
  local rc=0

  "$binary" "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed" "$TARGET_LENGTH" \
    > "$stdout_file" 2>&1 || rc=$?

  local version=""
  version="$(awk -F': ' '/^VERSION:/ {print $2; exit}' "$stdout_file")"
  if [ -z "$version" ]; then
    version="$variant"
  fi

  local reported_length=""
  reported_length="$(awk -F': ' '
    /Best GPU-population GA tour length/ {val=$2}
    /Best tour length/ {val=$2}
    END {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val);
      print val
    }
  ' "$stdout_file")"
  if [ -z "$reported_length" ]; then
    reported_length="NA"
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

  local status="ok"
  local notes=""
  if [ "$rc" -ne 0 ]; then
    status="failed_${rc}"
    notes="$(tail -n 1 "$stdout_file" | tr ',' ';' | tr '\r\n' '  ')"
  fi

  echo "${variant},${binary},${version},${DATASET},${TARGET_LENGTH},${run_index},${seed},${ISLANDS},${GENERATIONS},${MUTATION},${ELITE_POP},${reported_length},${elapsed_ms},${generations_completed},${target_reached},${status},${notes}" >> "$RUN_CSV"
}

compile_variant "build/CUDA-GA-GPU-Pop" "src/cuda/CUDA-GA-GPU-Pop.cu"
compile_variant "build/CUDA-GA-GPU-Pop-bankconflict" "src/cuda/variants/CUDA-GA-GPU-Pop-bankconflict.cu"
compile_variant "build/CUDA-GA-GPU-Pop-bitset" "src/cuda/variants/CUDA-GA-GPU-Pop-bitset.cu"
compile_variant "build/GA-GPU-POP-GlobalDist" "src/cuda/variants/CUDA-GA-GPU-Pop-GlobalDist.cu"

for run_index in $(seq 1 "$RUNS"); do
  seed=$((SEED_BASE + run_index - 1))
  echo "Run ${run_index}/${RUNS} with seed ${seed}"

  run_and_record "cuda_ga_gpu_pop" "./build/CUDA-GA-GPU-Pop" "$run_index" "$seed"
  run_and_record "cuda_ga_gpu_pop_bankconflict" "./build/CUDA-GA-GPU-Pop-bankconflict" "$run_index" "$seed"
  run_and_record "cuda_ga_gpu_pop_bitset" "./build/CUDA-GA-GPU-Pop-bitset" "$run_index" "$seed"
  run_and_record "cuda_ga_gpu_pop_global_dist" "./build/GA-GPU-POP-GlobalDist" "$run_index" "$seed"
done

awk -F, '
BEGIN {
  OFS=",";
  print "variant,version,dataset,target_length,runs,ok_runs,target_hits,mean_reported_length,mean_cuda_elapsed_ms,mean_generations_completed";
}
NR == 1 { next }
{
  variant = $1;
  version[variant] = $3;
  dataset[variant] = $4;
  target[variant] = $5;
  runs[variant]++;

  if ($16 == "ok") {
    ok[variant]++;
    if ($12 != "NA") {
      len_sum[variant] += $12;
      len_count[variant]++;
    }
    if ($13 != "NA") {
      time_sum[variant] += $13;
      time_count[variant]++;
    }
    if ($14 != "NA") {
      gen_sum[variant] += $14;
      gen_count[variant]++;
    }
    if ($15 == "yes") {
      hit_count[variant]++;
    }
  }
}
END {
  for (variant in runs) {
    mean_len = (len_count[variant] > 0) ? len_sum[variant] / len_count[variant] : "NA";
    mean_time = (time_count[variant] > 0) ? time_sum[variant] / time_count[variant] : "NA";
    mean_gen = (gen_count[variant] > 0) ? gen_sum[variant] / gen_count[variant] : "NA";
    print variant, version[variant], dataset[variant], target[variant], runs[variant], ok[variant] + 0, hit_count[variant] + 0, mean_len, mean_time, mean_gen;
  }
}
' "$RUN_CSV" > "$AVG_CSV"

echo "Per-run CSV: $RUN_CSV"
echo "Average CSV: $AVG_CSV"
