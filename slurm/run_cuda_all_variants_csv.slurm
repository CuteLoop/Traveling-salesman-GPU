#!/bin/bash
# Runs all current CUDA variants multiple times and writes one CSV.
#
# Usage:
#   sbatch --export=ALL,DATASET=sequential/tests/fixtures/smoke_20.tsp,RUNS=10 slurm/run_cuda_all_variants_csv.slurm
#
# Notes:
# - Each run uses `seed = SEED_BASE + run_index - 1` so stochastic quality can be compared.
# - CUDA timing is parsed from program output when available.

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
CSV_OUT="results/cuda_all_variants_${DATASET_TAG}_${JOB_TAG}.csv"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tsp_cuda_csv_${JOB_TAG}_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "variant,binary,version,dataset,run_index,seed,population,islands,generations,mutation_rate,elite_count,reported_length,cuda_elapsed_ms,generations_completed,target_reached,status,notes" > "$CSV_OUT"

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

  local status="ok"
  local notes=""
  if [ "$rc" -ne 0 ]; then
    status="failed_${rc}"
    notes="$(tail -n 1 "$stdout_file" | tr ',' ';' | tr '\r\n' '  ')"
  fi

  echo "${variant},${binary},${version},${DATASET},${run_index},${seed},${population},${islands},${generations},${mutation},${elite},${length},${elapsed_ms},${generations_completed},${target_reached},${status},${notes}" >> "$CSV_OUT"
}

compile_variant "build/GPU-Naive" "src/cuda/GPU-Naive.cu"
compile_variant "build/CUDA-GA" "src/cuda/CUDA-GA.cu"
compile_variant "build/CUDA-GA-GPU-Pop" "src/cuda/CUDA-GA-GPU-Pop.cu"
compile_variant "build/CUDA-GA-GPU-Pop-bankconflict" "src/cuda/variants/CUDA-GA-GPU-Pop-bankconflict.cu"
compile_variant "build/CUDA-GA-GPU-Pop-bitset" "src/cuda/variants/CUDA-GA-GPU-Pop-bitset.cu"
compile_variant "build/GA-GPU-POP-ParallelSort" "src/cuda/variants/CUDA-GA-GPU-Pop-ParallelSort.cu"
compile_variant "build/GA-GPU-POP-AoS" "src/cuda/variants/CUDA-GA-GPU-Pop-AoS.cu"
compile_variant "build/GA-GPU-POP-GlobalDist" "src/cuda/variants/CUDA-GA-GPU-Pop-GlobalDist.cu"
compile_variant "build/GA-GPU-POP-VerboseComments" "src/cuda/variants/CUDA-GA-GPU-Pop-VerboseComments.cu"
compile_variant "build/CUDA-GA-B1-stride" "src/cuda/variants/CUDA-GA-B1-stride.cu"
compile_variant "build/CUDA-GA-B2-bitmask" "src/cuda/variants/CUDA-GA-B2-bitmask.cu"
compile_variant "build/CUDA-GA-B3-reduce" "src/cuda/variants/CUDA-GA-B3-reduce.cu"
compile_variant "build/CUDA-GA-B3-shuffle" "src/cuda/variants/CUDA-GA-B3-shuffle.cu"
compile_variant "build/CUDA-GA-B4-global" "src/cuda/variants/CUDA-GA-B4-global.cu"

if [ "$DATASET_DIM" != "unknown" ] && [ "$DATASET_DIM" -le 99 ]; then
  compile_variant "build/CUDA-GA-B4-smem" "src/cuda/variants/CUDA-GA-B4-smem.cu"
else
  echo "Skipping build/CUDA-GA-B4-smem for dataset dimension ${DATASET_DIM} (> 99 or unknown)"
fi

for run_index in $(seq 1 "$RUNS"); do
  seed=$((SEED_BASE + run_index - 1))

  echo "Run ${run_index}/${RUNS} with seed ${seed}"

  run_and_record "gpu_naive" "build/GPU-Naive" "$run_index" "$seed" "NA" "NA" "NA" "NA" "NA" \
    ./build/GPU-Naive "$DATASET"

  run_and_record "cuda_ga" "build/CUDA-GA" "$run_index" "$seed" "$POP_HYBRID" "NA" "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" \
    ./build/CUDA-GA "$DATASET" "$POP_HYBRID" "$GENERATIONS" "$MUTATION" "$ELITE_HYBRID" "$seed"

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

  if [ "$DATASET_DIM" != "unknown" ] && [ "$DATASET_DIM" -le 99 ]; then
    run_and_record "cuda_ga_b4_smem" "build/CUDA-GA-B4-smem" "$run_index" "$seed" "NA" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" \
      ./build/CUDA-GA-B4-smem "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$seed"
  else
    echo "cuda_ga_b4_smem,build/CUDA-GA-B4-smem,cuda_ga_b4_smem,${DATASET},${run_index},${seed},NA,${ISLANDS},${GENERATIONS},${MUTATION},${ELITE_POP},NA,NA,NA,NA,skipped,dataset dimension ${DATASET_DIM} exceeds B4-smem limit 99" >> "$CSV_OUT"
  fi
done

echo "Wrote ${CSV_OUT}"
