#!/bin/bash
#SBATCH --job-name=tsp_cuda_ga_pop_sweep
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=01:00:00
#SBATCH -o run_cuda_ga_gpu_pop_sweep_%j.out
#SBATCH -e run_cuda_ga_gpu_pop_sweep_%j.error

set -euo pipefail

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
  module load cuda11/11.8
else
  echo "Warning: module cuda11/11.8 not found, falling back to cuda11/11.0"
  module load cuda11/11.0
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p results

if [ ! -x ./build/CUDA-GA-GPU-Pop ] ||
   [ ! -x ./build/CUDA-GA-GPU-Pop-bankconflict ] ||
   [ ! -x ./build/CUDA-GA-GPU-Pop-bitset ] ||
   [ ! -x ./build/GA-GPU-POP-ParallelSort ] ||
   [ ! -x ./build/GA-GPU-POP-AoS ] ||
   [ ! -x ./build/GA-GPU-POP-GlobalDist ]; then
  echo "One or more GPU-pop binaries not found, building first..."
  bash ./scripts/build.sh
fi

# Edit these arrays to control the sweep size. The defaults change one input
# lever at a time around a baseline so the CSV is easy to compare.
DATASETS=(
  "sequential/tests/fixtures/smoke_20.tsp"
  "data/berlin52.tsp"
)

VARIANTS=(
  "control:./build/CUDA-GA-GPU-Pop"
  "bankconflict:./build/CUDA-GA-GPU-Pop-bankconflict"
  "bitset:./build/CUDA-GA-GPU-Pop-bitset"
  "parallel_sort:./build/GA-GPU-POP-ParallelSort"
  "aos:./build/GA-GPU-POP-AoS"
  "global_dist:./build/GA-GPU-POP-GlobalDist"
)

SEEDS=(42 1337 2026)

# case_name:islands:generations:mutation_rate:elite_count
CASES=(
  "baseline:128:1000:0.05:2"
  "fewer_islands:64:1000:0.05:2"
  "more_islands:256:1000:0.05:2"
  "fewer_generations:128:500:0.05:2"
  "more_generations:128:2000:0.05:2"
  "low_mutation:128:1000:0.02:2"
  "high_mutation:128:1000:0.10:2"
  "low_elite:128:1000:0.05:1"
  "high_elite:128:1000:0.05:4"
)

JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_DIR="results/cuda_ga_gpu_pop_sweep_${JOB_TAG}"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
mkdir -p "$RUN_DIR"

echo "variant,case,dataset,islands,generations,mutation_rate,elite_count,seed,best_length,elapsed_seconds,status,output_file" > "$SUMMARY_CSV"

run_one() {
  local variant_name="$1"
  local binary="$2"
  local case_name="$3"
  local dataset="$4"
  local islands="$5"
  local generations="$6"
  local mutation_rate="$7"
  local elite_count="$8"
  local seed="$9"

  local dataset_name
  dataset_name="$(basename "$dataset" .tsp)"

  local out_file="${RUN_DIR}/${variant_name}_${case_name}_${dataset_name}_s${seed}.txt"
  local time_file="${out_file%.txt}.time"
  local status="ok"

  echo "Running ${variant_name} dataset=${dataset} islands=${islands} generations=${generations} mutation=${mutation_rate} elite=${elite_count} seed=${seed}"

  set +e
  /usr/bin/time -f "elapsed_seconds=%e" -o "$time_file" \
    "$binary" "$dataset" "$islands" "$generations" "$mutation_rate" "$elite_count" "$seed" \
    > "$out_file" 2>&1
  local rc=$?
  set -e

  if [ "$rc" -ne 0 ]; then
    status="failed_${rc}"
  fi

  local best_length
  best_length="$(awk -F': ' '/Best GPU-population GA tour length/ { value=$2 } END { print value }' "$out_file")"
  if [ -z "$best_length" ]; then
    best_length="NA"
  fi

  local elapsed_seconds
  elapsed_seconds="$(awk -F'=' '/elapsed_seconds/ { value=$2 } END { print value }' "$time_file")"
  if [ -z "$elapsed_seconds" ]; then
    elapsed_seconds="NA"
  fi

  echo "${variant_name},${case_name},${dataset},${islands},${generations},${mutation_rate},${elite_count},${seed},${best_length},${elapsed_seconds},${status},${out_file}" >> "$SUMMARY_CSV"
}

for variant in "${VARIANTS[@]}"; do
  variant_name="${variant%%:*}"
  binary="${variant#*:}"

  for dataset in "${DATASETS[@]}"; do
    for case_spec in "${CASES[@]}"; do
      IFS=':' read -r case_name islands generations mutation_rate elite_count <<< "$case_spec"
      for seed in "${SEEDS[@]}"; do
        run_one "$variant_name" "$binary" "$case_name" "$dataset" "$islands" "$generations" "$mutation_rate" "$elite_count" "$seed"
      done
    done
  done
done

echo "Sweep complete. Summary: $SUMMARY_CSV"
