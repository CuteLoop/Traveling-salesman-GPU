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

mkdir -p results

if [ ! -x ./build/CUDA-GA-B4-smem ]; then
  echo "Building all CUDA versions..."
  make all_cuda_versions
fi

DATASET_TAG="$(basename "$DATASET" .tsp)"

run_variant() {
  local name="$1"
  shift
  local out="results/${name}_${DATASET_TAG}_${SLURM_JOB_ID}.txt"
  echo "== Running ${name} on ${DATASET} =="
  "$@" > "$out" 2>&1
  echo "Wrote ${out}"
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
