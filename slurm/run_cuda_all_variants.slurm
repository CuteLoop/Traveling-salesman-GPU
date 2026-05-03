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

ISLANDS="${ISLANDS:-256}"
GENERATIONS="${GENERATIONS:-2000}"
MUTATION="${MUTATION:-0.03}"
ELITE_POP="${ELITE_POP:-2}"
ELITE_HYBRID="${ELITE_HYBRID:-4}"
SEED="${SEED:-100}"

mkdir -p results

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

build_cuda_variants() {
  local attempt=1
  local max_attempts=2

  while [ "$attempt" -le "$max_attempts" ]; do
    echo "Building CUDA variants with make all_cuda_versions (attempt ${attempt}/${max_attempts})..."
    if make all_cuda_versions; then
      return 0
    fi

    if [ "$attempt" -lt "$max_attempts" ]; then
      echo "Build failed; cleaning build directory before retry."
      rm -rf build
      mkdir -p build
    fi
    attempt=$((attempt + 1))
  done

  echo "Error: failed to build CUDA variants after ${max_attempts} attempts."
  return 1
}

verify_required_binaries() {
  local required=(
    build/GPU-Naive
    build/CUDA-GA
    build/CUDA-GA-GPU-Pop
    build/CUDA-GA-GPU-Pop-bankconflict
    build/CUDA-GA-GPU-Pop-bitset
    build/CUDA-GA-B1-stride
    build/CUDA-GA-B2-bitmask
    build/CUDA-GA-B3-reduce
    build/CUDA-GA-B3-shuffle
    build/CUDA-GA-B4-global
    build/CUDA-GA-B5-bigpop
  )

  if [ "$DATASET_DIM" != "unknown" ] && [ "$DATASET_DIM" -le 99 ]; then
    required+=(build/CUDA-GA-B4-smem)
  fi

  local missing=0
  local binary
  for binary in "${required[@]}"; do
    if [ ! -x "$binary" ]; then
      echo "Error: missing binary '$binary' after build."
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    return 1
  fi
}

echo "Ensuring CUDA variants are up to date..."
build_cuda_variants
verify_required_binaries

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
run_variant cuda_ga_b5_bigpop ./build/CUDA-GA-B5-bigpop "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"

# B4-smem only supports n <= 99.
if [ "$DATASET_DIM" != "unknown" ] && [ "$DATASET_DIM" -le 99 ]; then
  run_variant cuda_ga_b4_smem ./build/CUDA-GA-B4-smem "$DATASET" "$ISLANDS" "$GENERATIONS" "$MUTATION" "$ELITE_POP" "$SEED"
else
  echo "Skipping CUDA-GA-B4-smem for dataset '$DATASET' (dimension ${DATASET_DIM}; requires n <= 99)."
fi

echo "Done. Outputs are in results/ with suffix _${DATASET_TAG}_${SLURM_JOB_ID}.txt"
