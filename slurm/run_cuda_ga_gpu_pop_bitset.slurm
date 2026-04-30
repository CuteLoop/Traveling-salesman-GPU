#!/bin/bash
#SBATCH --job-name=tsp_cuda_ga_pop_bitset
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:10:00
#SBATCH -o run_cuda_ga_gpu_pop_bitset_%j.out
#SBATCH -e run_cuda_ga_gpu_pop_bitset_%j.error

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
mkdir -p results

if [ ! -x ./build/CUDA-GA-GPU-Pop-bitset ]; then
  echo "CUDA-GA-GPU-Pop-bitset binary not found, building variant..."
  make build/CUDA-GA-GPU-Pop-bitset
fi

OUT_TXT="results/cuda_ga_gpu_pop_bitset_${SLURM_JOB_ID}.txt"

echo "Running CUDA-GA-GPU-Pop-bitset on $DATASET"
/usr/bin/time -v ./build/CUDA-GA-GPU-Pop-bitset "$DATASET" 128 1000 0.05 2 42 > "$OUT_TXT" 2>&1

echo "Done. Wrote: $OUT_TXT"
