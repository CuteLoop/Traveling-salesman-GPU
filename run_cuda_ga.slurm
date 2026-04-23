#!/bin/bash
#SBATCH --job-name=tsp_cuda_ga
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:10:00
#SBATCH -o run_cuda_ga_%j.out
#SBATCH -e run_cuda_ga_%j.error

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

DATASET="sequential/tests/fixtures/smoke_20.tsp"
mkdir -p results

if [ ! -x ./build/CUDA-GA ]; then
  echo "CUDA-GA binary not found, building first..."
  bash ./build.sh
fi

OUT_TXT="results/cuda_ga_${SLURM_JOB_ID}.txt"

echo "Running CUDA-GA on $DATASET"
/usr/bin/time -v ./build/CUDA-GA "$DATASET" 512 1000 0.05 4 42 > "$OUT_TXT" 2>&1

echo "Done. Wrote: $OUT_TXT"
