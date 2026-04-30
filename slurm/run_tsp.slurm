#!/bin/bash
#SBATCH --job-name=tsp_benchmark
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=00:10:00
#SBATCH -o run_tsp.out
#SBATCH -e run_tsp.error

set -euo pipefail

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
  module load cuda11/11.8
else
  echo "Warning: module cuda11/11.8 not found, falling back to cuda11/11.0"
  module load cuda11/11.0
fi

# Prefer repository root resolved from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# If sbatch was invoked from a valid repo checkout, prefer it.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/Makefile" ]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
fi
cd "$REPO_DIR"

echo "=== Host / GPU info ==="
hostname
nvidia-smi || true

echo "=== Building ==="
bash ./scripts/build.sh

# Verified dataset in this repo (adjust if you prefer another .tsp)
DATASET="sequential/tests/fixtures/smoke_20.tsp"

if [ ! -f "$DATASET" ]; then
    echo "Error: dataset '$DATASET' not found in $REPO_DIR"
  echo "Update DATASET in slurm/run_tsp.slurm to a valid .tsp path."
    exit 1
fi

mkdir -p results
echo "Writing outputs under: $REPO_DIR/results"

echo "===== STARTING TSP BENCHMARKS ====="
echo "Dataset: $DATASET"
echo "-----------------------------------"

echo ">> 1. RUNNING SEQUENTIAL <<"
/usr/bin/time -v ./build/Sequential --instance "$DATASET" --pop 100 --gen 200 --seed 42 --elites 2 --tk 3 --pc 0.9 --pm 0.1 --csv results/sequential_stats.csv \
  > results/sequential.txt 2>&1 || true

echo ">> 2. RUNNING GPU-NAIVE <<"
/usr/bin/time -v ./build/GPU-Naive "$DATASET" \
  > results/gpu_naive.txt 2>&1 || true

echo ">> 3. RUNNING CUDA-GA (Hybrid) <<"
/usr/bin/time -v ./build/CUDA-GA "$DATASET" 512 1000 0.05 4 42 \
  > results/cuda_ga.txt 2>&1 || true

echo ">> 4. RUNNING CUDA-GA-GPU-Pop (Island Model) <<"
/usr/bin/time -v ./build/CUDA-GA-GPU-Pop "$DATASET" 128 1000 0.05 2 42 \
  > results/cuda_ga_gpu_pop.txt 2>&1 || true

echo "===== BENCHMARKING COMPLETE ====="
ls -lh results
