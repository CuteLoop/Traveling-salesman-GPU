#!/bin/bash
#SBATCH --job-name=tsp_seed_bench
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time=02:00:00
#SBATCH -o run_seed_benchmark_%j.out
#SBATCH -e run_seed_benchmark_%j.error

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
MODE="${MODE:-benchmark}"
IMPLEMENTATION_SET="${IMPLEMENTATION_SET:-all}"
REPEATS="${REPEATS:-}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -f "$DATASET" ]; then
  echo "Error: dataset '$DATASET' not found from repo root: $REPO_DIR"
  exit 1
fi

mkdir -p results

if [ ! -x ./build/CUDA-GA-B4-smem ]; then
  echo "Benchmark binaries not found, building first..."
  bash ./scripts/build.sh
fi

DATASET_TAG="$(basename "$DATASET" .tsp)"
RAW_CSV="results/${MODE}_${IMPLEMENTATION_SET}_${DATASET_TAG}_${SLURM_JOB_ID}_raw.csv"

cmd=(
  "$PYTHON_BIN" scripts/run_seed_benchmark.py
  --dataset "$DATASET"
  --mode "$MODE"
  --implementation-set "$IMPLEMENTATION_SET"
  --output-csv "$RAW_CSV"
  --results-dir results/audit_runs
  --timeout-sec "$TIMEOUT_SEC"
)

if [ -n "$REPEATS" ]; then
  cmd+=(--repeats "$REPEATS")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Aggregating raw CSV: $RAW_CSV"
"$PYTHON_BIN" scripts/aggregate_tsp_results.py "$RAW_CSV"

echo "Done. Raw CSV: $RAW_CSV"