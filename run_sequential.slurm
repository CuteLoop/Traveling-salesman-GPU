#!/bin/bash
#SBATCH --job-name=tsp_seq
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:10:00
#SBATCH -o run_sequential_%j.out
#SBATCH -e run_sequential_%j.error

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

if [ ! -x ./build/Sequential ]; then
  echo "Sequential binary not found, building first..."
  bash ./build.sh
fi

OUT_TXT="results/sequential_${SLURM_JOB_ID}.txt"
OUT_CSV="results/sequential_stats_${SLURM_JOB_ID}.csv"

echo "Running Sequential on $DATASET"
/usr/bin/time -v ./build/Sequential \
  --instance "$DATASET" \
  --pop 100 --gen 200 --seed 42 --elites 2 --tk 3 --pc 0.9 --pm 0.1 \
  --csv "$OUT_CSV" > "$OUT_TXT" 2>&1

echo "Done. Wrote: $OUT_TXT"
echo "Done. Wrote: $OUT_CSV"
