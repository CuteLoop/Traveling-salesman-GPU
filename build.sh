#!/bin/bash
set -euo pipefail

module purge
module load cuda11/11.0

echo "=== Building Traveling-salesman-GPU ==="
echo "PWD: $(pwd)"
which nvcc || true
nvcc --version || true

make clean
make -j4

echo "=== Build complete ==="
ls -lh build
