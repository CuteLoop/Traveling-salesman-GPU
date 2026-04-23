#!/bin/bash
set -euo pipefail

if [[ ! -f Makefile ]]; then
	echo "Error: top-level Makefile not found in $(pwd)"
	echo "Run this script from the repository root after pulling latest changes."
	exit 1
fi

module purge
if module avail cuda11/11.8 2>&1 | grep -q "cuda11/11.8"; then
	module load cuda11/11.8
else
	echo "Warning: module cuda11/11.8 not found, falling back to cuda11/11.0"
	module load cuda11/11.0
fi

echo "=== Building Traveling-salesman-GPU ==="
echo "PWD: $(pwd)"
which nvcc || true
nvcc --version || true

make clean
make -j4

echo "=== Build complete ==="
ls -lh build
