#!/bin/bash
set -euo pipefail

if [[ ! -f Makefile ]]; then
	echo "Error: top-level Makefile not found in $(pwd)"
	echo "Run this script from the repository root after pulling latest changes."
	exit 1
fi

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
