#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f Makefile ]]; then
	echo "Error: top-level Makefile not found in $REPO_DIR"
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

HOST_CC="gcc"
if ! command -v "$HOST_CC" >/dev/null 2>&1; then
	echo "Warning: gcc not found, using default make compiler selection"
	HOST_CC=""
fi

if [[ -n "$HOST_CC" ]]; then
	echo "Using host C compiler: $HOST_CC"
	make clean CC="$HOST_CC"
	make -j4 CC="$HOST_CC"
else
	make clean
	make -j4
fi

echo "=== Build complete ==="
ls -lh build
