#!/usr/bin/env bash
#SBATCH --job-name=tsp-gpu-pop-profile
#SBATCH --output=profiling/runs/tsp_gpu_pop_profile_%j.out
#SBATCH --error=profiling/runs/tsp_gpu_pop_profile_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail

# Adjust these module names for the cluster.
module load cuda

mkdir -p profiling/runs

nvcc -O3 -std=c++11 -lineinfo -Isrc/cpp src/cuda/CUDA-GA-GPU-Pop.cu src/cpp/tsplib_parser.cpp -o tsp_gpu_pop

# If running specifically on P100 nodes, use this instead:
# nvcc -O3 -std=c++11 -lineinfo -arch=sm_60 -Isrc/cpp src/cuda/CUDA-GA-GPU-Pop.cu src/cpp/tsplib_parser.cpp -o tsp_gpu_pop

INPUT="${1:?Usage: sbatch profiling/slurm_profile_gpu_pop.sbatch <file.tsp> [islands] [generations] [mutation_rate] [elite_count] [seed]}"
ISLANDS="${2:-128}"
GENERATIONS="${3:-1000}"
MUTATION_RATE="${4:-0.05}"
ELITE_COUNT="${5:-2}"
SEED="${6:-12345}"

bash profiling/profile_gpu_pop.sh "${INPUT}" "${ISLANDS}" "${GENERATIONS}" "${MUTATION_RATE}" "${ELITE_COUNT}" "${SEED}"

