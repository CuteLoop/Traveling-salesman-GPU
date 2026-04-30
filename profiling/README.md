# CUDA Profiling Baseline

This directory contains a repeatable profiling setup for `src/cuda/CUDA-GA-GPU-Pop.cu`.
Use it to capture a baseline before changing kernels or memory behavior.

## Build

Compile with optimization and line information:

```bash
nvcc -O3 -std=c++11 -lineinfo -Isrc/cpp src/cuda/CUDA-GA-GPU-Pop.cu src/cpp/tsplib_parser.cpp -o tsp_gpu_pop
```

For P100 nodes, add the P100 architecture target:

```bash
nvcc -O3 -std=c++11 -lineinfo -arch=sm_60 -Isrc/cpp src/cuda/CUDA-GA-GPU-Pop.cu src/cpp/tsplib_parser.cpp -o tsp_gpu_pop
```

`-lineinfo` keeps enough source-location metadata for profilers without turning off optimization.

## Baseline Run

Pick one fixed input and one fixed seed so optimization comparisons are fair:

```bash
./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

The baseline parameters are:

```text
islands=128
generations=1000
mutation_rate=0.05
elite_count=2
seed=12345
```

Use the same values for every comparison unless the experiment is specifically about scaling.

## Quick Timing

Use this when you only need wall-clock behavior:

```bash
/usr/bin/time -v ./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

## NVIDIA Nsight Systems

Use Nsight Systems to see CPU/GPU timeline behavior, kernel launch overhead, memory copies, and total CUDA activity:

```bash
nsys profile \
  --stats=true \
  --force-overwrite=true \
  --output=profile_gpu_pop_nsys \
  ./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

This creates `profile_gpu_pop_nsys.qdrep` and a text statistics summary.

## NVIDIA Nsight Compute

Use Nsight Compute for kernel-level metrics:

```bash
ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export profile_gpu_pop_ncu \
  ./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

For a lighter first pass:

```bash
ncu \
  --set basic \
  --target-processes all \
  ./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

The main kernel to inspect is:

```text
ga_island_kernel
```

## Older CUDA Toolchains

If the cluster has older CUDA tools, `nsys` and `ncu` may not be installed. Try `nvprof`:

```bash
nvprof ./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

For kernel metrics:

```bash
nvprof \
  --metrics achieved_occupancy,branch_efficiency,gld_efficiency,gst_efficiency \
  ./tsp_gpu_pop path/to/input.tsp 128 1000 0.05 2 12345
```

Metric names vary by CUDA version and GPU architecture, so remove unsupported metrics if needed.

## What To Record

For each baseline or optimization run, record:

```text
git commit:
cluster:
GPU model:
CUDA version:
compile command:
input file:
islands:
generations:
mutation_rate:
elite_count:
seed:
best tour length:
wall time:
ga_island_kernel time:
achieved occupancy:
registers per thread:
shared memory per block:
```

The best tour length should stay comparable when using the same seed and parameters.

