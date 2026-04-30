# Minimal HPC-ready Makefile for Traveling-salesman-GPU
# Builds sequential + CUDA binaries into build/

CC ?= gcc
NVCC ?= nvcc

CUDA_HOME ?=
ifeq ($(strip $(CUDA_HOME)),)
  NVCC_CMD := $(NVCC)
else
  NVCC_CMD := $(CUDA_HOME)/bin/$(NVCC)
endif

BUILD_DIR := build

# Verified file layout in this repository
SEQ_MAIN   := sequential/src/main.c
SEQ_SRCS   := \
	sequential/src/instance.c \
	sequential/src/tour.c \
	sequential/src/fitness.c \
	sequential/src/rng.c \
	sequential/src/init.c \
	sequential/src/selection.c \
	sequential/src/crossover.c \
	sequential/src/elitism.c \
	sequential/src/replacement.c \
	sequential/src/mutation.c \
	sequential/src/ga_stats.c \
	sequential/src/ga_driver.c

NAIVE_SRC  := src/cuda/GPU-Naive.cu
HYBRID_SRC := src/cuda/CUDA-GA.cu
ISLAND_SRC := src/cuda/CUDA-GA-GPU-Pop.cu
ISLAND_BANKCONFLICT_SRC := src/cuda/variants/CUDA-GA-GPU-Pop-bankconflict.cu
ISLAND_BITSET_SRC := src/cuda/variants/CUDA-GA-GPU-Pop-bitset.cu
ISLAND_PARALLEL_SORT_SRC := CUDA-GA-GPU-Pop-ParallelSort.cu
B1_STRIDE_SRC := src/cuda/variants/CUDA-GA-B1-stride.cu
B2_BITMASK_SRC := src/cuda/variants/CUDA-GA-B2-bitmask.cu
B3_REDUCE_SRC := src/cuda/variants/CUDA-GA-B3-reduce.cu
B3_SHUFFLE_SRC := src/cuda/variants/CUDA-GA-B3-shuffle.cu
B4_GLOBAL_SRC := src/cuda/variants/CUDA-GA-B4-global.cu
B4_SMEM_SRC := src/cuda/variants/CUDA-GA-B4-smem.cu
PARSER_SRC := src/cpp/tsplib_parser.cpp
PARSER_HDR := src/cpp/tsplib_parser.h

SEQ_BIN    := $(BUILD_DIR)/Sequential
NAIVE_BIN  := $(BUILD_DIR)/GPU-Naive
HYBRID_BIN := $(BUILD_DIR)/CUDA-GA
ISLAND_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop
ISLAND_BANKCONFLICT_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop-bankconflict
ISLAND_BITSET_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop-bitset
ISLAND_PARALLEL_SORT_BIN := $(BUILD_DIR)/GA-GPU-POP-ParallelSort
B1_STRIDE_BIN := $(BUILD_DIR)/CUDA-GA-B1-stride
B2_BITMASK_BIN := $(BUILD_DIR)/CUDA-GA-B2-bitmask
B3_REDUCE_BIN := $(BUILD_DIR)/CUDA-GA-B3-reduce
B3_SHUFFLE_BIN := $(BUILD_DIR)/CUDA-GA-B3-shuffle
B4_GLOBAL_BIN := $(BUILD_DIR)/CUDA-GA-B4-global
B4_SMEM_BIN := $(BUILD_DIR)/CUDA-GA-B4-smem

SMOKE20_DATASET := sequential/tests/fixtures/smoke_20.tsp
BERLIN52_DATASET := data/berlin52.tsp

CFLAGS     ?= -O3 -std=c11 -Wall -Wextra -Isequential/include
# Use C++11 for broad compatibility with older HPC host compilers.
NVCCFLAGS  ?= -O3 -std=c++11 -Xcompiler -std=gnu++11 -arch=sm_60 -lineinfo -Isrc/cpp

LDLIBS_SEQ := -lm

.PHONY: all clean dirs info variants all_cuda_versions sbatch_cuda_smoke20 sbatch_cuda_berlin52

all: dirs $(SEQ_BIN) $(NAIVE_BIN) $(HYBRID_BIN) $(ISLAND_BIN) $(ISLAND_BANKCONFLICT_BIN) $(ISLAND_BITSET_BIN) $(ISLAND_PARALLEL_SORT_BIN)

variants: dirs $(ISLAND_BANKCONFLICT_BIN) $(ISLAND_BITSET_BIN) $(ISLAND_PARALLEL_SORT_BIN)

all_cuda_versions: dirs $(NAIVE_BIN) $(HYBRID_BIN) $(ISLAND_BIN) $(ISLAND_BANKCONFLICT_BIN) $(ISLAND_BITSET_BIN) $(ISLAND_PARALLEL_SORT_BIN) $(B1_STRIDE_BIN) $(B2_BITMASK_BIN) $(B3_REDUCE_BIN) $(B3_SHUFFLE_BIN) $(B4_GLOBAL_BIN) $(B4_SMEM_BIN)

dirs:
	mkdir -p $(BUILD_DIR)

info:
	@echo "CC=$(CC)"
	@echo "NVCC=$(NVCC_CMD)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@echo "SEQ_MAIN=$(SEQ_MAIN)"
	@echo "NAIVE_SRC=$(NAIVE_SRC)"
	@echo "HYBRID_SRC=$(HYBRID_SRC)"
	@echo "ISLAND_SRC=$(ISLAND_SRC)"
	@echo "ISLAND_BANKCONFLICT_SRC=$(ISLAND_BANKCONFLICT_SRC)"
	@echo "ISLAND_BITSET_SRC=$(ISLAND_BITSET_SRC)"
	@echo "ISLAND_PARALLEL_SORT_SRC=$(ISLAND_PARALLEL_SORT_SRC)"
	@echo "B1_STRIDE_SRC=$(B1_STRIDE_SRC)"
	@echo "B2_BITMASK_SRC=$(B2_BITMASK_SRC)"
	@echo "B3_REDUCE_SRC=$(B3_REDUCE_SRC)"
	@echo "B3_SHUFFLE_SRC=$(B3_SHUFFLE_SRC)"
	@echo "B4_GLOBAL_SRC=$(B4_GLOBAL_SRC)"
	@echo "B4_SMEM_SRC=$(B4_SMEM_SRC)"
	@echo "PARSER_SRC=$(PARSER_SRC)"
	@echo "PARSER_HDR=$(PARSER_HDR)"

$(SEQ_BIN): $(SEQ_MAIN) $(SEQ_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS_SEQ)

$(NAIVE_BIN): $(NAIVE_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(NAIVE_SRC) $(PARSER_SRC)

$(HYBRID_BIN): $(HYBRID_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(HYBRID_SRC) $(PARSER_SRC)

$(ISLAND_BIN): $(ISLAND_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_SRC) $(PARSER_SRC)

$(ISLAND_BANKCONFLICT_BIN): $(ISLAND_BANKCONFLICT_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_BANKCONFLICT_SRC) $(PARSER_SRC)

$(ISLAND_BITSET_BIN): $(ISLAND_BITSET_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_BITSET_SRC) $(PARSER_SRC)

$(ISLAND_PARALLEL_SORT_BIN): $(ISLAND_PARALLEL_SORT_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_PARALLEL_SORT_SRC) $(PARSER_SRC)

$(B1_STRIDE_BIN): $(B1_STRIDE_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(B1_STRIDE_SRC) $(PARSER_SRC)

$(B2_BITMASK_BIN): $(B2_BITMASK_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(B2_BITMASK_SRC) $(PARSER_SRC)

$(B3_REDUCE_BIN): $(B3_REDUCE_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(B3_REDUCE_SRC) $(PARSER_SRC)

$(B3_SHUFFLE_BIN): $(B3_SHUFFLE_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(B3_SHUFFLE_SRC) $(PARSER_SRC)

$(B4_GLOBAL_BIN): $(B4_GLOBAL_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(B4_GLOBAL_SRC) $(PARSER_SRC)

$(B4_SMEM_BIN): $(B4_SMEM_SRC) $(PARSER_SRC) $(PARSER_HDR)
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(B4_SMEM_SRC) $(PARSER_SRC)

sbatch_cuda_smoke20:
	sbatch --export=ALL,DATASET=$(SMOKE20_DATASET) slurm/run_cuda_all_variants.slurm

sbatch_cuda_berlin52:
	sbatch --export=ALL,DATASET=$(BERLIN52_DATASET) slurm/run_cuda_all_variants.slurm

clean:
	rm -rf $(BUILD_DIR)
