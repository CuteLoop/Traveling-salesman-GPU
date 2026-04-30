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

NAIVE_SRC  := GPU-Naive.cu
HYBRID_SRC := CUDA-GA.cu
ISLAND_SRC := CUDA-GA-GPU-Pop.cu
ISLAND_BANKCONFLICT_SRC := CUDA-GA-GPU-Pop-bankconflict.cu
ISLAND_BITSET_SRC := CUDA-GA-GPU-Pop-bitset.cu
ISLAND_PARALLEL_SORT_SRC := CUDA-GA-GPU-Pop-ParallelSort.cu
ISLAND_AOS_SRC := CUDA-GA-GPU-Pop-AoS.cu
ISLAND_GLOBAL_DIST_SRC := CUDA-GA-GPU-Pop-GlobalDist.cu
ISLAND_VERBOSE_COMMENTS_SRC := CUDA-GA-GPU-Pop-VerboseComments.cu
PARSER_SRC := tsplib_parser.cpp

SEQ_BIN    := $(BUILD_DIR)/Sequential
NAIVE_BIN  := $(BUILD_DIR)/GPU-Naive
HYBRID_BIN := $(BUILD_DIR)/CUDA-GA
ISLAND_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop
ISLAND_BANKCONFLICT_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop-bankconflict
ISLAND_BITSET_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop-bitset
ISLAND_PARALLEL_SORT_BIN := $(BUILD_DIR)/GA-GPU-POP-ParallelSort
ISLAND_AOS_BIN := $(BUILD_DIR)/GA-GPU-POP-AoS
ISLAND_GLOBAL_DIST_BIN := $(BUILD_DIR)/GA-GPU-POP-GlobalDist
ISLAND_VERBOSE_COMMENTS_BIN := $(BUILD_DIR)/GA-GPU-POP-VerboseComments

CFLAGS     ?= -O3 -std=c11 -Wall -Wextra -Isequential/include
# Use C++11 for broad compatibility with older HPC host compilers.
NVCCFLAGS  ?= -O3 -std=c++11 -Xcompiler -std=gnu++11 -arch=sm_60 -lineinfo

LDLIBS_SEQ := -lm

.PHONY: all clean dirs info

all: dirs $(SEQ_BIN) $(NAIVE_BIN) $(HYBRID_BIN) $(ISLAND_BIN) $(ISLAND_BANKCONFLICT_BIN) $(ISLAND_BITSET_BIN) $(ISLAND_PARALLEL_SORT_BIN) $(ISLAND_AOS_BIN) $(ISLAND_GLOBAL_DIST_BIN) $(ISLAND_VERBOSE_COMMENTS_BIN)

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
	@echo "ISLAND_AOS_SRC=$(ISLAND_AOS_SRC)"
	@echo "ISLAND_GLOBAL_DIST_SRC=$(ISLAND_GLOBAL_DIST_SRC)"
	@echo "ISLAND_VERBOSE_COMMENTS_SRC=$(ISLAND_VERBOSE_COMMENTS_SRC)"
	@echo "PARSER_SRC=$(PARSER_SRC)"

$(SEQ_BIN): $(SEQ_MAIN) $(SEQ_SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS_SEQ)

$(NAIVE_BIN): $(NAIVE_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(NAIVE_SRC) $(PARSER_SRC)

$(HYBRID_BIN): $(HYBRID_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(HYBRID_SRC) $(PARSER_SRC)

$(ISLAND_BIN): $(ISLAND_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_SRC) $(PARSER_SRC)

$(ISLAND_BANKCONFLICT_BIN): $(ISLAND_BANKCONFLICT_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_BANKCONFLICT_SRC) $(PARSER_SRC)

$(ISLAND_BITSET_BIN): $(ISLAND_BITSET_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_BITSET_SRC) $(PARSER_SRC)

$(ISLAND_PARALLEL_SORT_BIN): $(ISLAND_PARALLEL_SORT_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_PARALLEL_SORT_SRC) $(PARSER_SRC)

$(ISLAND_AOS_BIN): $(ISLAND_AOS_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_AOS_SRC) $(PARSER_SRC)

$(ISLAND_GLOBAL_DIST_BIN): $(ISLAND_GLOBAL_DIST_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_GLOBAL_DIST_SRC) $(PARSER_SRC)

$(ISLAND_VERBOSE_COMMENTS_BIN): $(ISLAND_VERBOSE_COMMENTS_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_VERBOSE_COMMENTS_SRC) $(PARSER_SRC)

clean:
	rm -rf $(BUILD_DIR)
