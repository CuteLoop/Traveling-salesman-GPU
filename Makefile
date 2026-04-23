# Minimal HPC-ready Makefile for Traveling-salesman-GPU
# Builds sequential + CUDA binaries into build/

CXX ?= g++
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
PARSER_SRC := tsplib_parser.cpp

SEQ_BIN    := $(BUILD_DIR)/Sequential
NAIVE_BIN  := $(BUILD_DIR)/GPU-Naive
HYBRID_BIN := $(BUILD_DIR)/CUDA-GA
ISLAND_BIN := $(BUILD_DIR)/CUDA-GA-GPU-Pop

CXXFLAGS   ?= -O3 -std=c11 -Wall -Wextra -Isequential/include
NVCCFLAGS  ?= -O3 -std=c++17 -arch=sm_60 -lineinfo

LDLIBS_SEQ := -lm

.PHONY: all clean dirs info

all: dirs $(SEQ_BIN) $(NAIVE_BIN) $(HYBRID_BIN) $(ISLAND_BIN)

dirs:
	mkdir -p $(BUILD_DIR)

info:
	@echo "CXX=$(CXX)"
	@echo "NVCC=$(NVCC_CMD)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@echo "SEQ_MAIN=$(SEQ_MAIN)"
	@echo "NAIVE_SRC=$(NAIVE_SRC)"
	@echo "HYBRID_SRC=$(HYBRID_SRC)"
	@echo "ISLAND_SRC=$(ISLAND_SRC)"
	@echo "PARSER_SRC=$(PARSER_SRC)"

$(SEQ_BIN): $(SEQ_MAIN) $(SEQ_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS_SEQ)

$(NAIVE_BIN): $(NAIVE_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(NAIVE_SRC) $(PARSER_SRC)

$(HYBRID_BIN): $(HYBRID_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(HYBRID_SRC) $(PARSER_SRC)

$(ISLAND_BIN): $(ISLAND_SRC) $(PARSER_SRC) tsplib_parser.h
	$(NVCC_CMD) $(NVCCFLAGS) -o $@ $(ISLAND_SRC) $(PARSER_SRC)

clean:
	rm -rf $(BUILD_DIR)
