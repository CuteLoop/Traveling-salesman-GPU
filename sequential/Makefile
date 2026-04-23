# ===========================================================================
# Makefile — ga-tsp sequential C99 implementation
# University of Arizona · Applied Mathematics
#
# Targets
#   all      build the smoke executable (default)
#   test     build and run all test binaries
#   clean    remove all build artefacts
#   asan     build + test with AddressSanitizer enabled
#
# Usage
#   make                   # same as make all
#   make test              # run smoke test
#   make asan              # asan build + test
#   make clean && make all # clean rebuild
# ===========================================================================

# ===========================================================================
# Makefile — ga-tsp sequential C99 implementation
# University of Arizona · Applied Mathematics
#
# Targets
#   all      build the main executable (default)
#   test     build and run all test binaries
#   clean    remove all build artefacts
#   asan     build + test with AddressSanitizer enabled
# ===========================================================================

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS =

# BUILD=release  → -O3 -march=native -DNDEBUG (baseline benchmark)
# BUILD=profile  → -O2 -g -fno-omit-frame-pointer (perf/gprof)
ifeq ($(BUILD),release)
  CFLAGS = -std=c99 -Wall -Wextra -O3 -march=native -DNDEBUG -Iinclude
else ifeq ($(BUILD),profile)
  CFLAGS = -std=c99 -Wall -Wextra -O2 -g -fno-omit-frame-pointer -Iinclude
endif

# Directories
SRC_DIR  = src
TST_DIR  = tests
BIN_DIR  = bin

ifeq ($(OS),Windows_NT)
EXE_EXT := .exe
MKDIR_P = if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"
RM_RF   = if exist "$(BIN_DIR)" rmdir /s /q "$(BIN_DIR)"
else
EXE_EXT :=
MKDIR_P = mkdir -p "$(BIN_DIR)"
RM_RF   = rm -rf "$(BIN_DIR)"
endif

# Library sources (everything under src/ except mains)
LIB_SRCS = $(SRC_DIR)/instance.c $(SRC_DIR)/tour.c $(SRC_DIR)/fitness.c \
           $(SRC_DIR)/rng.c $(SRC_DIR)/init.c $(SRC_DIR)/selection.c \
           $(SRC_DIR)/crossover.c $(SRC_DIR)/elitism.c \
           $(SRC_DIR)/replacement.c $(SRC_DIR)/mutation.c \
           $(SRC_DIR)/ga_stats.c $(SRC_DIR)/ga_driver.c

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
.PHONY: all
all: $(BIN_DIR)/ga-tsp$(EXE_EXT) $(BIN_DIR)/skeleton$(EXE_EXT) $(BIN_DIR)/stress_crossover$(EXE_EXT) $(BIN_DIR)/stress_mutation$(EXE_EXT)

$(BIN_DIR)/ga-tsp$(EXE_EXT): $(SRC_DIR)/main.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Walking skeleton executable (Phase 4)
$(BIN_DIR)/skeleton$(EXE_EXT): $(SRC_DIR)/skeleton_main.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Stress crossover executable (Phase 6)
$(BIN_DIR)/stress_crossover$(EXE_EXT): $(TST_DIR)/stress_crossover.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Stress mutation executable (Phase 7)
$(BIN_DIR)/stress_mutation$(EXE_EXT): $(TST_DIR)/stress_mutation.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# ---------------------------------------------------------------------------
# Test target
# ---------------------------------------------------------------------------
TEST_NAMES = test_smoke test_instance test_distance_matrix test_tour_validity \
             test_fitness test_rng test_initialization test_selection \
             test_crossover test_elitism test_replacement test_mutation \
             test_ga_smoke test_regression

TEST_BINS = $(patsubst %,$(BIN_DIR)/%$(EXE_EXT),$(TEST_NAMES))

.PHONY: test
test: $(TEST_BINS)
	@echo --- Running tests ---
	@"$(BIN_DIR)/test_smoke$(EXE_EXT)"
	@"$(BIN_DIR)/test_instance$(EXE_EXT)"
	@"$(BIN_DIR)/test_distance_matrix$(EXE_EXT)"
	@"$(BIN_DIR)/test_tour_validity$(EXE_EXT)"
	@"$(BIN_DIR)/test_fitness$(EXE_EXT)"
	@"$(BIN_DIR)/test_rng$(EXE_EXT)"
	@"$(BIN_DIR)/test_initialization$(EXE_EXT)"
	@"$(BIN_DIR)/test_selection$(EXE_EXT)"
	@"$(BIN_DIR)/test_crossover$(EXE_EXT)"
	@"$(BIN_DIR)/test_elitism$(EXE_EXT)"
	@"$(BIN_DIR)/test_replacement$(EXE_EXT)"
	@"$(BIN_DIR)/test_mutation$(EXE_EXT)"
	@"$(BIN_DIR)/test_ga_smoke$(EXE_EXT)"
	@"$(BIN_DIR)/test_regression$(EXE_EXT)"
	@echo ---
	@echo All tests passed.

# test_smoke has no library dependencies
$(BIN_DIR)/test_smoke$(EXE_EXT): $(TST_DIR)/test_smoke.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# Phase 1 tests link against instance.c
$(BIN_DIR)/test_instance$(EXE_EXT): $(TST_DIR)/test_instance.c $(SRC_DIR)/instance.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

$(BIN_DIR)/test_distance_matrix$(EXE_EXT): $(TST_DIR)/test_distance_matrix.c $(SRC_DIR)/instance.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Phase 2 tests link against tour.c
$(BIN_DIR)/test_tour_validity$(EXE_EXT): $(TST_DIR)/test_tour_validity.c $(SRC_DIR)/tour.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Phase 3 tests: fitness links against instance.c, tour.c, fitness.c
$(BIN_DIR)/test_fitness$(EXE_EXT): $(TST_DIR)/test_fitness.c $(SRC_DIR)/instance.c $(SRC_DIR)/tour.c $(SRC_DIR)/fitness.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Phase 4 tests: RNG is self-contained
$(BIN_DIR)/test_rng$(EXE_EXT): $(TST_DIR)/test_rng.c $(SRC_DIR)/rng.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Phase 4 tests: initialization links against all libraries
$(BIN_DIR)/test_initialization$(EXE_EXT): $(TST_DIR)/test_initialization.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Phase 5 tests: selection links against tour.c, rng.c, selection.c
$(BIN_DIR)/test_selection$(EXE_EXT): $(TST_DIR)/test_selection.c $(SRC_DIR)/tour.c $(SRC_DIR)/rng.c $(SRC_DIR)/selection.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Phase 6 tests: crossover links against tour.c, rng.c, crossover.c
$(BIN_DIR)/test_crossover$(EXE_EXT): $(TST_DIR)/test_crossover.c $(SRC_DIR)/tour.c $(SRC_DIR)/rng.c $(SRC_DIR)/crossover.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Phase 8 tests: elitism links against tour.c, elitism.c
$(BIN_DIR)/test_elitism$(EXE_EXT): $(TST_DIR)/test_elitism.c $(SRC_DIR)/tour.c $(SRC_DIR)/elitism.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Phase 8 tests: replacement links against tour.c, replacement.c
$(BIN_DIR)/test_replacement$(EXE_EXT): $(TST_DIR)/test_replacement.c $(SRC_DIR)/tour.c $(SRC_DIR)/replacement.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Phase 7 tests: mutation links against tour.c, rng.c, mutation.c
$(BIN_DIR)/test_mutation$(EXE_EXT): $(TST_DIR)/test_mutation.c $(SRC_DIR)/tour.c $(SRC_DIR)/rng.c $(SRC_DIR)/mutation.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Phase 9 tests: ga_smoke links against all library sources
$(BIN_DIR)/test_ga_smoke$(EXE_EXT): $(TST_DIR)/test_ga_smoke.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# Phase 10 tests: regression links against all library sources
$(BIN_DIR)/test_regression$(EXE_EXT): $(TST_DIR)/test_regression.c $(LIB_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -lm -o $@ $^

# ---------------------------------------------------------------------------
# ASan target — rebuilds everything with sanitizers
# ---------------------------------------------------------------------------
.PHONY: asan
asan: CFLAGS  += -fsanitize=address,undefined -g -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address,undefined
asan: clean all test

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	$(RM_RF)

# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------
$(BIN_DIR):
	$(MKDIR_P)
