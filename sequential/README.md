# ga-tsp — Sequential Genetic Algorithm for TSP

> University of Arizona · Applied Mathematics  
> Sequential C99 implementation — first half of the GPU roadmap.

---

## Overview

`ga-tsp` is a clean-room, C99 implementation of a Genetic Algorithm (GA)
solver for the Travelling Salesman Problem (TSP).  It is written to be
**correct first**, then **fast**, and designed from the outset so that each
module can be ported to CUDA with minimal interface changes.

---

## Directory Layout

```
sequential/
├── include/          # Public headers (.h)
├── src/              # Implementation files (.c)
├── tests/
│   ├── fixtures/     # .tsp test input files
│   └── *.c           # Test sources (one binary per file)
├── scripts/          # Benchmark and profiling helpers
├── docs/             # Notes and references specific to this module
├── Makefile
└── README.md         # This file
```

---

## Build Requirements

| Tool | Minimum version |
|------|----------------|
| GCC  | 7.x (C99 + `-Wall -Wextra`) |
| GNU Make | 3.81 |
| Valgrind *(optional)* | any |

---

## Quick Start

```bash
# Full clean build
make clean && make all

# Run the test suite
make test

# AddressSanitizer + UBSan build
make asan
```

---

## Conventions

| Convention | Decision |
|------------|----------|
| C standard | C99 (`-std=c99`) |
| Naming | `snake_case` for all identifiers |
| Error handling | Return codes (`ga_status_t`); no global `errno` reliance |
| Memory ownership | Documented in header comment for every allocating function |
| Status codes | `GA_OK = 0`, `GA_ERR_ALLOC`, `GA_ERR_INVALID`, `GA_ERR_IO` |

---

## Roadmap

See [`docs/seq-doc/roadmap.md`](../docs/seq-doc/roadmap.md) for the full
phase-by-phase implementation plan.

| Phase | Subsystem | Status |
|-------|-----------|--------|
| 0 | Repository & build backbone | ✅ Complete |
| 1 | Instance loading & distance matrix | ⬜ Not started |
| 2 | Tour representation & validation | ⬜ Not started |
| 3 | Fitness evaluation | ⬜ Not started |
| 4 | RNG, initialization & walking skeleton | ⬜ Not started |
| 5 | Selection | ⬜ Not started |
| 6 | Crossover | ⬜ Not started |
| 7 | Mutation | ⬜ Not started |
| 8 | Elitism & replacement | ⬜ Not started |
| 9 | GA driver & statistics | ⬜ Not started |
| 10 | Profiling & regression lock | ⬜ Not started |

---

## License

Academic use only. See repository root for details.
