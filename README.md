# Traveling Salesman Problem — GPU Approaches

Experiments comparing CPU and GPU-accelerated heuristics for the Travelling Salesman Problem (TSP).  
The project follows a structured progression: **Python baseline → sequential C → naive GPU → optimized GPU**, so that each stage can be validated against the one before it.

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| **1. Python baseline** | ✅ Done | pyCombinatorial (GA, ACO, Hilbert SFC) — correctness reference |
| **2. Sequential C** | 🔲 Next | Single-threaded C implementation of the GA |
| **3. Naive GPU (CUDA)** | 🔲 Planned | Port fitness evaluation to GPU, no memory optimizations |
| **4. Optimized GPU (CUDA)** | 🔲 Planned | Shared memory, coalesced access, warp-level primitives |

The Python baseline is the ground truth. Every subsequent implementation must produce tours within an acceptable tolerance of the Python results before moving forward.

---

## Algorithm — Genetic Algorithm

![GA Flowchart](img/flow-chart.png)

---

## Project Structure

```
.
├── baselines/                        # Python baseline (correctness reference)
│   ├── ga_runner.py                  # Core helpers: load data, build matrix, run GA
│   ├── py_combinatorial_ga_example_berlin52.py
│   └── pycombinatorial_latlong_compare.py   # ACO / GA / Hilbert SFC on Madeira dataset
├── approaches/                       # C and CUDA implementations (WIP)
├── tests/                            # Pytest test suite
│   └── test_pycombinatorial_ga.py
├── results/                          # Output CSVs and HTML maps (git-ignored)
├── img/
│   └── flow-chart.png
├── requirements.txt
└── README.md
```

---

## 1. Set Up the Environment

> Python 3.10+ recommended. CUDA toolkit required for GPU approaches.

```bash
# Clone the repo
git clone https://github.com/<your-username>/Traveling-salesman-GPU.git
cd Traveling-salesman-GPU

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

> To regenerate `requirements.txt` after adding packages:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 3. Run the Baseline Examples

### 3a. Berlin52 — GA only

Runs the GA on the classic Berlin52 dataset and saves a summary CSV.

```bash
python baselines/py_combinatorial_ga_example_berlin52.py
```

### 3b. Madeira — ACO vs Hilbert SFC vs GA (lat/long dataset)

Runs all three algorithms on the Madeira island dataset and saves results + interactive HTML maps.

```bash
python baselines/pycombinatorial_latlong_compare.py
```

---

## 4. Baseline Results — Madeira Dataset (26 cities, lat/long)

> Results from a single run on CPU (Python 3.14, pyCombinatorial 2.1.8).  
> These serve as the **correctness target** for future C and CUDA implementations.

| Algorithm | Tour Distance (km) | Runtime (s) | Notes |
|-----------|-------------------|-------------|-------|
| **ACO** | 130.55 | 3.54 | 100 iterations, 15 ants, local search |
| **Hilbert SFC** | 135.32 | 0.08 | Space-filling curve heuristic, local search |
| **GA** | 130.55 | 38.07 | 300 generations, pop 30, local search |

Key observations:
- ACO and GA converge to the same tour distance (130.55 km), confirming solution quality.
- Hilbert SFC is ~450× faster but 3.6% longer — useful as a warm-start for other methods.
- GA runtime (38 s on CPU for 26 cities) is the primary motivation for GPU acceleration.

---

## 5. Run the Tests

Run all tests (`pytest` is included in `requirements.txt`):

```bash
python -m pytest tests/ -v
```

Run a single test file:

```bash
python -m pytest tests/test_pycombinatorial_ga.py -v
```

### Test Coverage

| Test | What it checks |
|------|---------------|
| `test_coordinates_loading` | Dataset fetches correctly; shape is `(n, 2)` |
| `test_distance_matrix_properties` | Matrix is square, diagonal is 0, symmetric |
| `test_ga_returns_valid_solution` | Route is a valid permutation; distance > 0 |
| `test_ga_improves_over_random` | GA solution ≤ random tour length |

> Tests use only 50 generations and population 10 to stay fast (< 30 s on CPU).

---

## 6. Next Steps

**Stage 2 — Sequential C implementation**
- Re-implement the GA (`initialize`, `evaluate`, `select`, `crossover`, `mutate`) in C
- Validate: tour distance must match Python baseline within ±0.1%
- Measure: establish single-thread CPU runtime as baseline for GPU speedup calculations

**Stage 3 — Naive GPU (CUDA)**
- Parallelize fitness evaluation across threads (one thread per candidate tour)
- No memory hierarchy optimizations yet
- Target: correct results, measure raw GPU vs CPU speedup

**Stage 4 — Optimized GPU (CUDA)**
- Shared memory for distance matrix tiles
- Coalesced global memory access
- Warp-level reduction for fitness aggregation
