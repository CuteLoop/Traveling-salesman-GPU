"""
Tests for the pyCombinatorial GA baseline.

Run with:
    pytest tests/test_pycombinatorial_ga.py -v
"""

import numpy as np
import pytest

from baselines.ga_runner import load_coordinates_from_url, build_distance_matrix, run_ga

DATA_URL = "https://github.com/Valdecy/Datasets/raw/master/Combinatorial/TSP-02-Coordinates.txt"

# ---------------------------------------------------------------------------
# 3.1  Dataset + distance-matrix tests
# ---------------------------------------------------------------------------

def test_coordinates_loading():
    coords = load_coordinates_from_url(DATA_URL)
    assert coords.shape[1] == 2
    assert coords.shape[0] > 0


def test_distance_matrix_properties():
    coords = load_coordinates_from_url(DATA_URL)
    D = build_distance_matrix(coords)

    assert D.shape[0] == D.shape[1]
    assert D.shape[0] == len(coords)

    # diagonal must be zero
    for i in range(5):
        assert abs(D[i, i]) < 1e-9

    # symmetry
    for i in range(5):
        for j in range(5):
            assert abs(D[i, j] - D[j, i]) < 1e-9


# ---------------------------------------------------------------------------
# 3.2  GA correctness test (critical)
# ---------------------------------------------------------------------------

def test_ga_returns_valid_solution():
    coords = load_coordinates_from_url(DATA_URL)
    D = build_distance_matrix(coords)

    params = {
        "population_size": 10,
        "elite": 1,
        "mutation_rate": 0.1,
        "mutation_search": 5,
        "generations": 50,   # keep test fast
        "verbose": False,
    }

    route, distance, elapsed = run_ga(D, params)

    n = len(coords)

    assert route is not None
    assert distance > 0
    assert elapsed >= 0

    # permutation validity
    assert len(route) == n
    assert len(set(route)) == n


# ---------------------------------------------------------------------------
# 3.3  Sanity performance test
# ---------------------------------------------------------------------------

def test_ga_improves_over_random():
    coords = load_coordinates_from_url(DATA_URL)
    D = build_distance_matrix(coords)

    n = len(coords)
    random_route = np.random.permutation(n)

    def tour_length(route):
        return sum(D[route[i], route[(i + 1) % n]] for i in range(n))

    random_length = tour_length(random_route)

    params = {
        "population_size": 10,
        "elite": 1,
        "mutation_rate": 0.1,
        "mutation_search": 5,
        "generations": 50,
        "verbose": False,
    }

    _, ga_length, _ = run_ga(D, params)

    assert ga_length <= random_length
