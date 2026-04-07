"""
GA-only baseline using pyCombinatorial on Berlin52-style dataset.

Outputs:
    - results/ga_berlin52_summary.csv
"""

from pathlib import Path
import time
import pandas as pd

from pyCombinatorial.algorithm import genetic_algorithm
from pyCombinatorial.utils import graphs, util


DATA_URL = "https://github.com/Valdecy/Datasets/raw/master/Combinatorial/TSP-02-Coordinates.txt"


def ensure_results_dir():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def load_coordinates(url=DATA_URL):
    df = pd.read_csv(url, sep="\t")
    return df.values


def run_ga(distance_matrix, parameters):
    t0 = time.perf_counter()
    route, distance = genetic_algorithm(distance_matrix, **parameters)
    elapsed = time.perf_counter() - t0
    return route, distance, elapsed


def main():
    results_dir = ensure_results_dir()

    print("Loading Berlin52 coordinates...")
    coordinates = load_coordinates()

    print("Building distance matrix...")
    distance_matrix = util.build_distance_matrix(coordinates)

    parameters = {
        "population_size": 15,
        "elite": 1,
        "mutation_rate": 0.1,
        "mutation_search": 8,
        "generations": 1000,
        "verbose": True,
    }

    print("\n=== Running Genetic Algorithm ===")
    route, distance, elapsed = run_ga(distance_matrix, parameters)

    print("\n=== Results ===")
    print(f"Distance: {distance:.4f}")
    print(f"Runtime: {elapsed:.4f} sec")
    print(f"Tour length: {len(route)}")

    # Save plot (browser-friendly)
    graphs.plot_tour(
        coordinates,
        city_tour=route,
        view="browser",
        size=10,
    )

    # Save summary
    summary_path = results_dir / "ga_berlin52_summary.csv"
    pd.DataFrame(
        [{
            "distance": distance,
            "runtime_sec": elapsed,
            "route_length": len(route),
            "population_size": parameters["population_size"],
            "generations": parameters["generations"],
        }]
    ).to_csv(summary_path, index=False)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()