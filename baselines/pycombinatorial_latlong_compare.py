"""
Reproduce the pyCombinatorial Madeira lat/long example locally and compare:
1) Ant Colony Optimization (ACO)
2) Space Filling Curve (Hilbert)
3) Genetic Algorithm (GA)

Usage:
    python baselines/pycombinatorial_latlong_compare.py

Outputs:
    - results/pycombinatorial_madeira_summary.csv
    - results/map_aco.html
    - results/map_ga.html
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import contextily as ctx
from pyproj import Transformer

from pyCombinatorial.algorithm import (
    ant_colony_optimization,
    genetic_algorithm,
    space_filling_curve_h,
)
from pyCombinatorial.utils import graphs, util


DATA_URL = (
    "https://github.com/Valdecy/Datasets/raw/refs/heads/master/"
    "Combinatorial/TSP-05-Lat%20Long-Madeira.txt"
)


def ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_latlong_dataframe(url: str = DATA_URL) -> pd.DataFrame:
    """
    Load the Madeira lat/long dataset used in the example.

    The original example drops the first column with:
        lat_long = lat_long.iloc[:,1:]
    """
    lat_long = pd.read_csv(url, sep="\t")
    lat_long = lat_long.iloc[:, 1:]
    # pandas 2.x: reset column labels to integers so pyCombinatorial can
    # access columns by position (e.g. row[0]) instead of by string name.
    lat_long.columns = range(lat_long.shape[1])
    return lat_long


def run_aco(distance_matrix):
    params = {
        "ants": 15,
        "iterations": 100,
        "alpha": 1,
        "beta": 2,
        "decay": 0.05,
        "local_search": True,
        "verbose": True,
    }

    t0 = time.perf_counter()
    route, distance = ant_colony_optimization(distance_matrix, **params)
    elapsed = time.perf_counter() - t0
    return route, distance, elapsed, params


def run_hilbert(coordinates, distance_matrix):
    params = {
        "local_search": True,
        "verbose": True,
    }

    t0 = time.perf_counter()
    route, distance = space_filling_curve_h(coordinates, distance_matrix, **params)
    elapsed = time.perf_counter() - t0
    return route, distance, elapsed, params


def run_ga(distance_matrix):
    """
    Parameter pattern taken from the pyCombinatorial README GA example.
    Tune later if needed.
    """
    params = {
        "population_size": 30,
        "elite": 1,
        "mutation_rate": 0.10,
        "mutation_search": 8,
        "generations": 300,
        "verbose": True,
    }

    t0 = time.perf_counter()
    route, distance = genetic_algorithm(distance_matrix, **params)
    elapsed = time.perf_counter() - t0
    return route, distance, elapsed, params


def save_latlong_map(lat_long_df, route, outpath: Path):
    """
    Save a folium-style HTML map using pyCombinatorial's lat/long plotting helper.
    """
    m = graphs.plot_tour_latlong(lat_long_df, route)
    m.save(str(outpath))


def save_tour_image(lat_long_df: pd.DataFrame, route: list, outpath: Path, title: str = "TSP Tour") -> None:
    """
    Save a static PNG of the tour overlaid on a real OpenStreetMap tile background.
    lat_long_df has integer column indices: col 0 = Lat, col 1 = Long.
    route is a list of 1-based city indices (pyCombinatorial convention).
    """
    lats = lat_long_df[0].values
    lons = lat_long_df[1].values

    # pyCombinatorial returns 1-based indices; convert to 0-based
    route0 = [i - 1 for i in route]

    # Project WGS84 → Web Mercator (EPSG:3857) for contextily
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    # ordered tour coordinates (closed loop)
    tour_idx = route0 + [route0[0]]
    tour_xs = [xs[i] for i in tour_idx]
    tour_ys = [ys[i] for i in tour_idx]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # --- tour path ---
    ax.plot(tour_xs, tour_ys,
            color="#FF4500", linewidth=2.2, zorder=3,
            solid_capstyle="round", solid_joinstyle="round",
            label="Tour route",
            path_effects=[pe.Stroke(linewidth=4, foreground="white", alpha=0.6),
                          pe.Normal()])

    # --- directional arrows along the route ---
    for k in range(0, len(tour_xs) - 1, max(1, len(tour_xs) // 12)):
        dx = tour_xs[k + 1] - tour_xs[k]
        dy = tour_ys[k + 1] - tour_ys[k]
        mx, my = (tour_xs[k] + tour_xs[k + 1]) / 2, (tour_ys[k] + tour_ys[k + 1]) / 2
        ax.annotate("", xy=(mx + dx * 0.01, my + dy * 0.01), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="-|>", color="#FF4500",
                                   lw=1.5, mutation_scale=14),
                    zorder=4)

    # --- all city dots ---
    ax.scatter(xs, ys, s=70, color="white", edgecolors="#1a1a2e",
               linewidths=1.2, zorder=5)

    # --- start city ---
    ax.scatter(xs[route0[0]], ys[route0[0]], s=140, color="#FFD700",
               edgecolors="#1a1a2e", linewidths=1.5, zorder=6, label="Start city")

    # --- city id labels with white halo for legibility ---
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(
            str(i + 1),
            xy=(x, y), xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5, fontweight="bold", color="#1a1a2e",
            zorder=7,
            path_effects=[pe.Stroke(linewidth=2.5, foreground="white"),
                          pe.Normal()],
        )

    # --- OpenStreetMap tile background ---
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom="auto", alpha=0.85)

    ax.set_axis_off()

    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
                       edgecolor="#cccccc", fancybox=True)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12,
                 color="#1a1a2e",
                 path_effects=[pe.Stroke(linewidth=3, foreground="white"),
                                pe.Normal()])

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(results_dir: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    outpath = results_dir / "pycombinatorial_madeira_summary.csv"
    df.to_csv(outpath, index=False)
    return outpath


def main():
    results_dir = ensure_results_dir()

    print("Loading Madeira lat/long dataset...")
    lat_long = load_latlong_dataframe()

    print("Building Haversine distance matrix...")
    distance_matrix = util.latlong_distance_matrix(lat_long)

    print("Converting lat/long to Cartesian coordinates...")
    coordinates = util.latlong_to_cartesian(lat_long)

    print("\n=== Running ACO ===")
    aco_route, aco_distance, aco_time, aco_params = run_aco(distance_matrix)
    save_latlong_map(lat_long, aco_route, results_dir / "map_aco.html")

    img_dir = Path("img")
    img_dir.mkdir(exist_ok=True)
    save_tour_image(
        lat_long, aco_route,
        img_dir / "madeira_aco_tour.png",
        title=f"Madeira TSP — ACO Tour  (distance = {aco_distance:.2f} km, {len(aco_route)} cities)",
    )
    print(f"Saved tour image to: {img_dir / 'madeira_aco_tour.png'}")

    print("\n=== Running Hilbert SFC ===")
    sfc_route, sfc_distance, sfc_time, sfc_params = run_hilbert(
        coordinates, distance_matrix
    )

    print("\n=== Running GA ===")
    ga_route, ga_distance, ga_time, ga_params = run_ga(distance_matrix)
    save_latlong_map(lat_long, ga_route, results_dir / "map_ga.html")

    summary_rows = [
        {
            "algorithm": "ACO",
            "distance": float(aco_distance),
            "runtime_sec": aco_time,
            "route_length": len(aco_route) if aco_route is not None else None,
            "parameters": str(aco_params),
        },
        {
            "algorithm": "Hilbert_SFC",
            "distance": float(sfc_distance),
            "runtime_sec": sfc_time,
            "route_length": len(sfc_route) if sfc_route is not None else None,
            "parameters": str(sfc_params),
        },
        {
            "algorithm": "GA",
            "distance": float(ga_distance),
            "runtime_sec": ga_time,
            "route_length": len(ga_route) if ga_route is not None else None,
            "parameters": str(ga_params),
        },
    ]

    summary_csv = save_summary(results_dir, summary_rows)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['algorithm']:>12} | "
            f"distance = {row['distance']:.4f} | "
            f"time = {row['runtime_sec']:.4f} sec | "
            f"route_length = {row['route_length']}"
        )

    print(f"\nSaved summary CSV to: {summary_csv}")
    print(f"Saved ACO map to: {results_dir / 'map_aco.html'}")
    print(f"Saved GA map  to: {results_dir / 'map_ga.html'}")

    # Optional notebook/browser plots:
    # graphs.plot_tour(coordinates, city_tour=sfc_route, view='browser', size=10)
    # graphs.plot_tour(coordinates, city_tour=ga_route, view='browser', size=10)


if __name__ == "__main__":
    main()