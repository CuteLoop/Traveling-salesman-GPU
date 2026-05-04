import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from pyCombinatorial.algorithm.ga import genetic_algorithm
from pyCombinatorial.utils import util


CUDA_EQUIV_POP = 512
CUDA_EQUIV_GENERATIONS = 2000
CUDA_EQUIV_MUTATION = 0.03
CUDA_EQUIV_ELITE = 4
CUDA_EQUIV_SEED = 100


def load_tsplib_coords(path):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    n = None
    in_coords = False
    coords = {}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        upper = line.upper()

        if upper.startswith("DIMENSION"):
            m = re.search(r":\s*(\d+)", line)
            if m:
                n = int(m.group(1))
            continue

        if upper == "NODE_COORD_SECTION":
            in_coords = True
            continue

        if upper == "EOF":
            break

        if in_coords:
            parts = line.split()
            if len(parts) < 3:
                continue
            idx = int(parts[0]) - 1
            x = float(parts[1])
            y = float(parts[2])
            coords[idx] = (x, y)

    if n is None:
        raise ValueError("DIMENSION not found in TSPLIB file")
    if len(coords) != n:
        raise ValueError(f"Expected {n} coordinates, parsed {len(coords)}")

    ordered = [coords[i] for i in range(n)]
    return pd.DataFrame(ordered).values


def normalize_route(route):
    route = [int(x) for x in route]
    if len(route) > 1 and route[0] == route[-1]:
        route = route[:-1]
    return route


def cycle_string(route):
    if not route:
        return ""
    cycle = route + [route[0]]
    return " -> ".join(str(x) for x in cycle)


def canonical_cycle(route):
    n = len(route)
    if n == 0:
        return tuple()

    min_city = min(route)
    starts = [i for i, c in enumerate(route) if c == min_city]

    candidates = []
    for s in starts:
        fwd = route[s:] + route[:s]
        rev = [route[(s - i) % n] for i in range(n)]
        candidates.append(tuple(fwd))
        candidates.append(tuple(rev))

    return min(candidates)


def routes_equivalent(route_a, route_b):
    if len(route_a) != len(route_b):
        return False
    return canonical_cycle(route_a) == canonical_cycle(route_b)


def run_python_ga(coords, pop, generations, mutation_rate, elite_count, verbose):
    dist = util.build_distance_matrix(coords)

    params = {
        "population_size": pop,
        "elite": elite_count,
        "mutation_rate": mutation_rate,
        "mutation_search": 8,
        "generations": generations,
        "verbose": verbose,
    }

    t0 = time.perf_counter()
    route, distance = genetic_algorithm(dist, **params)
    elapsed = time.perf_counter() - t0

    route = normalize_route(route)
    return route, float(distance), elapsed


def parse_sequential_output(stdout):
    dist_match = re.search(r"Best distance:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    if not dist_match:
        raise ValueError("Could not parse sequential best distance from output")
    distance = float(dist_match.group(1))

    lines = stdout.splitlines()
    route = None

    for i, line in enumerate(lines):
        if "Best tour (0-based indices):" in line:
            if i + 1 >= len(lines):
                raise ValueError("Sequential output missing route line")
            nums = re.findall(r"-?\d+", lines[i + 1])
            if not nums:
                raise ValueError("Sequential output route line has no indices")
            route = [int(x) for x in nums]
            break

    if route is None:
        raise ValueError("Could not parse sequential best tour from output")

    route = normalize_route(route)
    return route, distance


def run_sequential_ga(repo_root, tsp_file, pop, generations, mutation_rate, elite_count, seed, build_release):
    seq_dir = repo_root / "sequential"

    if build_release:
        subprocess.run(["make", "clean"], cwd=seq_dir, check=True)
        subprocess.run(["make", "all", "BUILD=release"], cwd=seq_dir, check=True)
    else:
        subprocess.run(["make", "all"], cwd=seq_dir, check=True)

    exe = seq_dir / "bin" / ("ga-tsp.exe" if sys.platform.startswith("win") else "ga-tsp")
    if not exe.exists():
        fallback = seq_dir / "bin" / "ga-tsp"
        if fallback.exists():
            exe = fallback
        else:
            raise FileNotFoundError(f"Sequential executable not found: {exe}")

    rel_tsp = str(Path(tsp_file).resolve().relative_to(seq_dir.resolve()))

    cmd = [
        str(exe),
        "--instance",
        rel_tsp,
        "--pop",
        str(pop),
        "--gen",
        str(generations),
        "--seed",
        str(seed),
        "--elites",
        str(elite_count),
        "--tk",
        "3",
        "--pc",
        "0.9",
        "--pm",
        str(mutation_rate),
        "--csv",
        "results_compare.csv",
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=seq_dir, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    route, distance = parse_sequential_output(result.stdout)
    return route, distance, elapsed, result.stdout


def main():
    parser = argparse.ArgumentParser(description="Compare Python baseline GA vs sequential C GA on same TSPLIB instance")
    parser.add_argument("--tsp", default="sequential/tests/fixtures/smoke_20.tsp", help="Path to TSPLIB coordinate file")
    parser.add_argument("--pop", type=int, default=CUDA_EQUIV_POP, help="Population size for both runs; default matches the CUDA hybrid population")
    parser.add_argument("--gen", type=int, default=CUDA_EQUIV_GENERATIONS, help="Generation count for both runs; default matches the CUDA standardized sweep")
    parser.add_argument("--mutation", type=float, default=CUDA_EQUIV_MUTATION, help="Mutation rate for both runs; default matches the CUDA standardized sweep")
    parser.add_argument("--elite", type=int, default=CUDA_EQUIV_ELITE, help="Elite count for both runs; default matches the CUDA hybrid elite count")
    parser.add_argument("--seed", type=int, default=CUDA_EQUIV_SEED, help="Seed for both runs; default matches the CUDA standardized sweep")
    parser.add_argument("--python-verbose", action="store_true", help="Enable verbose output in pyCombinatorial")
    parser.add_argument("--no-release-build", action="store_true", help="Do not force BUILD=release for sequential make")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tsp_path = (repo_root / args.tsp).resolve()
    if not tsp_path.exists():
        raise FileNotFoundError(f"TSP file not found: {tsp_path}")

    coords = load_tsplib_coords(tsp_path)

    py_route, py_dist, py_time = run_python_ga(
        coords,
        args.pop,
        args.gen,
        args.mutation,
        args.elite,
        args.python_verbose,
    )

    c_route, c_dist, c_time, c_stdout = run_sequential_ga(
        repo_root,
        tsp_path,
        args.pop,
        args.gen,
        args.mutation,
        args.elite,
        args.seed,
        build_release=not args.no_release_build,
    )

    same_tour = routes_equivalent(py_route, c_route)
    dist_diff = abs(py_dist - c_dist)

    print("=== Comparison: Python baseline GA vs Sequential C GA ===")
    print(f"Instance: {tsp_path}")
    print(f"Population: {args.pop}, Generations: {args.gen}, Mutation: {args.mutation}, Elite: {args.elite}, Seed: {args.seed}")
    print("Timing method: wall-clock around each solver run; sequential timing excludes the make build step.")
    print()
    print("Python baseline (pyCombinatorial):")
    print(f"  Distance: {py_dist:.6f}")
    print(f"  Time (s): {py_time:.6f}")
    print(f"  Route length: {len(py_route)}")
    print(f"  Tour order: {cycle_string(py_route)}")
    print()
    print("Sequential C implementation:")
    print(f"  Distance: {c_dist:.6f}")
    print(f"  Time (s): {c_time:.6f}")
    print(f"  Route length: {len(c_route)}")
    print(f"  Tour order: {cycle_string(c_route)}")
    print()
    print(f"Distance absolute difference: {dist_diff:.6f}")
    print(f"Equivalent tour (rotation/reversal-invariant): {same_tour}")

    if not same_tour:
        print("\nNote: Different tours can still have very similar (or equal) distances in GA runs.")

    # Save detailed snapshot for reproducibility
    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "python_vs_sequential_compare.csv"
    rows = [
        {
            "implementation": "python_baseline",
            "instance": str(tsp_path),
            "population": args.pop,
            "generations": args.gen,
            "mutation": args.mutation,
            "elite": args.elite,
            "seed": args.seed,
            "distance": py_dist,
            "time_sec": py_time,
            "time_measurement": "wall_clock_solver_only",
            "tour_length": len(py_route),
            "tour_order": cycle_string(py_route),
            "same_tour_rotation_reversal": same_tour,
            "distance_abs_diff": dist_diff,
        },
        {
            "implementation": "sequential_c",
            "instance": str(tsp_path),
            "population": args.pop,
            "generations": args.gen,
            "mutation": args.mutation,
            "elite": args.elite,
            "seed": args.seed,
            "distance": c_dist,
            "time_sec": c_time,
            "time_measurement": "wall_clock_solver_only_excludes_build",
            "tour_length": len(c_route),
            "tour_order": cycle_string(c_route),
            "same_tour_rotation_reversal": same_tour,
            "distance_abs_diff": dist_diff,
        },
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_file = out_dir / "python_vs_sequential_compare.txt"
    out_file.write_text(
        "\n".join(
            [
                "=== Python baseline GA vs Sequential C GA ===",
                f"instance={tsp_path}",
                f"pop={args.pop}",
                f"gen={args.gen}",
                f"mutation={args.mutation}",
                f"elite={args.elite}",
                f"seed={args.seed}",
                "timing_method=wall_clock_solver_only",
                "sequential_timing_excludes_build=true",
                f"python_distance={py_dist:.6f}",
                f"python_time_sec={py_time:.6f}",
                f"c_distance={c_dist:.6f}",
                f"c_time_sec={c_time:.6f}",
                f"distance_abs_diff={dist_diff:.6f}",
                f"same_tour={same_tour}",
                f"python_route={py_route}",
                f"c_route={c_route}",
                f"python_tour_order={cycle_string(py_route)}",
                f"c_tour_order={cycle_string(c_route)}",
                "",
                "--- sequential_stdout ---",
                c_stdout,
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nSaved comparison details to: {out_file}")
    print(f"Saved comparison CSV to: {out_csv}")


if __name__ == "__main__":
    main()
