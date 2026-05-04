import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


CUDA_EQUIV_POP = 512
CUDA_EQUIV_GENERATIONS = 2000
CUDA_EQUIV_MUTATION = 0.03
CUDA_EQUIV_ELITE = 4
CUDA_EQUIV_SEED_BASE = 100
DEFAULT_RUNS = 10
DEFAULT_TOURNAMENT = 3
DEFAULT_CROSSOVER = 0.9


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


def parse_sequential_output(stdout):
    dist_match = re.search(r"Best distance:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    if not dist_match:
        raise ValueError("Could not parse sequential best distance from output")
    distance = float(dist_match.group(1))

    lines = stdout.splitlines()
    route = None

    for idx, line in enumerate(lines):
        if "Best tour (0-based indices):" not in line:
            continue
        if idx + 1 >= len(lines):
            raise ValueError("Sequential output missing route line")
        nums = re.findall(r"-?\d+", lines[idx + 1])
        if not nums:
            raise ValueError("Sequential output route line has no indices")
        route = [int(x) for x in nums]
        break

    if route is None:
        raise ValueError("Could not parse sequential best tour from output")

    return normalize_route(route), distance


def build_sequential(seq_dir, build_release):
    if build_release:
        subprocess.run(["make", "clean"], cwd=seq_dir, check=True)
        subprocess.run(["make", "all", "BUILD=release"], cwd=seq_dir, check=True)
    else:
        subprocess.run(["make", "all"], cwd=seq_dir, check=True)


def resolve_executable(seq_dir):
    exe = seq_dir / "bin" / ("ga-tsp.exe" if sys.platform.startswith("win") else "ga-tsp")
    if exe.exists():
        return exe

    fallback = seq_dir / "bin" / "ga-tsp"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Sequential executable not found: {exe}")


def run_once(exe, seq_dir, tsp_file, pop, generations, mutation_rate, elite_count, tournament_k, crossover_rate, seed):
    rel_tsp = str(Path(tsp_file).resolve().relative_to(seq_dir.resolve()))
    run_csv_name = f"results_benchmark_seed_{seed}.csv"

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
        str(tournament_k),
        "--pc",
        str(crossover_rate),
        "--pm",
        str(mutation_rate),
        "--csv",
        run_csv_name,
    ]

    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=seq_dir, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    route, distance = parse_sequential_output(result.stdout)
    return {
        "seed": seed,
        "best_distance": distance,
        "time_sec": elapsed,
        "tour_length": len(route),
        "tour_order": cycle_string(route),
        "stdout": result.stdout,
    }


def sample_stddev(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def write_runs_csv(path, rows):
    fieldnames = [
        "run_index",
        "seed",
        "best_distance",
        "time_sec",
        "tour_length",
        "tour_order",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary):
    fieldnames = list(summary.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def print_summary_table(summary):
    rows = [
        ("Best length found", f"{summary['best_length_found']:.6f}"),
        ("Best length seed", str(summary["best_length_seed"])),
        ("Average length", f"{summary['avg_length']:.6f}"),
        ("Length stddev", f"{summary['std_length']:.6f}"),
        ("Average time (s)", f"{summary['avg_time_sec']:.6f}"),
        ("Time stddev (s)", f"{summary['std_time_sec']:.6f}"),
    ]
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"

    print("=== Summary ===")
    print(border)
    print(f"| {'Metric'.ljust(label_width)} | {'Value'.ljust(value_width)} |")
    print(border)
    for label, value in rows:
        print(f"| {label.ljust(label_width)} | {value.rjust(value_width)} |")
    print(border)
    print(f"Best tour: {summary['best_tour_order']}")


def main():
    parser = argparse.ArgumentParser(description="Run the sequential C GA benchmark multiple times and summarize best length and timing")
    parser.add_argument("--tsp", default="sequential/tests/fixtures/smoke_20.tsp", help="Path to TSPLIB instance")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of benchmark runs")
    parser.add_argument("--pop", type=int, default=CUDA_EQUIV_POP, help="Population size; default matches the CUDA hybrid population")
    parser.add_argument("--gen", type=int, default=CUDA_EQUIV_GENERATIONS, help="Generation count; default matches the CUDA standardized sweep")
    parser.add_argument("--mutation", type=float, default=CUDA_EQUIV_MUTATION, help="Mutation rate; default matches the CUDA standardized sweep")
    parser.add_argument("--elite", type=int, default=CUDA_EQUIV_ELITE, help="Elite count; default matches the CUDA hybrid elite count")
    parser.add_argument("--seed-base", type=int, default=CUDA_EQUIV_SEED_BASE, help="Base seed; each run uses seed_base + run_index")
    parser.add_argument("--tk", type=int, default=DEFAULT_TOURNAMENT, help="Tournament size for sequential GA")
    parser.add_argument("--pc", type=float, default=DEFAULT_CROSSOVER, help="Crossover probability for sequential GA")
    parser.add_argument("--no-release-build", action="store_true", help="Do not force BUILD=release for sequential make")
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    seq_dir = repo_root / "sequential"
    tsp_path = (repo_root / args.tsp).resolve()
    if not tsp_path.exists():
        raise FileNotFoundError(f"TSP file not found: {tsp_path}")

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "sequential_ga_benchmark_runs.csv"
    summary_csv = out_dir / "sequential_ga_benchmark_summary.csv"

    print("=== Sequential GA Benchmark ===")
    print(f"Instance: {tsp_path}")
    print(
        f"Runs: {args.runs}, Pop: {args.pop}, Gen: {args.gen}, Mutation: {args.mutation}, "
        f"Elite: {args.elite}, Seed base: {args.seed_base}, TK: {args.tk}, PC: {args.pc}"
    )
    print("Timing method: wall-clock around solver execution only; build time is excluded.")
    print()

    build_sequential(seq_dir, build_release=not args.no_release_build)
    exe = resolve_executable(seq_dir)

    run_rows = []
    raw_results = []

    for run_index in range(args.runs):
        seed = args.seed_base + run_index
        result = run_once(
            exe,
            seq_dir,
            tsp_path,
            args.pop,
            args.gen,
            args.mutation,
            args.elite,
            args.tk,
            args.pc,
            seed,
        )
        raw_results.append(result)

        row = {
            "run_index": run_index + 1,
            "seed": result["seed"],
            "best_distance": result["best_distance"],
            "time_sec": result["time_sec"],
            "tour_length": result["tour_length"],
            "tour_order": result["tour_order"],
        }
        run_rows.append(row)
        print(
            f"Run {run_index + 1:02d}: seed={result['seed']} "
            f"best_distance={result['best_distance']:.6f} time_sec={result['time_sec']:.6f}"
        )

    best_run = min(raw_results, key=lambda item: item["best_distance"])
    best_lengths = [item["best_distance"] for item in raw_results]
    elapsed_times = [item["time_sec"] for item in raw_results]

    summary = {
        "instance": str(tsp_path),
        "runs": args.runs,
        "population": args.pop,
        "generations": args.gen,
        "mutation": args.mutation,
        "elite": args.elite,
        "seed_base": args.seed_base,
        "tournament_k": args.tk,
        "crossover_probability": args.pc,
        "timing_method": "wall_clock_solver_only_excludes_build",
        "best_length_found": best_run["best_distance"],
        "best_length_seed": best_run["seed"],
        "best_tour_order": best_run["tour_order"],
        "avg_length": statistics.mean(best_lengths),
        "std_length": sample_stddev(best_lengths),
        "avg_time_sec": statistics.mean(elapsed_times),
        "std_time_sec": sample_stddev(elapsed_times),
    }

    write_runs_csv(runs_csv, run_rows)
    write_summary_csv(summary_csv, summary)

    print()
    print_summary_table(summary)
    print()
    print(f"Saved per-run CSV to: {runs_csv}")
    print(f"Saved summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()