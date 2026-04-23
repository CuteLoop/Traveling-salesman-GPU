import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def parse_tsplib_coords(tsp_path: Path):
    lines = tsp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_coords = False
    coords = {}
    n = None

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
            if len(parts) >= 3:
                idx = int(parts[0]) - 1
                x = float(parts[1])
                y = float(parts[2])
                coords[idx] = (x, y)

    if n is None:
        raise ValueError("DIMENSION not found in TSPLIB file")
    if len(coords) != n:
        raise ValueError(f"Expected {n} coordinates, parsed {len(coords)}")

    return [coords[i] for i in range(n)]


def normalize_route(route):
    if len(route) > 1 and route[0] == route[-1]:
        return route[:-1]
    return route


def to_zero_based(route, n_cities):
    route = normalize_route(route)
    if not route:
        return route

    # pyCombinatorial tours are often 1-based. Convert when detected.
    if min(route) >= 1 and max(route) == n_cities:
        return [c - 1 for c in route]
    return route


def route_to_cycle_string(route):
    route = normalize_route(route)
    return " -> ".join(str(x) for x in route + [route[0]])


def parse_local_python_and_seq(csv_path: Path, n_cities: int):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < 2:
        raise ValueError("Expected python and sequential rows in comparison CSV")

    py = next(r for r in rows if r["implementation"] == "python_baseline")
    seq = next(r for r in rows if r["implementation"] == "sequential_c")

    def parse_tour_order(s):
        raw = [int(x.strip()) for x in s.split("->")]
        return to_zero_based(raw, n_cities)

    return {
        "python_local": {
            "label": "Python baseline (local)",
            "runtime": float(py["time_sec"]),
            "best_length": float(py["distance"]),
            "tour": parse_tour_order(py["tour_order"]),
            "image": "img/smoke20_python_local.png",
        },
        "seq_local": {
            "label": "Sequential C (local)",
            "runtime": float(seq["time_sec"]),
            "best_length": float(seq["distance"]),
            "tour": parse_tour_order(seq["tour_order"]),
            "image": "img/smoke20_seq_local.png",
        },
    }


def newest_file(results_dir: Path, pattern: str):
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[0]


def parse_sequential_hpc(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    dist = float(re.search(r"Best distance:\s*([0-9]+(?:\.[0-9]+)?)", txt).group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", txt).group(1)

    lines = txt.splitlines()
    tour = None
    for i, line in enumerate(lines):
        if "Best tour (0-based indices):" in line and i + 1 < len(lines):
            nums = re.findall(r"-?\d+", lines[i + 1])
            tour = [int(x) for x in nums]
            break

    if tour is None:
        raise ValueError(f"Failed to parse sequential tour from {path}")

    return dist, elapsed, tour


def parse_gpu_naive(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    dist = float(re.search(r"Tour length:\s*([0-9]+(?:\.[0-9]+)?)", txt).group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", txt).group(1)

    lines = txt.splitlines()
    tour = None
    for i, line in enumerate(lines):
        if "Tour (0-based indices):" in line and i + 1 < len(lines):
            nums = re.findall(r"-?\d+", lines[i + 1])
            tour = [int(x) for x in nums]
            break

    if tour is None:
        raise ValueError(f"Failed to parse GPU-Naive tour from {path}")

    return dist, elapsed, tour


def parse_cuda_ga(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    dist = float(re.search(r"Best GA tour length:\s*([0-9]+(?:\.[0-9]+)?)", txt).group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", txt).group(1)

    lines = txt.splitlines()
    tour = None
    for i, line in enumerate(lines):
        if "Best tour (0-based indices):" in line and i + 1 < len(lines):
            nums = re.findall(r"-?\d+", lines[i + 1])
            tour = [int(x) for x in nums]
            break

    if tour is None:
        raise ValueError(f"Failed to parse CUDA-GA tour from {path}")

    return dist, elapsed, tour


def parse_cuda_ga_gpu_pop(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    dist = float(re.search(r"Best GPU-population GA tour length:\s*([0-9]+(?:\.[0-9]+)?)", txt).group(1))
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", txt).group(1)

    lines = txt.splitlines()
    tour = None
    for i, line in enumerate(lines):
        if "Best tour (0-based indices):" in line and i + 1 < len(lines):
            nums = re.findall(r"-?\d+", lines[i + 1])
            tour = [int(x) for x in nums]
            break

    if tour is None:
        raise ValueError(f"Failed to parse CUDA-GA-GPU-Pop tour from {path}")

    return dist, elapsed, tour


def render_tour(coords, route, output_path: Path, title: str):
    route = normalize_route(route)

    xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
    ys = [coords[i][1] for i in route] + [coords[route[0]][1]]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)

    ax.plot(
        xs,
        ys,
        color="#FF4500",
        linewidth=2.3,
        zorder=2,
        solid_capstyle="round",
        path_effects=[pe.Stroke(linewidth=4, foreground="white", alpha=0.6), pe.Normal()],
    )

    all_x = [x for x, _ in coords]
    all_y = [y for _, y in coords]
    ax.scatter(all_x, all_y, s=70, color="white", edgecolors="#1a1a2e", linewidths=1.2, zorder=3)

    start = route[0]
    ax.scatter(
        coords[start][0],
        coords[start][1],
        s=140,
        color="#FFD700",
        edgecolors="#1a1a2e",
        linewidths=1.5,
        zorder=4,
    )

    for i, (x, y) in enumerate(coords):
        ax.annotate(
            str(i),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="#1a1a2e",
            path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()],
            zorder=5,
        )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        color="#1a1a2e",
        path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
    )
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    tsp_file = repo_root / "sequential" / "tests" / "fixtures" / "smoke_20.tsp"
    coords = parse_tsplib_coords(tsp_file)

    local = parse_local_python_and_seq(results_dir / "python_vs_sequential_compare.csv", len(coords))

    seq_hpc_file = newest_file(results_dir, "sequential_*.txt")
    naive_file = newest_file(results_dir, "gpu_naive_*.txt")
    cuda_ga_file = newest_file(results_dir, "cuda_ga_*.txt")
    cuda_pop_file = newest_file(results_dir, "cuda_ga_gpu_pop_*.txt")

    seq_dist, seq_elapsed, seq_tour = parse_sequential_hpc(seq_hpc_file)
    naive_dist, naive_elapsed, naive_tour = parse_gpu_naive(naive_file)
    cga_dist, cga_elapsed, cga_tour = parse_cuda_ga(cuda_ga_file)
    pop_dist, pop_elapsed, pop_tour = parse_cuda_ga_gpu_pop(cuda_pop_file)

    seq_tour = to_zero_based(seq_tour, len(coords))
    naive_tour = to_zero_based(naive_tour, len(coords))
    cga_tour = to_zero_based(cga_tour, len(coords))
    pop_tour = to_zero_based(pop_tour, len(coords))

    tours = [
        {
            "label": local["python_local"]["label"],
            "runtime": f"{local['python_local']['runtime']:.6f}",
            "best_length": local["python_local"]["best_length"],
            "tour": local["python_local"]["tour"],
            "image": repo_root / local["python_local"]["image"],
        },
        {
            "label": local["seq_local"]["label"],
            "runtime": f"{local['seq_local']['runtime']:.6f}",
            "best_length": local["seq_local"]["best_length"],
            "tour": local["seq_local"]["tour"],
            "image": repo_root / local["seq_local"]["image"],
        },
        {
            "label": "Sequential (HPC)",
            "runtime": seq_elapsed,
            "best_length": seq_dist,
            "tour": seq_tour,
            "image": repo_root / "img" / "smoke20_seq_hpc.png",
        },
        {
            "label": "GPU-Naive (HPC)",
            "runtime": naive_elapsed,
            "best_length": naive_dist,
            "tour": naive_tour,
            "image": repo_root / "img" / "smoke20_gpu_naive_hpc.png",
        },
        {
            "label": "CUDA-GA hybrid (HPC)",
            "runtime": cga_elapsed,
            "best_length": cga_dist,
            "tour": cga_tour,
            "image": repo_root / "img" / "smoke20_cuda_ga_hpc.png",
        },
        {
            "label": "CUDA-GA GPU population (HPC)",
            "runtime": pop_elapsed,
            "best_length": pop_dist,
            "tour": pop_tour,
            "image": repo_root / "img" / "smoke20_cuda_ga_gpu_pop_hpc.png",
        },
    ]

    for item in tours:
        title = (
            f"{item['label']} | runtime={item['runtime']} s | "
            f"best length={item['best_length']}"
        )
        render_tour(coords, item["tour"], item["image"], title)
        print(f"Saved {item['image'].relative_to(repo_root)}")

    print("\nTour sequence summary:")
    for item in tours:
        seq = route_to_cycle_string(item["tour"])
        print(f"- {item['label']}: {seq} | length={item['best_length']}")


if __name__ == "__main__":
    main()
