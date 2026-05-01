from __future__ import annotations

import csv
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED_POOL = [
    42, 108, 999, 2024, 7,
    13, 21, 34, 55, 89,
    144, 233, 377, 610, 987,
    1597, 2584, 4181, 6765, 10946,
    12345, 27182, 31415, 42424, 65537,
    8675309, 1000003, 99991, 54321, 77777,
]

REFERENCE_LENGTHS = {
    "berlin52": 7542.0,
    "smoke_20": 73.0,
}


@dataclass(frozen=True)
class ImplementationSpec:
    key: str
    label: str
    family_group: str
    source_path: str
    binary_candidates: tuple[str, ...]
    cli: tuple[str, ...]
    uses_seed: bool
    determinism_expectation: str
    rng_scope: str
    rng_notes: str


IMPLEMENTATIONS: dict[str, ImplementationSpec] = {
    "sequential_c_ga": ImplementationSpec(
        key="sequential_c_ga",
        label="Sequential C GA",
        family_group="group_a_algorithm_family",
        source_path="sequential/src/main.c",
        binary_candidates=("build/Sequential", "sequential/bin/ga-tsp", "sequential/bin/ga-tsp.exe"),
        cli=("--instance", "{dataset}", "--pop", "100", "--gen", "200", "--seed", "{seed}", "--elites", "2", "--tk", "3", "--pc", "0.9", "--pm", "0.1", "--csv", "{stats_csv}"),
        uses_seed=True,
        determinism_expectation="deterministic",
        rng_scope="per-individual host RNG state",
        rng_notes="sequential/src/ga_driver.c seeds rng_states[i] with base seed + i",
    ),
    "gpu_naive": ImplementationSpec(
        key="gpu_naive",
        label="GPU-Naive",
        family_group="group_a_algorithm_family",
        source_path="src/cuda/GPU-Naive.cu",
        binary_candidates=("build/GPU-Naive",),
        cli=("{dataset}",),
        uses_seed=False,
        determinism_expectation="deterministic",
        rng_scope="no RNG",
        rng_notes="nearest-neighbor style deterministic baseline; seed field is recorded but ignored",
    ),
    "cuda_ga": ImplementationSpec(
        key="cuda_ga",
        label="CUDA-GA hybrid",
        family_group="group_a_algorithm_family",
        source_path="src/cuda/CUDA-GA.cu",
        binary_candidates=("build/CUDA-GA",),
        cli=("{dataset}", "512", "1000", "0.05", "4", "{seed}"),
        uses_seed=True,
        determinism_expectation="deterministic",
        rng_scope="single host std::mt19937",
        rng_notes="host-side std::mt19937 drives shuffle, selection, crossover, and mutation",
    ),
    "cuda_ga_gpu_pop": ImplementationSpec(
        key="cuda_ga_gpu_pop",
        label="CUDA-GA-GPU-Pop",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/CUDA-GA-GPU-Pop.cu",
        binary_candidates=("build/CUDA-GA-GPU-Pop",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="seed is mixed with island and thread indices inside kernel; changing block topology changes RNG assignment",
    ),
    "cuda_ga_gpu_pop_bankconflict": ImplementationSpec(
        key="cuda_ga_gpu_pop_bankconflict",
        label="CUDA-GA-GPU-Pop-bankconflict",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-GPU-Pop-bankconflict.cu",
        binary_candidates=("build/CUDA-GA-GPU-Pop-bankconflict",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control with padded shared-memory layout",
    ),
    "cuda_ga_gpu_pop_bitset": ImplementationSpec(
        key="cuda_ga_gpu_pop_bitset",
        label="CUDA-GA-GPU-Pop-bitset",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-GPU-Pop-bitset.cu",
        binary_candidates=("build/CUDA-GA-GPU-Pop-bitset",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control with bitset crossover bookkeeping",
    ),
    "cuda_ga_b1_stride": ImplementationSpec(
        key="cuda_ga_b1_stride",
        label="CUDA-GA-B1-stride",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-B1-stride.cu",
        binary_candidates=("build/CUDA-GA-B1-stride",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control",
    ),
    "cuda_ga_b2_bitmask": ImplementationSpec(
        key="cuda_ga_b2_bitmask",
        label="CUDA-GA-B2-bitmask",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-B2-bitmask.cu",
        binary_candidates=("build/CUDA-GA-B2-bitmask",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control",
    ),
    "cuda_ga_b3_reduce": ImplementationSpec(
        key="cuda_ga_b3_reduce",
        label="CUDA-GA-B3-reduce",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-B3-reduce.cu",
        binary_candidates=("build/CUDA-GA-B3-reduce",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control with different elite reduction path",
    ),
    "cuda_ga_b3_shuffle": ImplementationSpec(
        key="cuda_ga_b3_shuffle",
        label="CUDA-GA-B3-shuffle",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-B3-shuffle.cu",
        binary_candidates=("build/CUDA-GA-B3-shuffle",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control with warp-shuffle elite reduction",
    ),
    "cuda_ga_b4_global": ImplementationSpec(
        key="cuda_ga_b4_global",
        label="CUDA-GA-B4-global",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-B4-global.cu",
        binary_candidates=("build/CUDA-GA-B4-global",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control with global-memory distance lookup",
    ),
    "cuda_ga_b4_smem": ImplementationSpec(
        key="cuda_ga_b4_smem",
        label="CUDA-GA-B4-smem",
        family_group="group_b_incremental_optimization",
        source_path="src/cuda/variants/CUDA-GA-B4-smem.cu",
        binary_candidates=("build/CUDA-GA-B4-smem",),
        cli=("{dataset}", "128", "1000", "0.05", "2", "{seed}"),
        uses_seed=True,
        determinism_expectation="expected_deterministic_if_topology_fixed",
        rng_scope="per-block/per-thread xorshift32",
        rng_notes="same seeded per-thread xorshift pattern as control with n<=99 shared-memory specialization",
    ),
}

IMPLEMENTATION_SETS = {
    "all": tuple(IMPLEMENTATIONS.keys()),
    "group_a": (
        "sequential_c_ga",
        "cuda_ga",
        "cuda_ga_gpu_pop",
    ),
    "group_b": (
        "cuda_ga_gpu_pop",
        "cuda_ga_gpu_pop_bankconflict",
        "cuda_ga_gpu_pop_bitset",
        "cuda_ga_b1_stride",
        "cuda_ga_b2_bitmask",
        "cuda_ga_b3_reduce",
        "cuda_ga_b3_shuffle",
        "cuda_ga_b4_global",
        "cuda_ga_b4_smem",
    ),
}


RESULT_PATTERNS = (
    re.compile(r"Best GPU-population GA tour length:\s*([0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"Best GA tour length:\s*([0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"Best distance:\s*([0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"Best tour length:\s*([0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"Tour length:\s*([0-9]+(?:\.[0-9]+)?)"),
)

TOUR_MARKERS = (
    "Best tour (0-based indices):",
    "Tour (0-based indices):",
)


@dataclass
class TsplibInstance:
    path: Path
    name: str
    edge_weight_type: str
    dimension: int
    dist: list[int]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_tag(dataset_path: str | Path) -> str:
    return Path(dataset_path).stem


def platform_binary_candidates(relative_paths: tuple[str, ...]) -> list[Path]:
    root = repo_root()
    candidates: list[Path] = []
    for rel in relative_paths:
        rel_path = Path(rel)
        candidates.append(root / rel_path)
        if platform.system() == "Windows" and rel_path.suffix == "":
            candidates.append(root / rel_path.with_suffix(".exe"))
    return candidates


def resolve_binary(spec: ImplementationSpec) -> Path:
    for candidate in platform_binary_candidates(spec.binary_candidates):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No binary found for {spec.key}; tried: "
        + ", ".join(str(p) for p in platform_binary_candidates(spec.binary_candidates))
    )


def format_cli_args(spec: ImplementationSpec, dataset: str, seed: int, stats_csv: str) -> list[str]:
    values = {
        "dataset": dataset,
        "seed": str(seed),
        "stats_csv": stats_csv,
    }
    return [part.format(**values) for part in spec.cli]


def command_for(spec: ImplementationSpec, dataset: str, seed: int, stats_csv: str = "results.csv") -> list[str]:
    return [str(resolve_binary(spec))] + format_cli_args(spec, dataset, seed, stats_csv)


def shell_join(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def _nint(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def _geo_to_radians(x: float) -> float:
    deg = int(x)
    minute = x - deg
    return math.pi * (deg + 5.0 * minute / 3.0) / 180.0


def _coord_distance(edge_weight_type: str, a: tuple[float, float, float], b: tuple[float, float, float]) -> int:
    ax, ay, az = a
    bx, by, bz = b
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    if edge_weight_type == "EUC_2D":
        return _nint(math.hypot(dx, dy))
    if edge_weight_type == "EUC_3D":
        return _nint(math.sqrt(dx * dx + dy * dy + dz * dz))
    if edge_weight_type == "MAX_2D":
        return int(max(abs(dx), abs(dy)))
    if edge_weight_type == "MAX_3D":
        return int(max(abs(dx), abs(dy), abs(dz)))
    if edge_weight_type == "MAN_2D":
        return _nint(abs(dx) + abs(dy))
    if edge_weight_type == "MAN_3D":
        return _nint(abs(dx) + abs(dy) + abs(dz))
    if edge_weight_type == "CEIL_2D":
        return int(math.ceil(math.hypot(dx, dy)))
    if edge_weight_type == "ATT":
        rij = math.sqrt((dx * dx + dy * dy) / 10.0)
        tij = _nint(rij)
        return tij + 1 if tij < rij else tij
    if edge_weight_type == "GEO":
        lati = _geo_to_radians(ax)
        longi = _geo_to_radians(ay)
        latj = _geo_to_radians(bx)
        longj = _geo_to_radians(by)
        q1 = math.cos(longi - longj)
        q2 = math.cos(lati - latj)
        q3 = math.cos(lati + latj)
        dij = 6378.388 * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
        return int(dij)
    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE for canonical evaluator: {edge_weight_type}")


def _build_explicit_matrix(dimension: int, values: list[int], edge_weight_format: str) -> list[int]:
    dist = [0] * (dimension * dimension)
    index = 0

    def next_value() -> int:
        nonlocal index
        if index >= len(values):
            raise ValueError("EDGE_WEIGHT_SECTION ended too early")
        value = values[index]
        index += 1
        return value

    def fill_symmetric(i: int, j: int, value: int) -> None:
        dist[i * dimension + j] = value
        dist[j * dimension + i] = value

    if edge_weight_format == "FULL_MATRIX":
        if len(values) < dimension * dimension:
            raise ValueError("FULL_MATRIX missing values")
        for i in range(dimension):
            for j in range(dimension):
                dist[i * dimension + j] = next_value()
        return dist

    if edge_weight_format == "UPPER_ROW":
        for i in range(dimension):
            for j in range(i + 1, dimension):
                fill_symmetric(i, j, next_value())
        return dist

    if edge_weight_format == "LOWER_ROW":
        for i in range(1, dimension):
            for j in range(i):
                fill_symmetric(i, j, next_value())
        return dist

    if edge_weight_format == "UPPER_DIAG_ROW":
        for i in range(dimension):
            for j in range(i, dimension):
                fill_symmetric(i, j, next_value())
        return dist

    if edge_weight_format == "LOWER_DIAG_ROW":
        for i in range(dimension):
            for j in range(i + 1):
                fill_symmetric(i, j, next_value())
        return dist

    raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT for canonical evaluator: {edge_weight_format}")


def load_tsplib_matrix(path: str | Path) -> TsplibInstance:
    tsp_path = Path(path)
    raw_lines = tsp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    headers: dict[str, str] = {}
    coords: dict[int, tuple[float, float, float]] = {}
    edge_values: list[int] = []
    section = None

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if section == "NODE_COORD_SECTION":
            parts = line.split()
            if len(parts) >= 3:
                node = int(parts[0]) - 1
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3]) if len(parts) >= 4 else 0.0
                coords[node] = (x, y, z)
            continue
        if section == "EDGE_WEIGHT_SECTION":
            edge_values.extend(_nint(float(value)) for value in line.split())
            continue
        if upper in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"}:
            section = upper
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().upper()] = value.strip()

    dimension = int(headers["DIMENSION"])
    name = headers.get("NAME", tsp_path.stem)
    edge_weight_type = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    edge_weight_format = headers.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()

    if edge_weight_type == "EXPLICIT":
        dist = _build_explicit_matrix(dimension, edge_values, edge_weight_format)
        return TsplibInstance(path=tsp_path, name=name, edge_weight_type=edge_weight_type, dimension=dimension, dist=dist)

    if len(coords) != dimension:
        raise ValueError(f"Expected {dimension} coordinates in {tsp_path}, parsed {len(coords)}")

    ordered = [coords[index] for index in range(dimension)]
    dist = [0] * (dimension * dimension)
    for i in range(dimension):
        for j in range(i + 1, dimension):
            value = _coord_distance(edge_weight_type, ordered[i], ordered[j])
            dist[i * dimension + j] = value
            dist[j * dimension + i] = value
    return TsplibInstance(path=tsp_path, name=name, edge_weight_type=edge_weight_type, dimension=dimension, dist=dist)


def normalize_tour(route: list[int]) -> list[int]:
    if route and len(route) > 1 and route[0] == route[-1]:
        return route[:-1]
    return route[:]


def read_tour_csv(path: str | Path) -> list[int]:
    route: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            route.append(int(row["city"]))
    return normalize_tour(route)


def write_tour_csv(path: str | Path, route: list[int]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order", "city"])
        writer.writeheader()
        for index, city in enumerate(normalize_tour(route)):
            writer.writerow({"order": index, "city": city})


def parse_solver_output_text(text: str) -> dict[str, Any]:
    reported_length = None
    for pattern in RESULT_PATTERNS:
        match = pattern.search(text)
        if match:
            reported_length = float(match.group(1))
            break

    lines = text.splitlines()
    route: list[int] = []
    for index, line in enumerate(lines):
        if any(marker in line for marker in TOUR_MARKERS):
            for candidate in lines[index + 1:]:
                stripped = candidate.strip()
                if not stripped:
                    continue
                if re.fullmatch(r"[0-9 ]+", stripped):
                    route = [int(token) for token in stripped.split()]
                    break
            if route:
                break

    runtime_sec = None
    runtime_match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text)
    if runtime_match:
        runtime_sec = parse_wall_clock_seconds(runtime_match.group(1))

    exit_code = None
    exit_match = re.search(r"Exit status:\s*(-?[0-9]+)", text)
    if exit_match:
        exit_code = int(exit_match.group(1))

    return {
        "reported_best_length": reported_length,
        "route": normalize_tour(route),
        "runtime_sec": runtime_sec,
        "exit_code": exit_code,
    }


def parse_wall_clock_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
    return float(value)


def recompute_route_length(instance: TsplibInstance, route: list[int]) -> int:
    normalized = normalize_tour(route)
    total = 0
    for a, b in zip(normalized, normalized[1:] + [normalized[0]]):
        total += instance.dist[a * instance.dimension + b]
    return total


def validate_tour(route: list[int], n: int) -> dict[str, Any]:
    normalized = normalize_tour(route)
    if not normalized:
        return {
            "route_present": False,
            "normalized_route": [],
            "valid_tour": None,
            "missing_city_count": None,
            "duplicate_city_count": None,
            "invalid_indices": [],
            "wrong_tour_length": None,
        }

    counts: dict[int, int] = {}
    for city in normalized:
        counts[city] = counts.get(city, 0) + 1

    invalid_indices = [city for city in normalized if city < 0 or city >= n]
    duplicate_city_count = sum(count - 1 for city, count in counts.items() if 0 <= city < n and count > 1)
    missing_city_count = sum(1 for city in range(n) if counts.get(city, 0) == 0)
    wrong_tour_length = len(normalized) != n
    valid = not invalid_indices and duplicate_city_count == 0 and missing_city_count == 0 and not wrong_tour_length
    return {
        "route_present": True,
        "normalized_route": normalized,
        "valid_tour": valid,
        "missing_city_count": missing_city_count,
        "duplicate_city_count": duplicate_city_count,
        "invalid_indices": invalid_indices,
        "wrong_tour_length": wrong_tour_length,
    }


def load_route_from_artifacts(output_txt: Path, best_tour_csv: Path | None) -> dict[str, Any]:
    text = output_txt.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_solver_output_text(text)
    route = parsed["route"]
    if not route and best_tour_csv and best_tour_csv.exists():
        route = read_tour_csv(best_tour_csv)
    parsed["route"] = route
    parsed["raw_text"] = text
    return parsed


def audit_paths(dataset: str | Path,
                output_txt: str | Path,
                best_tour_csv: str | Path | None = None,
                tolerance: float = 1e-6) -> dict[str, Any]:
    dataset_path = Path(dataset)
    output_path = Path(output_txt)
    best_tour_path = Path(best_tour_csv) if best_tour_csv else None
    instance = load_tsplib_matrix(dataset_path)
    parsed = load_route_from_artifacts(output_path, best_tour_path)
    validation = validate_tour(parsed["route"], instance.dimension)
    recomputed_length = None
    abs_error = None
    if validation["normalized_route"]:
        recomputed_length = float(recompute_route_length(instance, validation["normalized_route"]))
    if recomputed_length is not None and parsed["reported_best_length"] is not None:
        abs_error = abs(recomputed_length - float(parsed["reported_best_length"]))

    return {
        "dataset": str(dataset_path).replace("\\", "/"),
        "dataset_name": instance.name,
        "edge_weight_type": instance.edge_weight_type,
        "dimension": instance.dimension,
        "output_txt": str(output_path).replace("\\", "/"),
        "best_tour_csv": "" if best_tour_path is None else str(best_tour_path).replace("\\", "/"),
        "reported_best_length": parsed["reported_best_length"],
        "recomputed_best_length": recomputed_length,
        "length_abs_error": abs_error,
        "length_match": None if abs_error is None else abs_error <= tolerance,
        "runtime_sec": parsed["runtime_sec"],
        "exit_code": parsed["exit_code"],
        "route_present": validation["route_present"],
        "valid_tour": validation["valid_tour"],
        "missing_city_count": validation["missing_city_count"],
        "duplicate_city_count": validation["duplicate_city_count"],
        "invalid_indices": " ".join(str(x) for x in validation["invalid_indices"]),
        "wrong_tour_length": validation["wrong_tour_length"],
        "normalized_tour": " ".join(str(x) for x in validation["normalized_route"]),
    }


def run_command(command: list[str], cwd: Path, timeout_sec: float | None = None) -> tuple[str, float, int]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    runtime_sec = time.perf_counter() - start
    text = proc.stdout
    if proc.stderr:
        if text and not text.endswith("\n"):
            text += "\n"
        text += proc.stderr
    return text, runtime_sec, proc.returncode


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def safe_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def bool_from_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["implementation"]), str(row["dataset"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (implementation, dataset), group_rows in sorted(grouped.items()):
        lengths = [safe_float(row.get("recomputed_best_length")) for row in group_rows]
        lengths = [value for value in lengths if value is not None]
        runtimes = [safe_float(row.get("runtime_sec")) for row in group_rows]
        runtimes = [value for value in runtimes if value is not None]
        valid_count = sum(1 for row in group_rows if bool_from_text(row.get("valid_tour", False)))

        dataset_key = dataset_tag(dataset)
        reference = REFERENCE_LENGTHS.get(dataset_key)
        def success_rate(threshold: float) -> float | None:
            if reference is None or not lengths:
                return None
            success = sum(1 for value in lengths if value <= reference * (1.0 + threshold))
            return success / len(lengths)

        summaries.append({
            "implementation": implementation,
            "dataset": dataset,
            "runs": len(group_rows),
            "valid_tour_rate": valid_count / len(group_rows) if group_rows else None,
            "min_best_length": min(lengths) if lengths else None,
            "max_best_length": max(lengths) if lengths else None,
            "mean_best_length": statistics.mean(lengths) if lengths else None,
            "median_best_length": statistics.median(lengths) if lengths else None,
            "stdev_best_length": statistics.stdev(lengths) if len(lengths) > 1 else 0.0 if lengths else None,
            "mean_runtime_sec": statistics.mean(runtimes) if runtimes else None,
            "median_runtime_sec": statistics.median(runtimes) if runtimes else None,
            "stdev_runtime_sec": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0 if runtimes else None,
            "success_rate_within_0pct": success_rate(0.0),
            "success_rate_within_1pct": success_rate(0.01),
            "success_rate_within_5pct": success_rate(0.05),
            "success_rate_within_10pct": success_rate(0.10),
            "reference_length": reference,
        })
    return summaries