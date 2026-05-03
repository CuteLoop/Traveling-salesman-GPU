from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
README_PATH = ROOT / "README.md"

STANDARD_SETTINGS = (
    "RUNS=20, ISLANDS=256, GENERATIONS=2000, MUTATION=0.03, "
    "ELITE_POP=2, ELITE_HYBRID=4, POP_HYBRID=512, SEED_BASE=100"
)

DATASETS = [
    {
        "tag": "smoke_20",
        "dataset": "sequential/tests/fixtures/smoke_20.tsp",
        "title": "smoke_20",
    },
    {
        "tag": "berlin52",
        "dataset": "data/berlin52.tsp",
        "title": "berlin52",
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_job_id(path: Path, prefix: str, tag: str) -> str | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_{re.escape(tag)}_(\d+)\.csv", path.name)
    return match.group(1) if match else None


def latest_completed_pair(tag: str) -> tuple[Path, Path]:
    avg_candidates = sorted(
        RESULTS_DIR.glob(f"cuda_all_variants_avg_{tag}_*.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for avg_path in avg_candidates:
        job_id = parse_job_id(avg_path, "cuda_all_variants_avg", tag)
        if not job_id:
            continue
        runs_path = RESULTS_DIR / f"cuda_all_variants_runs_{tag}_{job_id}.csv"
        if not runs_path.exists():
            continue
        avg_rows = read_csv_rows(avg_path)
        run_rows = read_csv_rows(runs_path)
        if avg_rows and run_rows:
            return runs_path, avg_path
    raise FileNotFoundError(f"No completed runs/avg CSV pair found for {tag}")


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "NA":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def sample_stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    rounded_int = round(value)
    if abs(value - rounded_int) < 1e-12:
        return str(int(rounded_int))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_variant_order(rows: list[dict[str, str]]) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        variant = row.get("variant", "").strip()
        if variant:
            ordered.setdefault(variant, None)
    return list(ordered)


def build_table_rows(runs_path: Path, avg_path: Path) -> list[str]:
    run_rows = [row for row in read_csv_rows(runs_path) if row.get("status") == "ok"]
    avg_rows = read_csv_rows(avg_path)
    avg_by_variant = {row["variant"]: row for row in avg_rows if row.get("variant")}

    table_lines = [
        "| Variant | Best length | Avg length | Length stddev | Best time (ms) | Avg time (ms) | Time stddev (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for variant in build_variant_order(run_rows + avg_rows):
        variant_runs = [row for row in run_rows if row.get("variant") == variant]
        length_runs = [row for row in variant_runs if parse_int(row.get("reported_length")) is not None]
        time_runs = [row for row in variant_runs if parse_float(row.get("cuda_elapsed_ms")) is not None]

        best_length = None
        if length_runs:
            best_row = min(
                length_runs,
                key=lambda row: (
                    parse_int(row.get("reported_length")),
                    parse_float(row.get("cuda_elapsed_ms")) if parse_float(row.get("cuda_elapsed_ms")) is not None else float("inf"),
                ),
            )
            best_length = parse_int(best_row.get("reported_length"))

        avg_row = avg_by_variant.get(variant, {})
        avg_length = parse_float(avg_row.get("mean_reported_length"))
        std_length = parse_float(avg_row.get("stddev_reported_length"))
        if avg_length is None and length_runs:
            values = [float(parse_int(row.get("reported_length"))) for row in length_runs]
            avg_length = sum(values) / len(values)
            std_length = sample_stddev(values)

        best_time = None
        if time_runs:
            best_time = min(parse_float(row.get("cuda_elapsed_ms")) for row in time_runs)

        avg_time = parse_float(avg_row.get("mean_cuda_elapsed_ms"))
        std_time = parse_float(avg_row.get("stddev_cuda_elapsed_ms"))
        if avg_time is None and time_runs:
            values = [parse_float(row.get("cuda_elapsed_ms")) for row in time_runs]
            values = [value for value in values if value is not None]
            avg_time = sum(values) / len(values)
            std_time = sample_stddev(values)

        table_lines.append(
            "| `{variant}` | {best_length} | {avg_length} | {std_length} | {best_time} | {avg_time} | {std_time} |".format(
                variant=variant,
                best_length=format_number(best_length),
                avg_length=format_number(avg_length),
                std_length=format_number(std_length),
                best_time=format_number(best_time),
                avg_time=format_number(avg_time),
                std_time=format_number(std_time),
            )
        )

    return table_lines


def generate_headline_results() -> str:
    sections = [
        "## Headline Results",
        "",
        "Detailed methodology lives in [docs/EXPERIMENTS/README.md](docs/EXPERIMENTS/README.md).",
        "",
    ]

    for dataset in DATASETS:
        runs_path, avg_path = latest_completed_pair(dataset["tag"])
        sections.extend(
            [
                f"### {dataset['title']}",
                "",
                "Latest completed standardized CUDA sweep:",
                "",
                f"- dataset: `{dataset['dataset']}`",
                "- runner: `slurm/run_cuda_all_variants_csv.slurm`",
                f"- settings: `{STANDARD_SETTINGS}`",
                "- latest completed artifacts used here:",
                f"  - `results/{runs_path.name}`",
                f"  - `results/{avg_path.name}`",
                "",
                "Per-implementation results from that sweep:",
                "",
            ]
        )
        sections.extend(build_table_rows(runs_path, avg_path))
        sections.append("")

        if dataset["tag"] == "smoke_20":
            sections.extend(
                [
                    "Notes:",
                    "",
                    "- Many variants tie at the smoke_20 optimum length `73`, so the timing columns are what separate them under this configuration.",
                    "- On smoke_20, `cuda_ga_gpu_pop_bitset` is the fastest timed variant among the implementations that also hit the best observed length.",
                    "",
                ]
            )
        else:
            sections.extend(
                [
                    "Interpretation:",
                    "",
                    "- `cuda_ga_b5_bigpop` is the best quality variant in the latest completed Berlin52 sweep and reaches the optimum `7542` consistently across runs.",
                    "- `cuda_ga_gpu_pop_global_dist` is the fastest timed CUDA variant on Berlin52 under the standardized settings, but it does not match the solution quality of `cuda_ga_b5_bigpop`.",
                    "- The current Berlin52 tradeoff is still quality versus speed, not one variant dominating both.",
                    "",
                ]
            )

    sections.extend(
        [
            "Timing caveat:",
            "",
            "- `NA` timing cells in these tables come from historical completed result files that were generated before those binaries were rerun with the now-standard timing export.",
            "- The newest `cuda_ga_no_greedy` and `cuda_ga_c1` through `cuda_ga_c5` variants will appear automatically once a completed standardized sweep writes non-empty run and average CSVs for them.",
            "",
        ]
    )

    return "\n".join(sections)


def main() -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")
    replacement = generate_headline_results()
    updated = re.sub(
        r"## Headline Results\n.*?\n## Optimization Story",
        replacement + "\n## Optimization Story",
        readme_text,
        flags=re.DOTALL,
    )
    if updated == readme_text:
        raise RuntimeError("Failed to locate README headline results section")
    README_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {README_PATH}")


if __name__ == "__main__":
    main()