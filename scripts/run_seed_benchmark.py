from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tsp_harness import (
    IMPLEMENTATIONS,
    IMPLEMENTATION_SETS,
    REFERENCE_LENGTHS,
    SEED_POOL,
    audit_paths,
    command_for,
    dataset_tag,
    load_route_from_artifacts,
    repo_root,
    resolve_binary,
    run_command,
    shell_join,
    write_csv,
    write_tour_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic-repeat or fixed-seed TSP benchmarks and emit raw audit-ready CSV.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset path relative to repo root or absolute path")
    parser.add_argument("--mode", choices=("determinism", "benchmark"), default="benchmark")
    parser.add_argument("--implementation-set", choices=tuple(IMPLEMENTATION_SETS.keys()), default="all")
    parser.add_argument("--implementations", nargs="*", help="Optional explicit implementation keys")
    parser.add_argument("--seeds", nargs="*", type=int, help="Explicit seed list")
    parser.add_argument("--repeats", type=int, help="Override repeat count")
    parser.add_argument("--results-dir", default="results/audit_runs", help="Directory for raw outputs")
    parser.add_argument("--output-csv", help="Path for raw benchmark CSV")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--skip-missing-binaries", action="store_true")
    return parser.parse_args()


def resolved_dataset_path(dataset: str) -> Path:
    path = Path(dataset)
    if path.is_absolute():
        return path
    return repo_root() / path


def selected_implementations(args: argparse.Namespace) -> list[str]:
    if args.implementations:
        return args.implementations
    return list(IMPLEMENTATION_SETS[args.implementation_set])


def selected_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    if args.mode == "determinism":
        return [42]
    return list(SEED_POOL)


def repeat_count(args: argparse.Namespace) -> int:
    if args.repeats is not None:
        return args.repeats
    return 5 if args.mode == "determinism" else 1


def default_output_csv(args: argparse.Namespace) -> Path:
    tag = dataset_tag(args.dataset)
    return repo_root() / "results" / f"{args.mode}_{args.implementation_set}_{tag}_raw.csv"


def main() -> int:
    args = parse_args()
    dataset_path = resolved_dataset_path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    rows = []
    results_dir = repo_root() / args.results_dir / args.mode / dataset_tag(dataset_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    implementations = selected_implementations(args)
    seeds = selected_seeds(args)
    repeats = repeat_count(args)

    for implementation_key in implementations:
        if implementation_key not in IMPLEMENTATIONS:
            raise SystemExit(f"Unknown implementation key: {implementation_key}")
        spec = IMPLEMENTATIONS[implementation_key]

        try:
            resolve_binary(spec)
        except FileNotFoundError as exc:
            if not args.skip_missing_binaries:
                raise
            for seed in seeds:
                for repeat_id in range(1, repeats + 1):
                    rows.append({
                        "implementation": spec.key,
                        "label": spec.label,
                        "comparison_group": spec.family_group,
                        "dataset": str(dataset_path.relative_to(repo_root())).replace("\\", "/") if dataset_path.is_relative_to(repo_root()) else str(dataset_path).replace("\\", "/"),
                        "seed": seed,
                        "repeat_id": repeat_id,
                        "command": "",
                        "runtime_sec": None,
                        "reported_best_length": None,
                        "recomputed_best_length": None,
                        "valid_tour": False,
                        "missing_city_count": None,
                        "duplicate_city_count": None,
                        "output_txt": "",
                        "best_tour_csv": "",
                        "edge_weight_type": "",
                        "dimension": "",
                        "length_abs_error": None,
                        "length_match": False,
                        "exit_code": None,
                        "status": "missing_binary",
                        "error": str(exc),
                        "seed_used_by_solver": spec.uses_seed,
                        "determinism_expectation": spec.determinism_expectation,
                        "rng_scope": spec.rng_scope,
                        "rng_notes": spec.rng_notes,
                        "reference_length": REFERENCE_LENGTHS.get(dataset_tag(dataset_path)),
                    })
            continue

        for seed in seeds:
            for repeat_id in range(1, repeats + 1):
                stem = f"{spec.key}_{dataset_tag(dataset_path)}_seed{seed}_r{repeat_id}"
                output_txt = results_dir / f"{stem}.txt"
                stats_csv = results_dir / f"{stem}_stats.csv"
                best_tour_csv = results_dir / f"{stem}_best_tour.csv"
                command = command_for(
                    spec,
                    str(dataset_path.relative_to(repo_root())).replace("\\", "/") if dataset_path.is_relative_to(repo_root()) else str(dataset_path),
                    seed,
                    str(stats_csv.relative_to(repo_root())).replace("\\", "/") if stats_csv.is_relative_to(repo_root()) else str(stats_csv),
                )

                text, runtime_sec, exit_code = run_command(command, repo_root(), timeout_sec=args.timeout_sec)
                output_txt.write_text(text, encoding="utf-8")

                parsed = load_route_from_artifacts(output_txt, None)
                if parsed["route"]:
                    write_tour_csv(best_tour_csv, parsed["route"])
                audited = audit_paths(
                    dataset=dataset_path,
                    output_txt=output_txt,
                    best_tour_csv=best_tour_csv if best_tour_csv.exists() else None,
                )
                rows.append({
                    "implementation": spec.key,
                    "label": spec.label,
                    "comparison_group": spec.family_group,
                    "dataset": audited["dataset"],
                    "seed": seed,
                    "repeat_id": repeat_id,
                    "command": shell_join(command),
                    "runtime_sec": runtime_sec,
                    "reported_best_length": audited["reported_best_length"],
                    "recomputed_best_length": audited["recomputed_best_length"],
                    "valid_tour": audited["valid_tour"],
                    "missing_city_count": audited["missing_city_count"],
                    "duplicate_city_count": audited["duplicate_city_count"],
                    "output_txt": audited["output_txt"],
                    "best_tour_csv": audited["best_tour_csv"],
                    "edge_weight_type": audited["edge_weight_type"],
                    "dimension": audited["dimension"],
                    "length_abs_error": audited["length_abs_error"],
                    "length_match": audited["length_match"],
                    "exit_code": exit_code,
                    "status": "ok" if exit_code == 0 and audited["valid_tour"] and audited["length_match"] else "audit_failed",
                    "error": "" if exit_code == 0 else f"process exited {exit_code}",
                    "seed_used_by_solver": spec.uses_seed,
                    "determinism_expectation": spec.determinism_expectation,
                    "rng_scope": spec.rng_scope,
                    "rng_notes": spec.rng_notes,
                    "reference_length": REFERENCE_LENGTHS.get(dataset_tag(dataset_path)),
                })

    output_csv = Path(args.output_csv) if args.output_csv else default_output_csv(args)
    write_csv(output_csv, rows)
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())