from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tsp_harness import audit_paths, read_csv_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TSP solver outputs against a canonical TSPLIB evaluator.",
    )
    parser.add_argument("--raw-csv", help="CSV with dataset/output_txt/best_tour_csv columns to audit")
    parser.add_argument("--dataset", help="Dataset path when auditing one or more result text files")
    parser.add_argument("--output-csv", required=True, help="Path to write audit CSV")
    parser.add_argument("results", nargs="*", help="Result .txt files to audit when --raw-csv is not used")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Absolute tolerance for reported vs recomputed length")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []

    if args.raw_csv:
        for row in read_csv_rows(args.raw_csv):
            audited = audit_paths(
                dataset=row["dataset"],
                output_txt=row["output_txt"],
                best_tour_csv=row.get("best_tour_csv") or None,
                tolerance=args.tolerance,
            )
            audited.update({
                "implementation": row.get("implementation", ""),
                "seed": row.get("seed", ""),
                "repeat_id": row.get("repeat_id", ""),
                "command": row.get("command", ""),
            })
            rows.append(audited)
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required when auditing explicit result files")
        if not args.results:
            raise SystemExit("Provide one or more result files when --raw-csv is not used")
        for result_path in args.results:
            rows.append(audit_paths(args.dataset, result_path, tolerance=args.tolerance))

    fieldnames = [
        "implementation",
        "dataset",
        "dataset_name",
        "edge_weight_type",
        "dimension",
        "seed",
        "repeat_id",
        "command",
        "output_txt",
        "best_tour_csv",
        "reported_best_length",
        "recomputed_best_length",
        "length_abs_error",
        "length_match",
        "runtime_sec",
        "exit_code",
        "route_present",
        "valid_tour",
        "missing_city_count",
        "duplicate_city_count",
        "invalid_indices",
        "wrong_tour_length",
        "normalized_tour",
    ]
    write_csv(Path(args.output_csv), rows, fieldnames=fieldnames)
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())