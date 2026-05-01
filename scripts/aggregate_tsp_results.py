from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tsp_harness import read_csv_rows, summarize_rows, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate raw TSP benchmark CSV into summary CSV, markdown, and plot-ready CSV.",
    )
    parser.add_argument("raw_csv", help="Raw CSV from scripts/run_seed_benchmark.py")
    parser.add_argument("--output-dir", help="Optional output directory; defaults next to raw CSV")
    return parser.parse_args()


def markdown_table(rows: list[dict[str, object]]) -> str:
    def fmt_number(value: object, decimals: int = 4) -> str:
        if value in (None, "", "None"):
            return "-"
        return f"{float(value):.{decimals}f}"

    def fmt_percent(value: object) -> str:
        if value in (None, "", "None"):
            return "-"
        return f"{float(value):.2%}"

    header = "| Implementation | Dataset | Runs | Mean best | Std best | Mean runtime (s) | Std runtime (s) | Valid rate | Within 1% | Within 5% |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {implementation} | {dataset} | {runs} | {mean_best} | {stdev_best} | {mean_runtime} | {stdev_runtime} | {valid_rate} | {within_1} | {within_5} |".format(
                implementation=row["implementation"],
                dataset=row["dataset"],
                runs=row["runs"],
                mean_best=fmt_number(row.get("mean_best_length")),
                stdev_best=fmt_number(row.get("stdev_best_length")),
                mean_runtime=fmt_number(row.get("mean_runtime_sec")),
                stdev_runtime=fmt_number(row.get("stdev_runtime_sec")),
                valid_rate=fmt_percent(row.get("valid_tour_rate")),
                within_1=fmt_percent(row.get("success_rate_within_1pct")),
                within_5=fmt_percent(row.get("success_rate_within_5pct")),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    rows = read_csv_rows(raw_csv)
    summary_rows = summarize_rows(rows)

    output_dir = Path(args.output_dir) if args.output_dir else raw_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = raw_csv.stem
    normalized_raw = output_dir / f"{stem}_normalized_raw.csv"
    summary_csv = output_dir / f"{stem}_summary.csv"
    plot_csv = output_dir / f"{stem}_plot_ready.csv"
    markdown_path = output_dir / f"{stem}_summary.md"

    write_csv(normalized_raw, rows)
    write_csv(summary_csv, summary_rows)
    write_csv(plot_csv, summary_rows)
    markdown_path.write_text(markdown_table(summary_rows), encoding="utf-8")

    print(f"Wrote {normalized_raw}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_csv}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())