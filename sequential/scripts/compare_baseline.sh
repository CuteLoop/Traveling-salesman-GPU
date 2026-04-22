#!/usr/bin/env bash
# ===========================================================================
# compare_baseline.sh — Phase 10: Regression comparator
#
# Compares best_distance values between a baseline CSV and a current CSV
# for matching (instance, seed) pairs.
#
# Usage:
#   bash scripts/compare_baseline.sh results/serial_baseline.csv results/current.csv
#
# Exit 0 if all distances are within 0.1% relative tolerance.
# Exit 1 if any deviation exceeds tolerance, printing the exact deviations.
# ===========================================================================

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <baseline.csv> <current.csv>"
    echo "  Compares best_distance for matching (instance, seed) pairs."
    exit 2
fi

BASELINE="$1"
CURRENT="$2"

if [ ! -f "$BASELINE" ]; then
    echo "Error: baseline file not found: $BASELINE"
    exit 2
fi
if [ ! -f "$CURRENT" ]; then
    echo "Error: current file not found: $CURRENT"
    exit 2
fi

echo "=== Regression Comparison ==="
echo "  Baseline: $BASELINE"
echo "  Current:  $CURRENT"
echo "  Tolerance: 0.1% relative"
echo ""

# Use awk to compare matching (instance, seed) rows.
# CSV header: instance,seed,population,generations,best_distance,elapsed_sec
awk -F',' '
BEGIN {
    fail = 0
    compared = 0
    tolerance = 0.001   # 0.1% relative
}

# Skip headers
NR == FNR && FNR == 1 { next }
NR != FNR && FNR == 1 { next }

# First file (baseline): store best_distance keyed by instance+seed
NR == FNR {
    key = $1 "," $2
    baseline[key] = $5
    next
}

# Second file (current): compare
{
    key = $1 "," $2
    if (key in baseline) {
        compared++
        base_val = baseline[key] + 0.0
        curr_val = $5 + 0.0

        if (base_val == 0.0) {
            if (curr_val != 0.0) {
                fail++
                printf "  FAIL: %-30s  baseline=%.6f  current=%.6f  (baseline is zero)\n", key, base_val, curr_val
            }
        } else {
            rel_diff = (curr_val - base_val) / base_val
            if (rel_diff < 0) rel_diff = -rel_diff

            if (rel_diff > tolerance) {
                fail++
                printf "  FAIL: %-30s  baseline=%.6f  current=%.6f  rel_diff=%.6f%%\n", key, base_val, curr_val, rel_diff * 100
            } else {
                printf "  PASS: %-30s  baseline=%.6f  current=%.6f  rel_diff=%.6f%%\n", key, base_val, curr_val, rel_diff * 100
            }
        }
    }
}

END {
    printf "\n%d pairs compared, %d failures.\n", compared, fail
    if (compared == 0) {
        print "WARNING: No matching (instance, seed) pairs found!"
        exit 1
    }
    exit (fail > 0) ? 1 : 0
}
' "$BASELINE" "$CURRENT"
