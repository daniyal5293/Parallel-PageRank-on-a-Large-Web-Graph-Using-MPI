#!/usr/bin/env bash
# Full experiment pipeline: build, test, benchmark, calibrate, probe, plot, dashboard.
#   bash scripts/run_all.sh [bench.py options]
#   RESULTS_DIR=results-check bash scripts/run_all.sh   # fresh run into a separate folder
# Takes ~80 min for a fresh results folder; run on AC power with the lid open.
# Safe to interrupt (Ctrl+C) and re-run: finished benchmark runs are skipped,
# and the dashboard is only rewritten in the very last step.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$HOME/.venvs/pdc/bin/python}"
export RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

make -s all
echo "== correctness tests"
make -s test | tee "$RESULTS_DIR/tests.log.tmp"
mv "$RESULTS_DIR/tests.log.tmp" "$RESULTS_DIR/tests.log"
echo "== benchmarks"
"$PY" -u scripts/bench.py "$@"
echo "== partition calibration"
bash scripts/calibrate_partition.sh 4
echo "== platform ceiling probe"
bash scripts/probe_platform.sh
echo "== plots and dashboard"
"$PY" scripts/plot_results.py
"$PY" scripts/build_dashboard.py
echo "ALL DONE"
