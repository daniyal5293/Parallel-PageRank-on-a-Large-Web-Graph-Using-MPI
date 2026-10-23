#!/usr/bin/env bash
# Full experiment pipeline: build, test, benchmark, calibrate, plot, dashboard.
#   bash scripts/run_all.sh [bench.py options]      (~1.5 h; run on AC power with the lid open)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$HOME/.venvs/pdc/bin/python}"
mkdir -p results

make -s all
echo "== correctness tests"
make -s test | tee results/tests.log
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
