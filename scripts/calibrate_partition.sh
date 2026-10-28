#!/usr/bin/env bash
# Calibrates the per-vertex weight w of the edge-balanced partitioner
# (rank cost = in-edges + w * vertices) by measuring solve time and compute
# imbalance for several w at P processes.
#   bash scripts/calibrate_partition.sh [P]      -> results/calibration.csv
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-$HOME/pdc-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
P="${1:-4}"
OUT="$RESULTS_DIR/calibration.csv"
mkdir -p "$RESULTS_DIR"
echo "dataset,procs,vertex_weight,t_solve,imbalance" > "$OUT.tmp"
for ds in web-Google web-BerkStan soc-LiveJournal1; do
    for w in 0 1 2 3 4 6 8; do
        mpirun -np "$P" --map-by core --bind-to core ./build/pagerank_mpi --graph "$DATA_DIR/$ds.bin" \
            --mode alltoallv --partition edge --vertex-weight "$w" --iters 30 --json /tmp/cal.json >/dev/null
        python3 -c "
import json; d = json.load(open('/tmp/cal.json'))
print('$ds,$P,$w,%.4f,%.3f' % (d['t_solve'], d['imbalance']))" | tee -a "$OUT.tmp"
    done
done
mv "$OUT.tmp" "$OUT"   # only a complete file replaces the old one
