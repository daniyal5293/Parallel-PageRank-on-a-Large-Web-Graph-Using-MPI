#!/usr/bin/env bash
# Calibrates the per-vertex weight w of the edge-balanced partitioner
# (rank cost = in-edges + w * vertices) by measuring solve time and compute
# imbalance for several w at P processes.
#   bash scripts/calibrate_partition.sh [P]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-$HOME/pdc-data}"
P="${1:-4}"
OUT=results/calibration.csv
mkdir -p results
echo "dataset,procs,vertex_weight,t_solve,imbalance" > "$OUT"
for ds in web-Google web-BerkStan soc-LiveJournal1; do
    for w in 0 1 2 3 4 6 8; do
        mpirun -np "$P" --map-by core --bind-to core ./build/pagerank_mpi --graph "$DATA_DIR/$ds.bin" \
            --mode alltoallv --partition edge --vertex-weight "$w" --iters 30 --json /tmp/cal.json >/dev/null
        python3 -c "
import json; d = json.load(open('/tmp/cal.json'))
print('$ds,$P,$w,%.4f,%.3f' % (d['t_solve'], d['imbalance']))" | tee -a "$OUT"
    done
done
