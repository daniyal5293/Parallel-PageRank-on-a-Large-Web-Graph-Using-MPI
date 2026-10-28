#!/usr/bin/env bash
# Locality experiment: natural vertex order vs decreasing out-degree order.
# Runs 30 fixed iterations, sequential and 4 processes (alltoallv, edge),
# 3 repeats each, and writes the median solve times to results/reorder.csv.
#   bash scripts/reorder_experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-$HOME/pdc-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
OUT="$RESULTS_DIR/reorder.csv"
mkdir -p "$RESULTS_DIR"
make -s build/reorder build/pagerank_seq build/pagerank_mpi
echo "dataset,order,procs,t_solve_median" > "$OUT.tmp"
solve() { python3 -c "import json; print(json.load(open('/tmp/reorder_run.json'))['t_solve'])"; }
for ds in web-Google soc-LiveJournal1; do
    [ -s "$DATA_DIR/$ds-degree.bin" ] || ./build/reorder "$DATA_DIR/$ds.bin" "$DATA_DIR/$ds-degree.bin"
    for order in natural degree; do
        g="$DATA_DIR/$ds.bin"; [ "$order" = degree ] && g="$DATA_DIR/$ds-degree.bin"
        for p in 1 4; do
            times=()
            for r in 1 2 3; do
                if [ "$p" -eq 1 ]; then
                    ./build/pagerank_seq --graph "$g" --iters 30 --json /tmp/reorder_run.json >/dev/null
                else
                    mpirun -np "$p" --map-by core --bind-to core ./build/pagerank_mpi --graph "$g" --iters 30 \
                        --mode alltoallv --partition edge --json /tmp/reorder_run.json >/dev/null
                fi
                times+=("$(solve)")
            done
            med=$(printf '%s\n' "${times[@]}" | sort -g | sed -n 2p)
            echo "$ds,$order,$p,$med" | tee -a "$OUT.tmp"
        done
    done
done
mv "$OUT.tmp" "$OUT"   # only a complete file replaces the old one
