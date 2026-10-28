#!/usr/bin/env bash
# Live demo for the viva (~1 minute). Writes nothing into results/.
#   bash scripts/demo.sh [graph-name]        default: web-Google
# Shows: the sequential baseline, then the MPI version with 4 processes for
# each communication strategy, so the times and identical top-10 lists can be
# compared side by side.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-$HOME/pdc-data}"
G="$DATA_DIR/${1:-web-Google}.bin"
make -s all
[ -s "$G" ] || { echo "graph $G not found; run scripts/download_datasets.sh and scripts/prepare_datasets.sh"; exit 1; }

line() { printf '\n==================== %s ====================\n' "$1"; }

line "1. Sequential PageRank (the baseline)"
./build/pagerank_seq --graph "$G" --topk 5

for mode in allgather alltoallv overlap; do
    line "2. MPI PageRank, 4 processes, mode=$mode"
    mpirun -np 4 --map-by core --bind-to core ./build/pagerank_mpi --graph "$G" --mode "$mode" --partition edge --topk 5
done

line "3. Load balance: vertex vs edge partition (4 processes)"
for part in vertex edge; do
    echo "--- partition=$part"
    mpirun -np 4 --map-by core --bind-to core ./build/pagerank_mpi --graph "$G" --mode alltoallv --partition "$part" \
        --topk 1 | sed -n '4,12p'
done
