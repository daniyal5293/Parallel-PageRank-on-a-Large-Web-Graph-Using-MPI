#!/usr/bin/env bash
# Correctness test suite (run with `make test`).
#   1. sequential PageRank vs networkx.pagerank on hand-made and R-MAT graphs
#   2. MPI PageRank vs sequential for every mode x partition x process count,
#      including runs where some ranks own no vertices
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-$HOME/.venvs/pdc/bin/python}"
BUILD=build
TMP=tests/tmp
MPIRUN="mpirun --oversubscribe"
PROCS="${PROCS:-1 2 3 4 5 8}"
MODES="allgather alltoallv overlap"
PARTITIONS="vertex edge"
ITERS=60   # fixed iteration count so MPI and sequential do identical work

rm -rf "$TMP" && mkdir -p "$TMP"
pass=0
fail=0
check() {  # check <description> <command...>
    local desc="$1"; shift
    if out=$("$@" 2>&1); then
        pass=$((pass + 1))
        printf '  PASS  %-48s %s\n' "$desc" "$(echo "$out" | tail -1)"
    else
        fail=$((fail + 1))
        printf '  FAIL  %s\n%s\n' "$desc" "$out"
    fi
}

echo "== preparing test graphs"
for f in tests/graphs/*.txt; do
    $BUILD/convert "$TMP/$(basename "${f%.txt}").bin" --input "$f" 2>/dev/null || { echo "convert failed: $f"; exit 1; }
done
$BUILD/gen_rmat "$TMP/rmat12.bin" --scale 12 --seed 7 2>/dev/null
$BUILD/gen_rmat "$TMP/rmat15.bin" --scale 15 --seed 3 2>/dev/null
GRAPHS=$(ls "$TMP"/*.bin)

echo "== sequential vs networkx"
for g in $GRAPHS; do
    name=$(basename "$g" .bin)
    $BUILD/pagerank_seq --graph "$g" --tol 1e-15 --max-iter 1000 --out "$g.nx.f64" >/dev/null
    check "seq $name" "$PY" tests/check_networkx.py "$g" "$g.nx.f64"
done
check "cycle4 uniform 0.25" "$PY" -c "
import numpy as np, sys
r = np.fromfile('$TMP/cycle4.bin.nx.f64'); print('max |r-0.25| =', abs(r-0.25).max()); sys.exit(int(abs(r-0.25).max() > 1e-12))"

echo "== MPI vs sequential (procs: $PROCS)"
for g in $GRAPHS; do
    name=$(basename "$g" .bin)
    $BUILD/pagerank_seq --graph "$g" --iters $ITERS --out "$g.seq.f64" --json "$g.seq.json" >/dev/null
    for p in $PROCS; do
        for mode in $MODES; do
            for part in $PARTITIONS; do
                out="$TMP/mpi.f64"
                if ! $MPIRUN -np "$p" $BUILD/pagerank_mpi --graph "$g" --mode "$mode" --partition "$part" \
                        --iters $ITERS --out "$out" --json "$TMP/mpi.json" >"$TMP/mpi.log" 2>&1; then
                    fail=$((fail + 1)); echo "  FAIL  $name p=$p $mode/$part (mpirun error)"; cat "$TMP/mpi.log"; continue
                fi
                check "$name p=$p $mode/$part" "$PY" tests/compare_ranks.py "$g.seq.f64" "$out" \
                    --json-a "$g.seq.json" --json-b "$TMP/mpi.json"
            done
        done
    done
done

echo "== convergence: MPI stops at the same iteration as sequential"
g="$TMP/rmat15.bin"
$BUILD/pagerank_seq --graph "$g" --json "$TMP/conv.seq.json" >/dev/null
for mode in $MODES; do
    $MPIRUN -np 4 $BUILD/pagerank_mpi --graph "$g" --mode "$mode" --json "$TMP/conv.mpi.json" >/dev/null 2>&1
    check "iterations rmat15 $mode p=4" "$PY" -c "
import json, sys
a = json.load(open('$TMP/conv.seq.json')); b = json.load(open('$TMP/conv.mpi.json'))
print('seq', a['iterations'], 'mpi', b['iterations'], 'converged', b['converged'])
sys.exit(not (a['iterations'] == b['iterations'] and b['converged']))"
done

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
