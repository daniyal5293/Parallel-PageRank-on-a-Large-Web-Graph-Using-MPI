#!/usr/bin/env bash
# Measures the machine's own scaling ceiling (compute, memory bandwidth, random
# gathers) for the same process counts and binding as the benchmarks.
#   bash scripts/probe_platform.sh        -> results/platform.csv
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/platform_probe
RESULTS_DIR="${RESULTS_DIR:-results}"
OUT="$RESULTS_DIR/platform.csv"
mkdir -p "$RESULTS_DIR"
CORES=$(( $(nproc) / 2 ))
echo "procs,repeat,t_compute,t_stream,stream_gbs,t_gather,gather_mreads" > "$OUT.tmp"
for r in 0 1 2; do
    for p in 1 2 3 4 6 8; do
        if [ "$p" -le "$CORES" ]; then bind="--map-by core --bind-to core"
        else bind="--use-hwthread-cpus --map-by core --bind-to hwthread"; fi
        mpirun -np "$p" $bind ./build/platform_probe --json /tmp/probe.json
        python3 -c "
import json; d = json.load(open('/tmp/probe.json'))
print(f\"{d['procs']},$r,{d['t_compute']},{d['t_stream']},{d['stream_gbs']},{d['t_gather']},{d['gather_mreads']}\")" >> "$OUT.tmp"
    done
done
mv "$OUT.tmp" "$OUT"   # only a complete file replaces the old one
echo "wrote $OUT"
