#!/usr/bin/env bash
# Converts the downloaded graphs to binary CSR and generates the R-MAT graphs
# used for strong and weak scaling.
#   bash scripts/prepare_datasets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-$HOME/pdc-data}"
RMAT_SCALES="${RMAT_SCALES:-20 21 22 23}"
mkdir -p "$DATA_DIR"
make -s all

# SNAP format: "# comment" lines then "src<TAB>dst".
for gz in "$DATA_DIR"/raw/*.txt.gz; do
    [ -e "$gz" ] || continue
    name=$(basename "$gz" .txt.gz)
    if [ -s "$DATA_DIR/$name.bin" ]; then echo "have $name.bin"; continue; fi
    echo "converting $name (SNAP)"
    pigz -dc "$gz" | ./build/convert "$DATA_DIR/$name.bin"
done

# KONECT format: archive with <name>/out.<name>, "% comment" lines then "src dst".
for bz in "$DATA_DIR"/raw/*.tar.bz2; do
    [ -e "$bz" ] || continue
    name=$(basename "$bz" .tar.bz2)
    if [ -s "$DATA_DIR/$name.bin" ]; then echo "have $name.bin"; continue; fi
    echo "converting $name (KONECT)"
    unbz=bzip2; command -v lbzip2 >/dev/null && unbz=lbzip2
    tar -I "$unbz" -xOf "$bz" --wildcards '*/out.*' | ./build/convert "$DATA_DIR/$name.bin"
done

for s in $RMAT_SCALES; do
    out="$DATA_DIR/rmat-$s.bin"
    if [ -s "$out" ]; then echo "have rmat-$s.bin"; continue; fi
    echo "generating rmat-$s"
    ./build/gen_rmat "$out" --scale "$s" --edgefactor 16 --seed 42
done
ls -lh "$DATA_DIR"/*.bin
