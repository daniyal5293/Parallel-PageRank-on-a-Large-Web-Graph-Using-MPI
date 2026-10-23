#!/usr/bin/env bash
# Downloads the SNAP graphs used in the experiments into $DATA_DIR/raw.
#   bash scripts/download_datasets.sh [web-Google web-BerkStan soc-LiveJournal1]
#
# Tries the canonical SNAP site first and falls back to the KONECT mirror of
# the same graphs (snap.stanford.edu is unreachable from some networks).
set -euo pipefail
DATA_DIR="${DATA_DIR:-$HOME/pdc-data}"
mkdir -p "$DATA_DIR/raw"

DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(web-Google web-BerkStan soc-LiveJournal1)

fetch() {  # fetch <url> <file>
    curl -fL --retry 3 --connect-timeout 20 -sS -o "$2.part" "$1" && mv "$2.part" "$2"
}

for name in "${DATASETS[@]}"; do
    snap="$DATA_DIR/raw/$name.txt.gz"
    konect="$DATA_DIR/raw/$name.tar.bz2"
    if [ -s "$snap" ] || [ -s "$konect" ]; then echo "have $name"; continue; fi
    echo "downloading $name ..."
    if fetch "https://snap.stanford.edu/data/$name.txt.gz" "$snap"; then
        echo "  from SNAP"
    elif fetch "http://konect.cc/files/download.tsv.$name.tar.bz2" "$konect"; then
        echo "  from KONECT mirror"
    else
        rm -f "$snap.part" "$konect.part"
        echo "  FAILED to download $name" >&2
    fi
done
ls -lh "$DATA_DIR/raw"
