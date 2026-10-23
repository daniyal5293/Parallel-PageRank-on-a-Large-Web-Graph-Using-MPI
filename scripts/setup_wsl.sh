#!/usr/bin/env bash
# One-time environment setup inside WSL Ubuntu (or any Debian/Ubuntu box).
#   bash scripts/setup_wsl.sh
# Installs the compiler, OpenMPI and tools, then creates the Python venv used
# by the test, benchmark and plotting scripts.
set -euo pipefail

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
    build-essential make openmpi-bin libopenmpi-dev \
    python3-venv python3-pip pigz lbzip2 curl ca-certificates

if [ "$(id -u)" -ne 0 ]; then
    bash "$(dirname "$0")/make_venv.sh"
else
    echo "Running as root: skipped the venv. Re-run as your normal user to create it."
fi

mpirun --version | head -1
g++ --version | head -1
