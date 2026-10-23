#!/usr/bin/env bash
# Creates the Python venv used by tests/benchmarks/plots (run as your normal user).
set -euo pipefail
VENV="${VENV:-$HOME/.venvs/pdc}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q numpy scipy pandas matplotlib networkx
"$VENV/bin/python" -c "import numpy, networkx, matplotlib, pandas; print('venv ready:', '$VENV')"
