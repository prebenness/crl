#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/bootstrap_venv.sh
#
# Optional env vars:
#   PY_BIN=python3.12 bash scripts/bootstrap_venv.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_VERSION="$(cat "$ROOT_DIR/.python-version")"
DEFAULT_PY="python${PY_VERSION}"
PY_BIN="${PY_BIN:-$DEFAULT_PY}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PY_BIN' not found in PATH."
  echo "Set PY_BIN explicitly, e.g.:"
  echo "  PY_BIN=python3.12 bash scripts/bootstrap_venv.sh"
  exit 1
fi

"$PY_BIN" -m venv "$ROOT_DIR/venv"
# shellcheck source=/dev/null
source "$ROOT_DIR/venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"

python - <<'PY'
import sys
print(f"Bootstrap complete with Python {sys.version.split()[0]}")
PY
