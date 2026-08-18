#!/usr/bin/env bash
# Cascadia Sentinel — one-command environment setup and demo runner.
# All dependency definitions live in run_all_demos.py; no requirements files are used.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
MOIRAI_VENV="${MOIRAI_VENV:-$ROOT_DIR/venv_moirai}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

usage() {
    cat <<'EOF'
Usage:
  ./run_all_demos.sh --self-test
  ./run_all_demos.sh --real-data
  ./run_all_demos.sh --install-only
  ./run_all_demos.sh --clean-install --self-test
EOF
}

MODE=""
INSTALL_ONLY=0
CLEAN_INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --self-test) MODE="--self-test" ;;
        --real-data) MODE="--real-data" ;;
        --install-only) INSTALL_ONLY=1 ;;
        --clean-install) CLEAN_INSTALL=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage; exit 1 ;;
    esac
done

command -v "$PYTHON_CMD" >/dev/null 2>&1 || { echo "Python 3.10+ is required" >&2; exit 1; }

if [[ "$CLEAN_INSTALL" -eq 1 ]]; then
    rm -rf "$VENV_DIR" "$MOIRAI_VENV"
fi

[[ -d "$VENV_DIR" ]] || "$PYTHON_CMD" -m venv "$VENV_DIR"
[[ -d "$MOIRAI_VENV" ]] || "$PYTHON_CMD" -m venv "$MOIRAI_VENV"

"$VENV_DIR/bin/python" "$ROOT_DIR/run_all_demos.py" --setup-main
"$MOIRAI_VENV/bin/python" "$ROOT_DIR/run_all_demos.py" --setup-moirai

if [[ "$INSTALL_ONLY" -eq 1 ]]; then
    echo "Environment setup complete."
    exit 0
fi

[[ -n "$MODE" ]] || { echo "Choose --self-test or --real-data" >&2; usage; exit 1; }

export VENV_DIR MOIRAI_VENV
exec "$VENV_DIR/bin/python" "$ROOT_DIR/run_all_demos.py" "$MODE"
