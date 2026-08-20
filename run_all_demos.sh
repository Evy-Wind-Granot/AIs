#!/usr/bin/env bash
# Canonical launcher for the three instrument demos.
# The implementation lives in run_all_demos.py so paths are defined once.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_CMD:-python3}" "$ROOT/run_all_demos.py" "$@"
