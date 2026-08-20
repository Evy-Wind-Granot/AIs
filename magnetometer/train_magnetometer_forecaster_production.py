#!/usr/bin/env python3
"""Canonical production forecaster training entry point under magnetometer/."""
from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    root_entrypoint = Path(__file__).resolve().parents[1] / "train_magnetometer_forecaster_production.py"
    runpy.run_path(str(root_entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
