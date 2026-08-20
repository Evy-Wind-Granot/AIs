#!/usr/bin/env python3
"""Canonical magnetometer forecaster training entry point.

The implementation is kept behind the production forecasting package; this file
exists so operational commands live under the magnetometer subsystem.
"""
from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    root_entrypoint = Path(__file__).resolve().parents[1] / "train_magnetometer_forecaster.py"
    runpy.run_path(str(root_entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
