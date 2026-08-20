#!/usr/bin/env python3
"""Dependency-light magnetometer detector smoke test.

This intentionally does not fetch network data.  It exercises the canonical
QDC/residual/classification path with deterministic synthetic data so the
repository-wide demo runner has a fast, reproducible self-test.
"""
from __future__ import annotations

import numpy as np

from magnetometer_demo import run_analysis


def main() -> int:
    cadence_s = 60
    n = 24 * 60
    t = np.arange(n, dtype=float)
    quiet = 50000.0 + 15.0 * np.sin(2.0 * np.pi * t / (24.0 * 60.0))
    disturbance = np.zeros(n)
    disturbance[8 * 60 : 10 * 60] = 60.0
    disturbance[10 * 60 : 10 * 60 + 20] += np.linspace(0.0, 140.0, 20)
    signal = quiet + disturbance

    result = run_analysis(signal, cadence_s, label="synthetic self-test")

    for key in ("baseline", "residual", "flags", "validation"):
        if key not in result or len(result[key]) != n:
            raise AssertionError(f"invalid detector result: {key}")

    if not np.isfinite(result["baseline"]).all():
        raise AssertionError("baseline contains non-finite values")

    print("Magnetometer detector self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
