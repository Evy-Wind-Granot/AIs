"""Baseline/QDC processing primitives.

This module owns the pure numerical baseline operations used by the magnetometer
pipeline.  It intentionally has no network, cache, CLI, or pipeline-state
responsibilities.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg


def handle_gaps(series: pd.Series, max_gap_samples: int = 3) -> pd.Series:
    """Regularize a time series and linearly fill short gaps."""
    if series.empty:
        return series

    deltas = series.index.to_series().diff().dropna()
    if deltas.empty:
        return series

    freq_s = max(1, int(deltas.median().total_seconds()))
    freq = pd.Timedelta(seconds=freq_s)
    regular_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=freq,
        tz="UTC",
    )
    series = series.reindex(regular_index)
    return series.interpolate(method="linear", limit=max_gap_samples)


@lru_cache(maxsize=8)
def _hanning(m: int) -> np.ndarray:
    """Return an immutable cached Hann window."""
    w = np.hanning(m)
    w.flags.writeable = False
    return w


def build_design_matrix(t_hours: np.ndarray) -> np.ndarray:
    """Build the harmonic QDC design matrix used by the legacy pipeline."""
    t = np.asarray(t_hours, dtype=float)
    cols = [
        np.ones_like(t),
        np.sin(2 * np.pi * t / 24),
        np.cos(2 * np.pi * t / 24),
        np.sin(2 * np.pi * t / 12),
        np.cos(2 * np.pi * t / 12),
        np.sin(2 * np.pi * t / 8),
        np.cos(2 * np.pi * t / 8),
        np.sin(2 * np.pi * t / 6),
        np.cos(2 * np.pi * t / 6),
    ]
    return np.column_stack(cols)


def robust_harmonic_baseline(
    x: np.ndarray,
    cadence_s: float,
    n_iter: int = 4,
    outlier_threshold_nt: float = 30.0,
    t_hours: Optional[np.ndarray] = None,
    design_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the robust harmonic quiet-day baseline.

    The numerical algorithm and weighting rules intentionally match the
    validated implementation in the production pipeline.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if t_hours is None and design_matrix is None:
        t_hours = np.arange(n) * cadence_s / 3600.0

    if design_matrix is not None:
        A = design_matrix
    else:
        assert t_hours is not None
        A = build_design_matrix(t_hours)

    valid = np.isfinite(x)
    w = np.ones(n)
    w[~valid] = 0.0
    coeffs = np.zeros(A.shape[1])

    for _ in range(n_iter):
        if valid.sum() < A.shape[1]:
            break

        Aw = A[valid] * w[valid, np.newaxis]
        xw = x[valid] * w[valid]
        coeffs, *_ = linalg.lstsq(Aw, xw, check_finite=False)[:2]
        pred = A @ coeffs
        resid = x - pred
        mad = np.median(np.abs(resid[valid] - np.median(resid[valid])))
        sigma = 1.4826 * mad + 1e-12

        w = np.ones(n)
        w[~valid] = 0.0
        w[np.abs(resid) > outlier_threshold_nt] = 0.1
        w[np.abs(resid) > 3 * sigma] = 0.01

    return A @ coeffs, coeffs


__all__ = [
    "build_design_matrix",
    "robust_harmonic_baseline",
    "handle_gaps",
]
