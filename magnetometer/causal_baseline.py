#!/usr/bin/env python3
"""Strictly causal harmonic baseline for magnetometer residual generation.

The baseline at time t is fitted only from samples strictly before t. Fits are
updated on a bounded cadence and use a trailing window, preventing future-data
leakage into detector features or validation metrics.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import linalg


def build_design_matrix(t_hours: np.ndarray, t_ref_min: float, t_ref_max: float) -> np.ndarray:
    t = np.asarray(t_hours, dtype=float)
    scale = max(float(t_ref_max) - float(t_ref_min), 1e-12)
    t_norm = np.clip((t - float(t_ref_min)) / scale, -0.5, 1.5)
    return np.column_stack(
        [
            np.ones_like(t),
            t_norm,
            t_norm**2,
            np.sin(2 * np.pi * t / 24),
            np.cos(2 * np.pi * t / 24),
            np.sin(2 * np.pi * t / 12),
            np.cos(2 * np.pi * t / 12),
            np.sin(2 * np.pi * t / 8),
            np.cos(2 * np.pi * t / 8),
            np.sin(2 * np.pi * t / 6),
            np.cos(2 * np.pi * t / 6),
        ]
    )


def robust_harmonic_fit(
    x: np.ndarray,
    cadence_s: float,
    t_hours: np.ndarray,
    n_iter: int = 4,
    outlier_threshold_nt: float = 30.0,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    t = np.asarray(t_hours, dtype=float)
    valid = np.isfinite(values) & np.isfinite(t)
    if valid.sum() < 12:
        return np.full(11, np.nan, dtype=float)
    A = build_design_matrix(t[valid], float(np.min(t[valid])), float(np.max(t[valid])))
    y = values[valid]
    weights = np.ones(y.size, dtype=float)
    coeffs = np.zeros(A.shape[1], dtype=float)
    for _ in range(max(1, int(n_iter))):
        Aw = A * weights[:, None]
        yw = y * weights
        coeffs, *_ = linalg.lstsq(Aw, yw)
        resid = y - A @ coeffs
        mad = float(np.median(np.abs(resid - np.median(resid))))
        sigma = 1.4826 * mad + 1e-12
        weights = np.ones_like(resid)
        weights[np.abs(resid) > float(outlier_threshold_nt)] = 0.1
        weights[np.abs(resid) > 3.0 * sigma] = 0.01
    return coeffs


def compute_causal_qdc_baseline(
    x: np.ndarray,
    cadence_s: float,
    *,
    fit_window_hours: float = 24.0,
    update_minutes: float = 15.0,
    min_history_fraction: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a strictly causal harmonic baseline and residual.

    Every baseline block is predicted using a fit over samples strictly before
    the block. No sample is used to fit its own baseline or any future block.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if cadence_s <= 0 or not np.isfinite(cadence_s):
        raise ValueError("cadence_s must be positive and finite")
    n = values.size
    baseline = np.full(n, np.nan, dtype=float)
    if n == 0:
        return baseline, values.copy()

    window = max(12, int(round(float(fit_window_hours) * 3600.0 / cadence_s)))
    step = max(1, int(round(float(update_minutes) * 60.0 / cadence_s)))
    t = np.arange(n, dtype=float) * float(cadence_s) / 3600.0
    last_coeffs: np.ndarray | None = None

    for block_start in range(window, n, step):
        history_start = max(0, block_start - window)
        history = values[history_start:block_start]
        t_hist = t[history_start:block_start]
        valid_fraction = float(np.isfinite(history).mean()) if history.size else 0.0
        if valid_fraction < float(min_history_fraction):
            continue
        coeffs = robust_harmonic_fit(history, cadence_s, t_hist)
        if np.all(np.isfinite(coeffs)):
            last_coeffs = coeffs
        block_end = min(n, block_start + step)
        if last_coeffs is None:
            continue
        A_pred = build_design_matrix(t[block_start:block_end], float(t_hist[0]), float(t_hist[-1]))
        baseline[block_start:block_end] = A_pred @ last_coeffs

    # Strictly causal fallback: each sample only sees samples before it.
    fallback_window = max(1, min(window, max(1, n)))
    trailing = (
        pd.Series(values, copy=False)
        .shift(1)
        .rolling(fallback_window, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    missing = ~np.isfinite(baseline)
    baseline[missing] = trailing[missing]
    residual = values - baseline
    residual[~np.isfinite(values)] = np.nan
    return baseline, residual
