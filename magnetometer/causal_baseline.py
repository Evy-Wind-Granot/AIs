#!/usr/bin/env python3
"""Strictly causal robust harmonic quiet-day baseline.

The baseline fits 24h, 12h, 8h, and 6h harmonics with a quadratic trend in
overlapping trailing windows. Active intervals are downweighted with a
Huber-like iteration, and storm-heavy windows reuse the previous quiet
quadratic trend so long storms cannot be absorbed into baseline drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import linalg

BASELINE_VERSION = "robust-harmonic-overlap-v2"
HARMONIC_PERIOD_HOURS = (24.0, 12.0, 8.0, 6.0)


@dataclass(frozen=True)
class HarmonicFitConfig:
    fit_window_hours: float = 24.0
    stride_fraction: float = 0.5
    min_history_fraction: float = 0.50
    max_iterations: int = 5
    huber_k: float = 1.5
    hard_outlier_nt: float = 75.0
    storm_lock_threshold_nt: float = 35.0
    storm_lock_fraction: float = 0.05


def build_design_matrix(t_hours: np.ndarray, t_ref_min: float, t_ref_max: float) -> np.ndarray:
    """Build quadratic plus multi-harmonic design matrix."""
    t = np.asarray(t_hours, dtype=float)
    scale = max(float(t_ref_max) - float(t_ref_min), 1e-12)
    tn = np.clip((t - float(t_ref_min)) / scale, -0.5, 1.5)
    columns = [np.ones_like(t), tn, tn**2]
    for period in HARMONIC_PERIOD_HOURS:
        angle = 2.0 * np.pi * t / period
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def _robust_scale(residual: np.ndarray) -> float:
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return 1.0
    mad = float(np.median(np.abs(finite - np.median(finite))))
    return max(1.4826 * mad, 1.0)


def robust_harmonic_fit(
    x: np.ndarray,
    cadence_s: float,
    t_hours: np.ndarray,
    *,
    config: HarmonicFitConfig | None = None,
    locked_quadratic: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a quiet-day harmonic baseline with robust iterative weights."""
    cfg = config or HarmonicFitConfig()
    values = np.asarray(x, dtype=float)
    t = np.asarray(t_hours, dtype=float)
    valid = np.isfinite(values) & np.isfinite(t)
    if valid.sum() < build_design_matrix(np.arange(1, dtype=float), 0.0, 1.0).shape[1] + 2:
        return np.full(3 + 2 * len(HARMONIC_PERIOD_HOURS), np.nan)

    design = build_design_matrix(t[valid], float(np.min(t[valid])), float(np.max(t[valid])))
    y = values[valid]
    weights = np.ones(y.size, dtype=float)
    coeff = np.zeros(design.shape[1], dtype=float)

    for _ in range(max(1, int(cfg.max_iterations))):
        if locked_quadratic is None:
            coeff, *_ = linalg.lstsq(design * weights[:, None], y * weights)
        else:
            locked = np.asarray(locked_quadratic, dtype=float)
            harmonic_design = design[:, 3:]
            target = y - design[:, :3] @ locked
            harmonic_coeff, *_ = linalg.lstsq(harmonic_design * weights[:, None], target * weights)
            coeff = np.r_[locked, harmonic_coeff]

        residual = y - design @ coeff
        sigma = _robust_scale(residual)
        abs_r = np.abs(residual)
        weights = np.ones_like(abs_r)
        huber = cfg.huber_k * sigma
        high = abs_r > huber
        weights[high] = huber / np.maximum(abs_r[high], 1e-12)
        weights[abs_r > cfg.hard_outlier_nt] *= 0.1

    return coeff


def _window_params(n: int, cadence_s: float, cfg: HarmonicFitConfig) -> tuple[int, int]:
    window = max(12, int(round(cfg.fit_window_hours * 3600.0 / cadence_s)))
    stride = max(1, int(round(window * cfg.stride_fraction)))
    return min(window, max(n, 1)), stride


def _fill_causal_fallback(values: np.ndarray, baseline: np.ndarray, window: int) -> np.ndarray:
    prev = pd.Series(values, dtype=float).shift(1)
    trailing = prev.rolling(max(1, min(window, values.size)), min_periods=1).median().to_numpy(dtype=float)
    out = baseline.copy()
    missing = ~np.isfinite(out)
    out[missing] = trailing[missing]
    if out.size and not np.isfinite(out[0]):
        finite = values[np.isfinite(values)]
        out[0] = float(finite[0]) if finite.size else 0.0
    return out


def compute_causal_qdc_baseline(
    x: np.ndarray,
    cadence_s: float,
    *,
    fit_window_hours: float = 24.0,
    update_minutes: float | None = None,
    min_history_fraction: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return strictly-causal quiet-day baseline and residual.

    When update_minutes is omitted, refit every hour rather than projecting a
    24-hour quadratic/harmonic fit across a full future day. This sharply
    reduces causal extrapolation drift while keeping the baseline deterministic.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if cadence_s <= 0 or not np.isfinite(cadence_s):
        raise ValueError("cadence_s must be positive and finite")
    if values.size == 0:
        return values.copy(), values.copy()

    stride_fraction = 1.0 / 24.0 if update_minutes is None else max(
        1.0 / max(1.0, fit_window_hours * 60.0),
        float(update_minutes) / (fit_window_hours * 60.0),
    )
    cfg = HarmonicFitConfig(
        fit_window_hours=fit_window_hours,
        stride_fraction=min(max(stride_fraction, 1.0 / 24.0), 1.0),
        min_history_fraction=min_history_fraction,
    )
    n = values.size
    window, stride = _window_params(n, cadence_s, cfg)
    t_hours = np.arange(n, dtype=float) * float(cadence_s) / 3600.0
    baseline_sum = np.zeros(n, dtype=float)
    weight_sum = np.zeros(n, dtype=float)
    last_quiet_quadratic: np.ndarray | None = None

    for predict_start in range(window, n, stride):
        hist_start = max(0, predict_start - window)
        hist = values[hist_start:predict_start]
        hist_t = t_hours[hist_start:predict_start]
        if hist.size == 0 or float(np.isfinite(hist).mean()) < cfg.min_history_fraction:
            continue

        quick_level = np.abs(hist - np.nanmedian(hist))
        storm_fraction = float(np.nanmean(quick_level >= cfg.storm_lock_threshold_nt)) if np.isfinite(quick_level).any() else 0.0
        locked = last_quiet_quadratic if storm_fraction > cfg.storm_lock_fraction else None
        coeff = robust_harmonic_fit(hist, cadence_s, hist_t, config=cfg, locked_quadratic=locked)
        if not np.all(np.isfinite(coeff)):
            continue
        if storm_fraction <= cfg.storm_lock_fraction:
            last_quiet_quadratic = coeff[:3].copy()

        # Only project the fit over the next update interval. The previous
        # implementation projected a 24-hour fit for 24 hours, allowing the
        # quadratic trend to drift far outside the observed range.
        predict_end = min(n, predict_start + stride)
        target_t = t_hours[predict_start:predict_end]
        design = build_design_matrix(target_t, float(hist_t[0]), float(hist_t[-1]))
        prediction = design @ coeff
        weights = np.hanning(prediction.size * 2 + 1)[1::2]
        if weights.size != prediction.size:
            weights = np.ones(prediction.size, dtype=float)
        baseline_sum[predict_start:predict_end] += prediction * weights
        weight_sum[predict_start:predict_end] += weights

    baseline = np.full(n, np.nan, dtype=float)
    ready = weight_sum > 0
    baseline[ready] = baseline_sum[ready] / weight_sum[ready]
    baseline = _fill_causal_fallback(values, baseline, window)
    residual = values - baseline
    residual[~np.isfinite(values)] = np.nan
    return baseline, residual
