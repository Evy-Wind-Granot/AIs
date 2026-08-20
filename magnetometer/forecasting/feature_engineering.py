#!/usr/bin/env python3
"""Causal feature engineering for geomagnetic short-horizon forecasting.

The deterministic QDC/Harmonic residual remains the core signal. This module
turns that residual plus aligned Kp/Dst context into causal rolling features
and sequence tensors suitable for tree models or future sequence architectures.

No feature at timestamp ``t`` reads observations after ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_WINDOWS_MINUTES: Tuple[int, ...] = (15, 60, 180, 360)
DEFAULT_LAGS_MINUTES: Tuple[int, ...] = (1, 5, 15, 30, 60, 180)
DEFAULT_SEQUENCE_LENGTH = 60

BASE_FEATURES = (
    "residual",
    "residual_abs",
    "dbdt",
    "dbdt_abs",
    "kp",
    "dst",
    "kp_available",
    "dst_available",
)


def _window_samples(minutes: int, cadence_s: float) -> int:
    return max(2, int(round(minutes * 60.0 / max(float(cadence_s), 1.0))))


def _safe_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _rolling_energy(x: pd.Series, window: int) -> pd.Series:
    return x.pow(2).rolling(window, min_periods=max(2, window // 3)).mean()


def make_forecast_features(
    frame: pd.DataFrame,
    *,
    cadence_s: float = 60.0,
    windows_minutes: Sequence[int] = DEFAULT_WINDOWS_MINUTES,
    lags_minutes: Sequence[int] = DEFAULT_LAGS_MINUTES,
) -> pd.DataFrame:
    """Build a causal feature frame from residual + aligned index columns.

    Expected input columns are ``residual`` and optionally ``kp`` / ``dst``.
    The index must be a monotonically increasing UTC DatetimeIndex.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a pandas DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("frame index must be timezone-aware")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted ascending")
    if "residual" not in frame.columns:
        raise ValueError("frame must contain a 'residual' column")

    residual = _safe_series(frame, "residual")
    cadence = max(float(cadence_s), 1.0)
    out = pd.DataFrame(index=frame.index)
    out["residual"] = residual
    out["residual_abs"] = residual.abs()
    out["dbdt"] = residual.diff() / cadence * 60.0
    out["dbdt_abs"] = out["dbdt"].abs()

    kp = _safe_series(frame, "kp")
    dst = _safe_series(frame, "dst")
    out["kp"] = kp
    out["dst"] = dst
    out["kp_available"] = kp.notna().astype(float)
    out["dst_available"] = dst.notna().astype(float)

    # Forward-fill external indices only. Never interpolate them backwards or
    # manufacture values before the first known reference observation.
    out["kp"] = out["kp"].ffill()
    out["dst"] = out["dst"].ffill()

    for minutes in sorted(set(int(v) for v in windows_minutes)):
        if minutes <= 0:
            raise ValueError("window sizes must be positive")
        w = _window_samples(minutes, cadence)
        label = f"{minutes}m"
        out[f"residual_std_{label}"] = residual.rolling(
            w, min_periods=max(2, w // 3)
        ).std()
        out[f"residual_ptp_{label}"] = residual.rolling(
            w, min_periods=max(2, w // 3)
        ).max() - residual.rolling(w, min_periods=max(2, w // 3)).min()
        out[f"residual_energy_{label}"] = _rolling_energy(residual, w)
        out[f"abs_residual_mean_{label}"] = residual.abs().rolling(
            w, min_periods=max(2, w // 3)
        ).mean()
        out[f"abs_residual_max_{label}"] = residual.abs().rolling(
            w, min_periods=max(2, w // 3)
        ).max()
        out[f"dbdt_std_{label}"] = out["dbdt"].rolling(
            w, min_periods=max(2, w // 3)
        ).std()
        out[f"dbdt_max_{label}"] = out["dbdt_abs"].rolling(
            w, min_periods=max(2, w // 3)
        ).max()

    for minutes in sorted(set(int(v) for v in lags_minutes)):
        if minutes <= 0:
            raise ValueError("lag sizes must be positive")
        lag = max(1, int(round(minutes * 60.0 / cadence)))
        out[f"residual_lag_{minutes}m"] = residual.shift(lag)
        out[f"dbdt_lag_{minutes}m"] = out["dbdt"].shift(lag)

    # Local solar-time harmonics help capture residual diurnal structure while
    # preserving the physical QDC layer rather than asking the ML model to
    # rediscover all deterministic periodic structure from scratch.
    utc_hours = (
        frame.index.hour.to_numpy(dtype=float)
        + frame.index.minute.to_numpy(dtype=float) / 60.0
        + frame.index.second.to_numpy(dtype=float) / 3600.0
    )
    out["utc_sin_24h"] = np.sin(2.0 * np.pi * utc_hours / 24.0)
    out["utc_cos_24h"] = np.cos(2.0 * np.pi * utc_hours / 24.0)
    out["utc_sin_12h"] = np.sin(2.0 * np.pi * utc_hours / 12.0)
    out["utc_cos_12h"] = np.cos(2.0 * np.pi * utc_hours / 12.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_future_targets(
    residual: pd.Series,
    *,
    cadence_s: float = 60.0,
    horizons_hours: Sequence[int] = (1, 3, 6),
    storm_threshold_nt: float = 35.0,
) -> pd.DataFrame:
    """Create future peak-amplitude regression and storm-breach targets.

    For each timestamp t and horizon H, the regression target is the maximum
    absolute residual in the *future* interval (t, t+H], while the binary target
    indicates whether that interval breaches the configured storm threshold.
    """
    if not isinstance(residual.index, pd.DatetimeIndex):
        raise TypeError("residual must use a DatetimeIndex")
    x = pd.to_numeric(residual, errors="coerce").abs()
    cadence = max(float(cadence_s), 1.0)
    out = pd.DataFrame(index=residual.index)

    for horizon in sorted(set(int(h) for h in horizons_hours)):
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        steps = max(1, int(round(horizon * 3600.0 / cadence)))
        future = pd.concat(
            [x.shift(-offset) for offset in range(1, steps + 1)], axis=1
        )
        peak = future.max(axis=1, skipna=True)
        # A target is valid only when the complete forecast horizon exists.
        enough_future = x.shift(-steps).notna()
        peak = peak.where(enough_future)
        out[f"target_peak_abs_{horizon}h"] = peak
        out[f"target_storm_{horizon}h"] = (peak >= float(storm_threshold_nt)).astype(float)
        out.loc[peak.isna(), f"target_storm_{horizon}h"] = np.nan

    return out


def build_supervised_dataset(
    frame: pd.DataFrame,
    *,
    cadence_s: float = 60.0,
    windows_minutes: Sequence[int] = DEFAULT_WINDOWS_MINUTES,
    lags_minutes: Sequence[int] = DEFAULT_LAGS_MINUTES,
    horizons_hours: Sequence[int] = (1, 3, 6),
    storm_threshold_nt: float = 35.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return causally engineered features and horizon-aligned targets."""
    features = make_forecast_features(
        frame,
        cadence_s=cadence_s,
        windows_minutes=windows_minutes,
        lags_minutes=lags_minutes,
    )
    targets = make_future_targets(
        _safe_series(frame, "residual"),
        cadence_s=cadence_s,
        horizons_hours=horizons_hours,
        storm_threshold_nt=storm_threshold_nt,
    )
    common = features.index.intersection(targets.index)
    return features.loc[common], targets.loc[common]


def sequence_tensor(
    features: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Convert causal feature rows to [samples, timesteps, features] tensors."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    values = features.loc[:, list(feature_names)].to_numpy(dtype=float)
    tensors = []
    timestamps = []
    for end in range(sequence_length - 1, len(values)):
        window = values[end - sequence_length + 1 : end + 1]
        if not np.isfinite(window).all():
            continue
        tensors.append(window)
        timestamps.append(features.index[end])

    if not tensors:
        return np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), pd.DatetimeIndex([], tz=features.index.tz)
    return np.asarray(tensors, dtype=np.float32), pd.DatetimeIndex(timestamps)
