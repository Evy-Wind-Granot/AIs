#!/usr/bin/env python3
"""Causal feature engineering for production geomagnetic forecasting.

All features at timestamp t use observations at or before t.  The feature set
is deliberately detector-oriented: multi-timescale excursion persistence,
trend/acceleration, threshold occupancy, event recency, and Kp/Dst context are
included so the classifier can distinguish transient noise from a developing
geomagnetic event.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_WINDOWS_MINUTES: Tuple[int, ...] = (5, 15, 30, 60, 180, 360, 720)
DEFAULT_LAGS_MINUTES: Tuple[int, ...] = (1, 5, 15, 30, 60, 180, 360)
DEFAULT_SEQUENCE_LENGTH = 60
EXCURSION_THRESHOLDS_NT = (10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0)

BASE_FEATURES = (
    "residual", "residual_abs", "dbdt", "dbdt_abs", "dbdt_accel",
    "kp", "dst", "kp_available", "dst_available",
)


def _window_samples(minutes: int, cadence_s: float) -> int:
    return max(2, int(round(minutes * 60.0 / max(float(cadence_s), 1.0))))


def _safe_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _rolling_energy(x: pd.Series, window: int) -> pd.Series:
    return x.pow(2).rolling(window, min_periods=max(2, window // 3)).mean()


def _rolling_slope(x: pd.Series, window: int, cadence_s: float) -> pd.Series:
    """Causal least-squares slope in nT/hour over each trailing window."""
    n = len(x)
    values = x.to_numpy(dtype=float)
    result = np.full(n, np.nan, dtype=float)
    w = max(2, int(window))
    t = np.arange(w, dtype=float) * float(cadence_s) / 3600.0
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom <= 0:
        return pd.Series(result, index=x.index)
    for end in range(w - 1, n):
        segment = values[end - w + 1:end + 1]
        if not np.isfinite(segment).all():
            continue
        result[end] = float(np.dot(tc, segment - segment.mean()) / denom)
    return pd.Series(result, index=x.index)


def make_forecast_features(
    frame: pd.DataFrame,
    *,
    cadence_s: float = 60.0,
    windows_minutes: Sequence[int] = DEFAULT_WINDOWS_MINUTES,
    lags_minutes: Sequence[int] = DEFAULT_LAGS_MINUTES,
) -> pd.DataFrame:
    """Build a deterministic causal feature frame from residual + indices."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a pandas DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("frame index must be timezone-aware")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError("frame index must not contain duplicate timestamps")
    if "residual" not in frame.columns:
        raise ValueError("frame must contain a 'residual' column")

    cadence = max(float(cadence_s), 1.0)
    residual = _safe_series(frame, "residual")
    out = pd.DataFrame(index=frame.index)
    out["residual"] = residual
    out["residual_abs"] = residual.abs()
    out["dbdt"] = residual.diff() / cadence * 60.0
    out["dbdt_abs"] = out["dbdt"].abs()
    out["dbdt_accel"] = out["dbdt"].diff() / cadence * 60.0
    out["dbdt_accel_abs"] = out["dbdt_accel"].abs()

    kp = _safe_series(frame, "kp")
    dst = _safe_series(frame, "dst")
    out["kp_available"] = kp.notna().astype(float)
    out["dst_available"] = dst.notna().astype(float)
    out["kp"] = kp.ffill()
    out["dst"] = dst.ffill()

    # Geomagnetic context and trends. These are all trailing/causal.
    for minutes in (15, 60, 180, 360, 720):
        w = _window_samples(minutes, cadence)
        label = f"{minutes}m"
        out[f"kp_max_{label}"] = out["kp"].rolling(w, min_periods=max(2, w // 3)).max()
        out[f"kp_mean_{label}"] = out["kp"].rolling(w, min_periods=max(2, w // 3)).mean()
        out[f"dst_min_{label}"] = out["dst"].rolling(w, min_periods=max(2, w // 3)).min()
        out[f"dst_mean_{label}"] = out["dst"].rolling(w, min_periods=max(2, w // 3)).mean()
        out[f"dst_range_{label}"] = (
            out["dst"].rolling(w, min_periods=max(2, w // 3)).max()
            - out["dst"].rolling(w, min_periods=max(2, w // 3)).min()
        )
        if minutes in (15, 60, 180):
            out[f"kp_change_{label}"] = out["kp"] - out["kp"].shift(w)
            out[f"dst_change_{label}"] = out["dst"] - out["dst"].shift(w)

    for minutes in sorted(set(int(v) for v in windows_minutes)):
        if minutes <= 0:
            raise ValueError("window sizes must be positive")
        w = _window_samples(minutes, cadence)
        label = f"{minutes}m"
        min_periods = max(2, w // 3)
        roll = residual.rolling(w, min_periods=min_periods)
        abs_roll = residual.abs().rolling(w, min_periods=min_periods)
        out[f"residual_mean_{label}"] = roll.mean()
        out[f"residual_std_{label}"] = roll.std()
        out[f"residual_ptp_{label}"] = roll.max() - roll.min()
        out[f"residual_energy_{label}"] = _rolling_energy(residual, w)
        out[f"residual_rms_{label}"] = np.sqrt(_rolling_energy(residual, w))
        out[f"abs_residual_mean_{label}"] = abs_roll.mean()
        out[f"abs_residual_max_{label}"] = abs_roll.max()
        out[f"abs_residual_p90_{label}"] = abs_roll.quantile(0.90)
        out[f"abs_residual_p95_{label}"] = abs_roll.quantile(0.95)
        out[f"dbdt_std_{label}"] = out["dbdt"].rolling(w, min_periods=min_periods).std()
        out[f"dbdt_max_{label}"] = out["dbdt_abs"].rolling(w, min_periods=min_periods).max()
        out[f"dbdt_accel_max_{label}"] = out["dbdt_accel_abs"].rolling(w, min_periods=min_periods).max()
        if minutes <= 180:
            out[f"residual_slope_{label}"] = _rolling_slope(residual, w, cadence)
            out[f"abs_slope_{label}"] = _rolling_slope(residual.abs(), w, cadence)

        for threshold in EXCURSION_THRESHOLDS_NT:
            tag = str(int(threshold))
            mask = residual.abs() >= threshold
            out[f"above_{tag}nt_fraction_{label}"] = mask.astype(float).rolling(w, min_periods=min_periods).mean()
            out[f"above_{tag}nt_count_{label}"] = mask.astype(float).rolling(w, min_periods=min_periods).sum()

    # Event-state features: persistence and recency are often more predictive
    # than the instantaneous residual alone.
    abs_res = residual.abs()
    for threshold in EXCURSION_THRESHOLDS_NT:
        tag = str(int(threshold))
        mask = abs_res >= threshold
        # Consecutive samples above a threshold, expressed in minutes.
        groups = (~mask).cumsum()
        run_length = mask.groupby(groups, sort=False).cumcount() + 1
        out[f"consecutive_above_{tag}nt_min"] = (run_length.where(mask, 0) * cadence / 60.0)
        last_event = pd.Series(frame.index.where(mask), index=frame.index).ffill()
        out[f"minutes_since_{tag}nt"] = (
            (frame.index.to_series() - last_event).dt.total_seconds() / 60.0
        ).fillna(1e6)

    for minutes in sorted(set(int(v) for v in lags_minutes)):
        if minutes <= 0:
            raise ValueError("lag sizes must be positive")
        lag = max(1, int(round(minutes * 60.0 / cadence)))
        out[f"residual_lag_{minutes}m"] = residual.shift(lag)
        out[f"abs_residual_lag_{minutes}m"] = abs_res.shift(lag)
        out[f"dbdt_lag_{minutes}m"] = out["dbdt"].shift(lag)

    # Local trend ratios / distances from operational thresholds.
    for threshold in (15.0, 25.0, 35.0, 50.0, 75.0):
        tag = str(int(threshold))
        out[f"distance_to_{tag}nt"] = threshold - abs_res
        out[f"overshoot_{tag}nt"] = np.maximum(abs_res - threshold, 0.0)

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
    # Warm-up values are filled only from the past. No bfill/interpolation.
    out = out.ffill().fillna(0.0)
    return out


def _future_window_max(x: pd.Series, steps: int) -> pd.Series:
    shifted = x.shift(-1)
    return shifted.iloc[::-1].rolling(steps, min_periods=steps).max().iloc[::-1]


def make_future_targets(
    residual: pd.Series,
    *,
    cadence_s: float = 60.0,
    horizons_hours: Sequence[int] = (1, 3, 6),
    storm_threshold_nt: float = 35.0,
) -> pd.DataFrame:
    """Create future peak-amplitude and storm-breach targets."""
    if not isinstance(residual.index, pd.DatetimeIndex):
        raise TypeError("residual must use a DatetimeIndex")
    x = pd.to_numeric(residual, errors="coerce").abs()
    cadence = max(float(cadence_s), 1.0)
    out = pd.DataFrame(index=residual.index)
    for horizon in sorted(set(int(h) for h in horizons_hours)):
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        steps = max(1, int(round(horizon * 3600.0 / cadence)))
        peak = _future_window_max(x, steps)
        out[f"target_peak_abs_{horizon}h"] = peak
        storm = (peak >= float(storm_threshold_nt)).astype(float)
        out.loc[peak.isna(), f"target_storm_{horizon}h"] = np.nan
        out.loc[peak.notna(), f"target_storm_{horizon}h"] = storm.loc[peak.notna()]
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
    features = make_forecast_features(
        frame, cadence_s=cadence_s, windows_minutes=windows_minutes, lags_minutes=lags_minutes
    )
    targets = make_future_targets(
        _safe_series(frame, "residual"), cadence_s=cadence_s,
        horizons_hours=horizons_hours, storm_threshold_nt=storm_threshold_nt,
    )
    common = features.index.intersection(targets.index)
    return features.loc[common], targets.loc[common]


def sequence_tensor(
    features: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    values = features.loc[:, list(feature_names)].to_numpy(dtype=float)
    tensors = []
    timestamps = []
    for end in range(sequence_length - 1, len(values)):
        window = values[end - sequence_length + 1:end + 1]
        if not np.isfinite(window).all():
            continue
        tensors.append(window)
        timestamps.append(features.index[end])
    if not tensors:
        return np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), pd.DatetimeIndex([], tz=features.index.tz)
    return np.asarray(tensors, dtype=np.float32), pd.DatetimeIndex(timestamps)
