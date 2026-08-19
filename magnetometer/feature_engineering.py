#!/usr/bin/env python3
"""Causal feature engineering for production geomagnetic forecasting.

All features at timestamp t use observations at or before t.  Feature blocks
are assembled off-frame and concatenated once so large forecasting jobs do not
create highly-fragmented pandas DataFrames.
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
    """Causal least-squares slope in nT/hour over each trailing window.

    This is implemented with a single convolution plus a rolling finite-count
    check instead of a Python loop over every timestamp.
    """
    values = x.to_numpy(dtype=float)
    n = len(values)
    w = max(2, int(window))
    result = np.full(n, np.nan, dtype=float)
    if n < w:
        return pd.Series(result, index=x.index)

    t = np.arange(w, dtype=float) * float(cadence_s) / 3600.0
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom <= 0.0:
        return pd.Series(result, index=x.index)

    safe = np.nan_to_num(values, nan=0.0)
    numerators = np.convolve(safe, tc[::-1], mode="valid")
    finite_count = pd.Series(np.isfinite(values), index=x.index).rolling(w, min_periods=w).sum().to_numpy()
    valid = finite_count[w - 1:] == float(w)
    result[w - 1:] = np.where(valid, numerators / denom, np.nan)
    return pd.Series(result, index=x.index)


def _append_block(blocks: list[pd.DataFrame], data: dict[str, pd.Series | np.ndarray], index: pd.Index) -> None:
    """Append a feature block as one DataFrame to avoid repeated frame inserts."""
    blocks.append(pd.DataFrame(data, index=index))


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
    index = frame.index
    residual = _safe_series(frame, "residual")
    residual_abs = residual.abs()
    dbdt = residual.diff() / cadence * 60.0
    dbdt_abs = dbdt.abs()
    dbdt_accel = dbdt.diff() / cadence * 60.0
    dbdt_accel_abs = dbdt_accel.abs()

    kp = _safe_series(frame, "kp")
    dst = _safe_series(frame, "dst")
    kp_available = kp.notna().astype(float)
    dst_available = dst.notna().astype(float)
    kp_filled = kp.ffill()
    dst_filled = dst.ffill()

    blocks: list[pd.DataFrame] = []
    _append_block(
        blocks,
        {
            "residual": residual,
            "residual_abs": residual_abs,
            "dbdt": dbdt,
            "dbdt_abs": dbdt_abs,
            "dbdt_accel": dbdt_accel,
            "dbdt_accel_abs": dbdt_accel_abs,
            "kp": kp_filled,
            "dst": dst_filled,
            "kp_available": kp_available,
            "dst_available": dst_available,
        },
        index,
    )

    # Geomagnetic context and trends. These are all trailing/causal.
    context: dict[str, pd.Series] = {}
    for minutes in (15, 60, 180, 360, 720):
        w = _window_samples(minutes, cadence)
        min_periods = max(2, w // 3)
        label = f"{minutes}m"
        kp_roll = kp_filled.rolling(w, min_periods=min_periods)
        dst_roll = dst_filled.rolling(w, min_periods=min_periods)
        context[f"kp_max_{label}"] = kp_roll.max()
        context[f"kp_mean_{label}"] = kp_roll.mean()
        context[f"dst_min_{label}"] = dst_roll.min()
        context[f"dst_mean_{label}"] = dst_roll.mean()
        context[f"dst_range_{label}"] = dst_roll.max() - dst_roll.min()
        if minutes in (15, 60, 180):
            context[f"kp_change_{label}"] = kp_filled - kp_filled.shift(w)
            context[f"dst_change_{label}"] = dst_filled - dst_filled.shift(w)
    _append_block(blocks, context, index)

    # Residual, derivative, excursion, and slope blocks.
    for minutes in sorted(set(int(v) for v in windows_minutes)):
        if minutes <= 0:
            raise ValueError("window sizes must be positive")
        w = _window_samples(minutes, cadence)
        label = f"{minutes}m"
        min_periods = max(2, w // 3)
        roll = residual.rolling(w, min_periods=min_periods)
        abs_roll = residual_abs.rolling(w, min_periods=min_periods)
        energy = _rolling_energy(residual, w)
        block: dict[str, pd.Series | np.ndarray] = {
            f"residual_mean_{label}": roll.mean(),
            f"residual_std_{label}": roll.std(),
            f"residual_ptp_{label}": roll.max() - roll.min(),
            f"residual_energy_{label}": energy,
            f"residual_rms_{label}": np.sqrt(energy),
            f"abs_residual_mean_{label}": abs_roll.mean(),
            f"abs_residual_max_{label}": abs_roll.max(),
            f"abs_residual_p90_{label}": abs_roll.quantile(0.90),
            f"abs_residual_p95_{label}": abs_roll.quantile(0.95),
            f"dbdt_std_{label}": dbdt.rolling(w, min_periods=min_periods).std(),
            f"dbdt_max_{label}": dbdt_abs.rolling(w, min_periods=min_periods).max(),
            f"dbdt_accel_max_{label}": dbdt_accel_abs.rolling(w, min_periods=min_periods).max(),
        }
        if minutes <= 180:
            block[f"residual_slope_{label}"] = _rolling_slope(residual, w, cadence)
            block[f"abs_slope_{label}"] = _rolling_slope(residual_abs, w, cadence)

        abs_array = residual_abs.to_numpy(dtype=float)
        for threshold in EXCURSION_THRESHOLDS_NT:
            tag = str(int(threshold))
            mask = pd.Series(abs_array >= threshold, index=index)
            rolling_mask = mask.astype(float).rolling(w, min_periods=min_periods)
            block[f"above_{tag}nt_fraction_{label}"] = rolling_mask.mean()
            block[f"above_{tag}nt_count_{label}"] = rolling_mask.sum()
        _append_block(blocks, block, index)

    # Event-state features: persistence and recency.
    event_block: dict[str, pd.Series | np.ndarray] = {}
    for threshold in EXCURSION_THRESHOLDS_NT:
        tag = str(int(threshold))
        mask = residual_abs >= threshold
        groups = (~mask).cumsum()
        run_length = mask.groupby(groups, sort=False).cumcount() + 1
        event_block[f"consecutive_above_{tag}nt_min"] = run_length.where(mask, 0) * cadence / 60.0
        last_event = pd.Series(index.where(mask), index=index).ffill()
        event_block[f"minutes_since_{tag}nt"] = (
            (index.to_series() - last_event).dt.total_seconds() / 60.0
        ).fillna(1e6)
    _append_block(blocks, event_block, index)

    lag_block: dict[str, pd.Series] = {}
    for minutes in sorted(set(int(v) for v in lags_minutes)):
        if minutes <= 0:
            raise ValueError("lag sizes must be positive")
        lag = max(1, int(round(minutes * 60.0 / cadence)))
        lag_block[f"residual_lag_{minutes}m"] = residual.shift(lag)
        lag_block[f"abs_residual_lag_{minutes}m"] = residual_abs.shift(lag)
        lag_block[f"dbdt_lag_{minutes}m"] = dbdt.shift(lag)
    _append_block(blocks, lag_block, index)

    threshold_block: dict[str, pd.Series] = {}
    for threshold in (15.0, 25.0, 35.0, 50.0, 75.0):
        tag = str(int(threshold))
        threshold_block[f"distance_to_{tag}nt"] = threshold - residual_abs
        threshold_block[f"overshoot_{tag}nt"] = np.maximum(residual_abs - threshold, 0.0)
    _append_block(blocks, threshold_block, index)

    utc_hours = (
        index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
        + index.second.to_numpy(dtype=float) / 3600.0
    )
    _append_block(
        blocks,
        {
            "utc_sin_24h": np.sin(2.0 * np.pi * utc_hours / 24.0),
            "utc_cos_24h": np.cos(2.0 * np.pi * utc_hours / 24.0),
            "utc_sin_12h": np.sin(2.0 * np.pi * utc_hours / 12.0),
            "utc_cos_12h": np.cos(2.0 * np.pi * utc_hours / 12.0),
        },
        index,
    )

    out = pd.concat(blocks, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    # Warm-up values are filled only from the past. No bfill/interpolation.
    return out.ffill().fillna(0.0)


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
    values = features.loc[:, list(feature_names)].to_numpy(dtype=np.float32, copy=False)
    if len(values) < sequence_length:
        return np.empty((0, sequence_length, len(feature_names)), dtype=np.float32), pd.DatetimeIndex([], tz=features.index.tz)
    windows = np.lib.stride_tricks.sliding_window_view(values, (sequence_length, values.shape[1]))
    windows = windows[:, 0]
    finite = np.isfinite(windows).all(axis=(1, 2))
    tensors = np.ascontiguousarray(windows[finite], dtype=np.float32)
    timestamps = features.index[sequence_length - 1:][finite]
    return tensors, pd.DatetimeIndex(timestamps)
