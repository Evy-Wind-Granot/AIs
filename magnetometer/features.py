"""Causal feature engineering for short-horizon geomagnetic forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the causal forecasting feature set."""

    cadence_s: float = 60.0
    lookback_hours: float = 12.0
    windows_min: tuple[int, ...] = (15, 30, 60, 180, 360, 720)
    kp_release_lag_min: int = 180
    dst_release_lag_min: int = 60

    def __post_init__(self) -> None:
        if self.cadence_s <= 0:
            raise ValueError("cadence_s must be positive")
        if self.lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        if not self.windows_min or any(w <= 0 for w in self.windows_min):
            raise ValueError("windows_min must contain positive values")
        if max(self.windows_min) > self.lookback_hours * 60:
            raise ValueError("largest feature window cannot exceed lookback_hours")
        if self.kp_release_lag_min < 0 or self.dst_release_lag_min < 0:
            raise ValueError("index release lags cannot be negative")


def _rolling_energy(series: pd.Series, window: int) -> pd.Series:
    """Mean-square residual energy over a causal window."""
    return series.pow(2).rolling(window, min_periods=window).mean()


def _rate_of_change(series: pd.Series, cadence_s: float) -> pd.Series:
    """First derivative in nT/s using the immediately preceding sample."""
    return series.diff() / cadence_s


def _normalise_index(
    values: pd.Series | np.ndarray | None,
    target_index: pd.DatetimeIndex,
    name: str,
    release_lag_min: int,
) -> pd.Series:
    """Align an external index without exposing values before their release time."""
    if values is None:
        return pd.Series(np.nan, index=target_index, dtype=float)

    if isinstance(values, pd.Series):
        aligned = values.astype(float).copy()
        aligned.index = pd.DatetimeIndex(pd.to_datetime(aligned.index, utc=True))
        if not aligned.index.is_monotonic_increasing:
            aligned = aligned.sort_index()
        aligned.index = aligned.index + pd.Timedelta(minutes=release_lag_min)
        return aligned.reindex(target_index, method="ffill")

    arr = np.asarray(values, dtype=float)
    if len(arr) != len(target_index):
        raise ValueError(f"{name} length must match residual length")
    return pd.Series(arr, index=target_index, dtype=float)


def build_features(
    residual: pd.Series | np.ndarray,
    kp: pd.Series | np.ndarray | None = None,
    dst: pd.Series | np.ndarray | None = None,
    *,
    index: pd.DatetimeIndex | None = None,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """Build a causal feature matrix from residual and optional global indices.

    All rolling statistics use only samples at or before the feature timestamp.
    Series-valued Kp/Dst inputs are delayed by conservative release lags: three
    hours for Kp and one hour for Dst. This prevents finalized global indices
    from leaking into an earlier prediction timestamp.
    """
    if isinstance(residual, pd.Series):
        series = residual.astype(float).copy()
        if index is not None:
            if len(index) != len(series):
                raise ValueError("index length must match residual length")
            series.index = index
    else:
        values = np.asarray(residual, dtype=float)
        if index is None:
            index = pd.date_range(
                "1970-01-01",
                periods=len(values),
                freq=pd.Timedelta(seconds=config.cadence_s),
                tz="UTC",
            )
        if len(index) != len(values):
            raise ValueError("index length must match residual length")
        series = pd.Series(values, index=index)

    idx = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("feature timestamps must be unique and strictly increasing")
    series.index = idx

    frame = pd.DataFrame(index=idx)
    frame["residual"] = series
    frame["residual_abs"] = series.abs()
    frame["residual_dbdt"] = _rate_of_change(series, config.cadence_s)
    frame["residual_dbdt_abs"] = frame["residual_dbdt"].abs()

    for minutes in config.windows_min:
        samples = max(1, int(round(minutes * 60.0 / config.cadence_s)))
        roll = series.rolling(samples, min_periods=samples)
        prefix = f"{minutes}m"
        frame[f"residual_std_{prefix}"] = roll.std(ddof=0)
        frame[f"residual_ptp_{prefix}"] = roll.max() - roll.min()
        frame[f"residual_energy_{prefix}"] = _rolling_energy(series, samples)
        frame[f"residual_mean_{prefix}"] = roll.mean()
        frame[f"residual_abs_mean_{prefix}"] = series.abs().rolling(
            samples, min_periods=samples
        ).mean()
        frame[f"dbdt_abs_mean_{prefix}"] = frame["residual_dbdt_abs"].rolling(
            samples, min_periods=samples
        ).mean()

    max_lag = int(round(config.lookback_hours * 3600.0 / config.cadence_s))
    for minutes in (15, 30, 60, 180, 360, 720):
        lag = max(1, int(round(minutes * 60.0 / config.cadence_s)))
        if lag <= max_lag:
            frame[f"residual_lag_{minutes}m"] = series.shift(lag)

    frame["kp"] = _normalise_index(kp, idx, "kp", config.kp_release_lag_min)
    frame["dst"] = _normalise_index(dst, idx, "dst", config.dst_release_lag_min)

    for minutes in (180, 360, 720):
        samples = max(1, int(round(minutes * 60.0 / config.cadence_s)))
        frame[f"kp_mean_{minutes}m"] = frame["kp"].rolling(
            samples, min_periods=1
        ).mean()
        frame[f"kp_max_{minutes}m"] = frame["kp"].rolling(
            samples, min_periods=1
        ).max()
        frame[f"dst_mean_{minutes}m"] = frame["dst"].rolling(
            samples, min_periods=1
        ).mean()
        frame[f"dst_min_{minutes}m"] = frame["dst"].rolling(
            samples, min_periods=1
        ).min()

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def build_targets(
    residual: pd.Series | np.ndarray,
    *,
    cadence_s: float = 60.0,
    horizons_hours: Sequence[int] = (1, 3, 6),
    amplitude_window_min: int = 180,
) -> dict[int, pd.Series]:
    """Create strictly-future disturbance-amplitude targets.

    For timestamp ``t`` and horizon ``h``, the target is the peak-to-peak
    residual amplitude in ``[t+h, t+h+window)``. No target sample overlaps the
    feature timestamp, preventing look-ahead leakage at short horizons.
    """
    if isinstance(residual, pd.Series):
        series = residual.astype(float).copy()
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    else:
        values = np.asarray(residual, dtype=float)
        index = pd.date_range(
            "1970-01-01",
            periods=len(values),
            freq=pd.Timedelta(seconds=cadence_s),
            tz="UTC",
        )
        series = pd.Series(values, index=index)

    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("residual timestamps must be unique and increasing")
    if amplitude_window_min <= 0:
        raise ValueError("amplitude_window_min must be positive")

    window = max(1, int(round(amplitude_window_min * 60.0 / cadence_s)))
    rolling_amplitude = (
        series.rolling(window, min_periods=window).max()
        - series.rolling(window, min_periods=window).min()
    )

    targets: dict[int, pd.Series] = {}
    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("forecast horizons must be positive")
        horizon_samples = max(1, int(round(horizon * 3600.0 / cadence_s)))
        # rolling_amplitude[j] covers [j-window+1, j]. We need the first
        # target sample at t+h, hence the endpoint is h+window-1.
        shift = horizon_samples + window - 1
        targets[int(horizon)] = rolling_amplitude.shift(-shift)
    return targets


__all__ = ["FeatureConfig", "build_features", "build_targets"]
