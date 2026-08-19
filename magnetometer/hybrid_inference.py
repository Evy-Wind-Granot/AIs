#!/usr/bin/env python3
"""Hybrid deterministic/ML inference for real-time magnetometer monitoring."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from .models.forecaster import ForecastResult, GeomagneticForecaster


def _normalize_index_series(series: pd.Series | None) -> pd.Series | None:
    """Return a UTC-indexed numeric Series, or None when no usable data exists."""
    if series is None or series.empty:
        return None
    result = pd.to_numeric(series, errors="coerce").dropna().copy()
    if result.empty:
        return None
    result.index = pd.to_datetime(result.index, utc=True)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.astype(float)


def build_aligned_forecast_frame(
    residual: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    kp_series: pd.Series | None = None,
    dst_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the causal ML input frame from deterministic residual output."""
    if len(residual) != len(index):
        raise ValueError("residual and index must have equal length")
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("index must be a timezone-aware pandas DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")

    frame = pd.DataFrame(index=index)
    frame["residual"] = pd.to_numeric(np.asarray(residual, dtype=float), errors="coerce")
    tolerance = pd.Timedelta("3h")
    for column, source in (("kp", kp_series), ("dst", dst_series)):
        normalized = _normalize_index_series(source)
        if normalized is None:
            frame[column] = np.nan
        else:
            frame[column] = normalized.reindex(index, method="ffill", tolerance=tolerance)
    return frame


def hybrid_status_payload(
    frame: pd.DataFrame,
    *,
    deterministic_tier: str,
    forecaster: GeomagneticForecaster,
    cadence_s: float = 60.0,
) -> Dict[str, Any]:
    """Merge current deterministic state and future ML predictions."""
    result: ForecastResult = forecaster.predict(
        frame,
        cadence_s=cadence_s,
        current_rule_tier=deterministic_tier,
    )

    forecast_levels = [v["forecast_tier"] for v in result.horizons.values()]
    highest = "quiet"
    rank = {
        "quiet": 0,
        "unsettled": 1,
        "active": 2,
        "minor_storm": 3,
        "major_storm": 4,
        "severe_storm": 5,
        "unknown": -1,
    }
    if forecast_levels:
        highest = max(forecast_levels, key=lambda value: rank.get(str(value), -1))

    trend = "stable"
    current_rank = rank.get(deterministic_tier, 0)
    highest_rank = rank.get(highest, 0)
    if highest_rank > current_rank:
        trend = "escalating"
    elif highest_rank < current_rank:
        trend = "de-escalating"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realtime": {
            "tier": deterministic_tier,
            "residual_nt": result.current_residual_nt,
        },
        "forecast": result.horizons,
        "hybrid": {
            "forecast_highest_tier": highest,
            "forecast_trend": trend,
            "anomaly_delta": result.anomaly_delta,
            "divergence": result.divergence,
            "model_confidence_available": all(
                value.get("model_confidence") is not None for value in result.horizons.values()
            ),
        },
        "model": {
            "version": result.model_version,
            "horizons_hours": sorted(int(v) for v in result.horizons.keys()),
        },
    }
