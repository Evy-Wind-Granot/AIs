#!/usr/bin/env python3
"""Deterministic, strictly-causal production detector.

The detector is shared by live/demo and certification paths.  It uses only the
current sample and trailing historical data.  No centered windows, future data,
or retroactive gap filling are permitted.

Design goals:
* conservative against isolated telemetry spikes;
* sensitive to sustained and genuinely strong geomagnetic excursions;
* explicit causal warm-up;
* state hysteresis so one physical disturbance is not fragmented into many
  events by short threshold dips;
* nested severity levels and safe handling of missing samples.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_UNSETTLED_NT = 10.0
DEFAULT_MAJOR_STORM_NT = 100.0
DEFAULT_SEVERE_STORM_NT = 200.0
DEFAULT_ANOMALY_DELTA_NT = 100.0


def _window(seconds: float, cadence_s: float, cap: int = 0) -> int:
    n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
    return min(n, cap) if cap else n


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window, min_periods=window)
        .median()
        .to_numpy(dtype=float)
    )


def _rolling_quantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window, min_periods=window)
        .quantile(quantile)
        .to_numpy(dtype=float)
    )


def _hysteresis_mask(
    evidence_on: np.ndarray,
    evidence_off: np.ndarray,
    min_on: int,
    min_off: int,
) -> np.ndarray:
    """Apply online confirmation/release hysteresis without look-ahead."""
    on = np.asarray(evidence_on, dtype=bool)
    off = np.asarray(evidence_off, dtype=bool)
    if on.shape != off.shape:
        raise ValueError("evidence_on and evidence_off must have identical shapes")

    n = len(on)
    out = np.zeros(n, dtype=bool)
    state = False
    candidate = 0
    min_on = max(1, int(min_on))
    min_off = max(1, int(min_off))

    for i in range(n):
        if not state:
            if on[i]:
                candidate += 1
                if candidate >= min_on:
                    state = True
                    candidate = 0
            else:
                candidate = 0
        else:
            if off[i]:
                candidate += 1
                if candidate >= min_off:
                    state = False
                    candidate = 0
            else:
                candidate = 0
        out[i] = state
    return out


def _causal_anomaly_mask(
    x: np.ndarray, cadence_s: float
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Detect abrupt telemetry steps using historical robust statistics only."""
    diff = np.diff(x, prepend=np.nan)
    w = _window(3 * 3600, cadence_s, 721)
    d = pd.Series(diff, dtype=float)
    med = d.rolling(w, min_periods=w).median()
    mad = (d - med).abs().rolling(w, min_periods=w).median()
    threshold = np.maximum(
        DEFAULT_ANOMALY_DELTA_NT,
        8.0 * 1.4826 * mad.to_numpy(dtype=float),
    )
    threshold[~np.isfinite(threshold)] = DEFAULT_ANOMALY_DELTA_NT
    med_values = med.to_numpy(dtype=float)
    anomaly = (
        np.isfinite(diff)
        & np.isfinite(med_values)
        & (np.abs(diff - med_values) >= threshold)
    )
    finite_threshold = threshold[np.isfinite(threshold)]
    median_threshold = (
        float(np.median(finite_threshold))
        if finite_threshold.size
        else DEFAULT_ANOMALY_DELTA_NT
    )
    return anomaly, median_threshold, threshold


def detect_activity_masks(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: float = 15.0,
    storm_threshold: float = 35.0,
    unsettled_threshold: float = DEFAULT_UNSETTLED_NT,
    major_threshold: float = DEFAULT_MAJOR_STORM_NT,
    severe_threshold: float = DEFAULT_SEVERE_STORM_NT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return strictly-causal activity/severity masks and diagnostics.

    Storm onset uses either sustained storm-level energy or a strong short
    excursion with independent medium-timescale corroboration.  Storm release
    is intentionally slower than onset: brief dips do not fragment a physical
    disturbance into multiple events.
    """
    x = np.asarray(residual, dtype=float)
    if x.ndim != 1:
        raise ValueError("residual must be a one-dimensional array")
    if cadence_s <= 0 or not np.isfinite(cadence_s):
        raise ValueError("cadence_s must be a positive finite number")
    if active_threshold <= 0 or storm_threshold <= 0:
        raise ValueError("activity thresholds must be positive")
    if not (active_threshold < storm_threshold <= major_threshold <= severe_threshold):
        raise ValueError("thresholds must satisfy active < storm <= major <= severe")

    magnitude = np.abs(x)
    valid = np.isfinite(magnitude)
    safe = np.where(valid, magnitude, np.nan)

    fast = _rolling_median(safe, _window(5 * 60, cadence_s, 31))
    medium = _rolling_median(safe, _window(15 * 60, cadence_s, 61))
    upper_30m = _rolling_quantile(safe, _window(30 * 60, cadence_s, 121), 0.75)
    slow = _rolling_median(safe, _window(60 * 60, cadence_s, 181))
    slow_3h = _rolling_median(safe, _window(3 * 3600, cadence_s, 361))

    arrays = (fast, medium, upper_30m, slow, slow_3h)
    history_ready = (
        np.isfinite(fast)
        & np.isfinite(medium)
        & np.isfinite(upper_30m)
        & np.isfinite(slow)
        & np.isfinite(slow_3h)
        & valid
    )
    for arr in arrays:
        arr[~np.isfinite(arr)] = 0.0

    active_evidence = history_ready & (
        (medium >= active_threshold)
        | ((slow >= 0.70 * active_threshold) & (medium >= 0.55 * active_threshold))
        | ((slow_3h >= 0.55 * active_threshold) & (medium >= 0.50 * active_threshold))
        | ((upper_30m >= 1.10 * active_threshold) & (medium >= 0.50 * active_threshold))
        | ((fast >= 1.35 * active_threshold) & (medium >= 0.50 * active_threshold))
    )

    # A storm can be established in two ways:
    #   1) sustained storm-level 15-minute energy;
    #   2) a genuinely strong short/30-minute excursion corroborated by
    #      medium-term activity.  This restores sensitivity to storm onsets
    #      that are sharp rather than slowly accumulating.
    strong_short = (
        (fast >= 1.55 * storm_threshold)
        & (medium >= 0.55 * storm_threshold)
    )
    strong_30m = (
        (upper_30m >= storm_threshold)
        & (medium >= 0.65 * storm_threshold)
    )
    sustained = (
        (medium >= storm_threshold)
        | ((slow >= storm_threshold) & (medium >= 0.70 * storm_threshold))
        | ((slow_3h >= storm_threshold) & (medium >= 0.70 * storm_threshold))
    )
    storm_evidence = history_ready & (sustained | strong_short | strong_30m)

    active = _hysteresis_mask(
        active_evidence,
        history_ready & (medium <= 0.70 * active_threshold),
        max(1, int(round(5 * 60 / cadence_s))),
        max(1, int(round(30 * 60 / cadence_s))),
    )

    # Ten minutes confirms onset.  Sixty minutes of recovery evidence is
    # required to end a storm, preventing short dips from creating separate
    # events and preserving recall across realistic storm structure.
    storm = _hysteresis_mask(
        storm_evidence,
        history_ready & (medium <= 0.65 * storm_threshold),
        max(1, int(round(10 * 60 / cadence_s))),
        max(1, int(round(60 * 60 / cadence_s))),
    )

    major_evidence = history_ready & (
        (medium >= major_threshold)
        | ((upper_30m >= 0.90 * major_threshold) & (medium >= 0.80 * major_threshold))
        | ((fast >= 1.15 * major_threshold) & (medium >= 0.80 * major_threshold))
    )
    severe_evidence = history_ready & (
        (medium >= severe_threshold)
        | ((upper_30m >= 0.90 * severe_threshold) & (medium >= 0.80 * severe_threshold))
        | ((fast >= 1.10 * severe_threshold) & (medium >= 0.80 * severe_threshold))
    )

    major = _hysteresis_mask(
        major_evidence,
        history_ready & (medium <= 0.75 * major_threshold),
        max(1, int(round(10 * 60 / cadence_s))),
        max(1, int(round(30 * 60 / cadence_s))),
    ) & storm

    severe = _hysteresis_mask(
        severe_evidence,
        history_ready & (medium <= 0.75 * severe_threshold),
        max(1, int(round(10 * 60 / cadence_s))),
        max(1, int(round(30 * 60 / cadence_s))),
    ) & major

    active &= valid & history_ready
    storm &= valid & history_ready
    major &= valid & history_ready
    severe &= valid & history_ready

    anomaly, anomaly_median_threshold, anomaly_threshold = _causal_anomaly_mask(x, cadence_s)

    diagnostics = {
        "fast_5m_nt": fast,
        "medium_15m_nt": medium,
        "upper_30m_p75_nt": upper_30m,
        "slow_60m_nt": slow,
        "slow_3h_nt": slow_3h,
        "storm_evidence": storm_evidence,
        "storm_strong_short_evidence": history_ready & strong_short,
        "storm_strong_30m_evidence": history_ready & strong_30m,
        "storm_sustained_evidence": history_ready & sustained,
        "history_ready": history_ready,
        "unsettled_threshold_nt": np.full(len(x), unsettled_threshold, dtype=float),
        "anomaly_threshold_nt": anomaly_threshold,
        "anomaly_median_threshold_nt": np.full(len(x), anomaly_median_threshold, dtype=float),
        "anomaly": anomaly & valid,
    }
    return active, storm, major, severe, diagnostics


def flag_activity(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: float = 15.0,
    storm_threshold: float = 35.0,
    unsettled_threshold: float = DEFAULT_UNSETTLED_NT,
    major_threshold: float = DEFAULT_MAJOR_STORM_NT,
    severe_threshold: float = DEFAULT_SEVERE_STORM_NT,
) -> np.ndarray:
    """Classify residuals with causal multi-timescale hysteresis."""
    x = np.asarray(residual, dtype=float)
    active, storm, major, severe, diagnostics = detect_activity_masks(
        x,
        cadence_s=cadence_s,
        active_threshold=active_threshold,
        storm_threshold=storm_threshold,
        unsettled_threshold=unsettled_threshold,
        major_threshold=major_threshold,
        severe_threshold=severe_threshold,
    )

    medium = diagnostics["medium_15m_nt"]
    ready = diagnostics["history_ready"]
    flags = np.full(len(x), "quiet", dtype=object)
    flags[ready & (medium >= unsettled_threshold)] = "unsettled"
    flags[active] = "active"
    flags[storm] = "minor_storm"
    flags[major] = "major_storm"
    flags[severe] = "severe_storm"

    anomaly = diagnostics["anomaly"]
    flags[anomaly & ready & ~active & ~storm] = "anomaly"
    flags[~np.isfinite(x)] = "quiet"
    return flags
