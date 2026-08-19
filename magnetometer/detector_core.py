#!/usr/bin/env python3
"""Deterministic production detector shared by demo and certification paths.

The detector is deliberately non-ML.  It operates on the local QDC residual
and combines robust multi-timescale envelopes with explicit stateful
hysteresis.  Calibration supplies the active/storm onset thresholds; the final
held-out test is never used to select them.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_UNSETTLED_NT = 10.0
DEFAULT_MAJOR_STORM_NT = 100.0
DEFAULT_SEVERE_STORM_NT = 200.0
DEFAULT_ANOMALY_DELTA_NT = 100.0


def _odd_window(seconds: float, cadence_s: float, cap: int = 0) -> int:
    n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
    if cap:
        n = min(n, cap)
    if n > 1 and n % 2 == 0:
        n += 1
    return n


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _rolling_quantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window, center=True, min_periods=max(1, window // 3))
        .quantile(quantile)
        .to_numpy(dtype=float)
    )


def _hysteresis_mask(
    evidence_on: np.ndarray,
    evidence_off: np.ndarray,
    min_on: int,
    min_off: int,
) -> np.ndarray:
    """Convert noisy evidence into a deterministic state machine."""
    on = np.asarray(evidence_on, dtype=bool)
    off = np.asarray(evidence_off, dtype=bool)
    n = len(on)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out

    min_on = max(1, int(min_on))
    min_off = max(1, int(min_off))
    state = False
    candidate = 0

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


def detect_activity_masks(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: float = 15.0,
    storm_threshold: float = 35.0,
    unsettled_threshold: float = DEFAULT_UNSETTLED_NT,
    major_threshold: float = DEFAULT_MAJOR_STORM_NT,
    severe_threshold: float = DEFAULT_SEVERE_STORM_NT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return deterministic activity/storm severity masks and diagnostics.

    Evidence is deliberately multi-timescale:

    * 5-minute median catches onset without allowing isolated spikes through;
    * 15-minute median is the primary operational signal;
    * 30-minute 75th percentile catches intermittent but sustained excursions;
    * 60-minute median captures established disturbances;
    * 3-hour median captures long, moderate disturbances that never remain
      above a short-window threshold continuously.

    State transitions use independent release hysteresis.  This keeps a short
    recovery inside one physical disturbance from creating a new event.
    """
    x = np.asarray(residual, dtype=float)
    magnitude = np.abs(x)
    valid = np.isfinite(magnitude)
    safe = np.where(valid, magnitude, np.nan)

    fast = _rolling_median(safe, _odd_window(5 * 60, cadence_s, 31))
    medium = _rolling_median(safe, _odd_window(15 * 60, cadence_s, 61))
    upper_30m = _rolling_quantile(safe, _odd_window(30 * 60, cadence_s, 121), 0.75)
    slow = _rolling_median(safe, _odd_window(60 * 60, cadence_s, 181))
    slow_3h = _rolling_median(safe, _odd_window(3 * 3600, cadence_s, 361))

    for arr in (fast, medium, upper_30m, slow, slow_3h):
        arr[~np.isfinite(arr)] = 0.0

    # Activity: sustained moderate excursions or a strong rapid onset.
    active_evidence = (
        (medium >= active_threshold)
        | ((slow >= 0.70 * active_threshold) & (medium >= 0.55 * active_threshold))
        | ((slow_3h >= 0.55 * active_threshold) & (medium >= 0.50 * active_threshold))
        | ((upper_30m >= 1.10 * active_threshold) & (medium >= 0.50 * active_threshold))
        | (fast >= 1.35 * active_threshold)
    )

    # Storm: retain the strong precision of the 15-minute path, while allowing
    # long-duration moderate disturbances to qualify when multiple independent
    # timescales agree.  No single fast spike is sufficient on its own.
    storm_evidence = (
        (medium >= storm_threshold)
        | ((slow >= 0.65 * storm_threshold) & (medium >= 0.65 * storm_threshold))
        | ((slow_3h >= 0.55 * storm_threshold) & (medium >= 0.60 * storm_threshold))
        | ((upper_30m >= 0.95 * storm_threshold) & (medium >= 0.55 * storm_threshold))
        | ((fast >= 1.30 * storm_threshold) & (medium >= 0.55 * storm_threshold))
    )

    active_release = 0.70 * active_threshold
    storm_release = 0.65 * storm_threshold

    active = _hysteresis_mask(
        active_evidence,
        medium <= active_release,
        max(1, int(round(5 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(30 * 60 / max(cadence_s, 1.0)))),
    )
    storm = _hysteresis_mask(
        storm_evidence,
        medium <= storm_release,
        max(1, int(round(8 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(45 * 60 / max(cadence_s, 1.0)))),
    )

    major_evidence = (
        (medium >= major_threshold)
        | ((upper_30m >= 0.90 * major_threshold) & (medium >= 0.70 * major_threshold))
        | (fast >= 1.15 * major_threshold)
    )
    severe_evidence = (
        (medium >= severe_threshold)
        | ((upper_30m >= 0.90 * severe_threshold) & (medium >= 0.70 * severe_threshold))
        | (fast >= 1.10 * severe_threshold)
    )

    major = _hysteresis_mask(
        major_evidence,
        medium <= 0.75 * major_threshold,
        max(1, int(round(8 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(45 * 60 / max(cadence_s, 1.0)))),
    ) & storm
    severe = _hysteresis_mask(
        severe_evidence,
        medium <= 0.75 * severe_threshold,
        max(1, int(round(8 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(45 * 60 / max(cadence_s, 1.0)))),
    ) & major

    active &= valid
    storm &= valid
    major &= valid
    severe &= valid

    # Robust telemetry-step anomaly.  It remains a separate diagnostic class;
    # anomalies do not automatically become geomagnetic storms.
    diff = np.diff(x, prepend=x[0])
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size:
        med = float(np.median(finite_diff))
        mad = float(np.median(np.abs(finite_diff - med)))
        adaptive = max(DEFAULT_ANOMALY_DELTA_NT, 8.0 * 1.4826 * mad)
    else:
        adaptive = DEFAULT_ANOMALY_DELTA_NT
    anomaly = np.isfinite(diff) & (np.abs(diff) >= adaptive)

    envelopes = {
        "fast_5m_nt": fast,
        "medium_15m_nt": medium,
        "upper_30m_p75_nt": upper_30m,
        "slow_60m_nt": slow,
        "slow_3h_nt": slow_3h,
        "unsettled_threshold_nt": np.full(len(x), unsettled_threshold, dtype=float),
        "anomaly_threshold_nt": np.full(len(x), adaptive, dtype=float),
    }
    return active, storm, major, severe, {**envelopes, "anomaly": anomaly}


def flag_activity(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: float = 15.0,
    storm_threshold: float = 35.0,
    unsettled_threshold: float = DEFAULT_UNSETTLED_NT,
    major_threshold: float = DEFAULT_MAJOR_STORM_NT,
    severe_threshold: float = DEFAULT_SEVERE_STORM_NT,
) -> np.ndarray:
    """Classify residuals using deterministic multi-timescale hysteresis."""
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
    flags = np.full(len(x), "quiet", dtype=object)
    flags[medium >= unsettled_threshold] = "unsettled"
    flags[active] = "active"
    flags[storm] = "minor_storm"
    flags[major] = "major_storm"
    flags[severe] = "severe_storm"

    anomaly = diagnostics["anomaly"]
    flags[anomaly & ~active & ~storm] = "anomaly"
    flags[~np.isfinite(x)] = "quiet"
    return flags
