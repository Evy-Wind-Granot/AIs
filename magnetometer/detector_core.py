#!/usr/bin/env python3
"""Deterministic, strictly-causal production detector.

The detector is shared by the live/demo and certification paths.  It is
intentionally non-ML and uses only the current sample plus historical data.
No centered windows, future samples, or full-window statistics are used.
Calibration supplies thresholds; the held-out final-test period is never used
for detector parameter selection.
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
    """Apply confirmation/release hysteresis without looking ahead."""
    on = np.asarray(evidence_on, dtype=bool)
    off = np.asarray(evidence_off, dtype=bool)
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


def _bridge_short_false_runs(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge short gaps only after an event is established."""
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or out.size < 3:
        return out
    false_idx = np.flatnonzero(~out)
    if false_idx.size == 0:
        return out
    starts = false_idx[np.r_[True, np.diff(false_idx) > 1]]
    ends = false_idx[np.r_[np.diff(false_idx) > 1, True]]
    for s, e in zip(starts, ends):
        if s > 0 and e < len(out) - 1 and (e - s + 1) <= max_gap:
            out[s:e + 1] = True
    return out


def _causal_anomaly_mask(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """Detect abrupt telemetry steps using historical robust statistics only."""
    diff = np.diff(x, prepend=np.nan)
    w = _window(3 * 3600, cadence_s, 721)
    d = pd.Series(diff, dtype=float)
    med = d.rolling(w, min_periods=w).median()
    mad = (d - med).abs().rolling(w, min_periods=w).median()
    threshold = np.maximum(DEFAULT_ANOMALY_DELTA_NT, 8.0 * 1.4826 * mad.to_numpy(dtype=float))
    threshold[~np.isfinite(threshold)] = DEFAULT_ANOMALY_DELTA_NT
    anomaly = np.isfinite(diff) & (np.abs(diff - med.to_numpy(dtype=float)) >= threshold)
    return anomaly, float(np.nanmedian(threshold)), threshold


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

    Every rolling feature is trailing/causal.  The first samples that do not
    have enough history remain unclassified rather than being evaluated with
    partial future-free windows, which makes startup behavior explicit.
    """
    x = np.asarray(residual, dtype=float)
    magnitude = np.abs(x)
    valid = np.isfinite(magnitude)
    safe = np.where(valid, magnitude, np.nan)

    fast = _rolling_median(safe, _window(5 * 60, cadence_s, 31))
    medium = _rolling_median(safe, _window(15 * 60, cadence_s, 61))
    upper_30m = _rolling_quantile(safe, _window(30 * 60, cadence_s, 121), 0.75)
    slow = _rolling_median(safe, _window(60 * 60, cadence_s, 181))
    slow_3h = _rolling_median(safe, _window(3 * 3600, cadence_s, 361))

    arrays = (fast, medium, upper_30m, slow, slow_3h)
    history_ready = np.isfinite(fast) & np.isfinite(medium) & np.isfinite(upper_30m) & np.isfinite(slow) & np.isfinite(slow_3h)
    for arr in arrays:
        arr[~np.isfinite(arr)] = 0.0

    active_evidence = history_ready & (
        (medium >= active_threshold)
        | ((slow >= 0.70 * active_threshold) & (medium >= 0.55 * active_threshold))
        | ((slow_3h >= 0.55 * active_threshold) & (medium >= 0.50 * active_threshold))
        | ((upper_30m >= 1.10 * active_threshold) & (medium >= 0.50 * active_threshold))
        | ((fast >= 1.35 * active_threshold) & (medium >= 0.50 * active_threshold))
    )

    # Storm qualification is intentionally stricter than activity.  A single
    # high sample/window cannot create a storm; amplitude must have corroborating
    # persistence or multi-timescale support.
    storm_evidence = history_ready & (
        (medium >= storm_threshold)
        | ((slow >= 0.70 * storm_threshold) & (medium >= 0.70 * storm_threshold))
        | ((slow_3h >= 0.60 * storm_threshold) & (medium >= 0.65 * storm_threshold))
        | ((upper_30m >= 1.00 * storm_threshold) & (medium >= 0.60 * storm_threshold))
        | ((fast >= 1.35 * storm_threshold) & (medium >= 0.60 * storm_threshold))
    )

    active = _hysteresis_mask(
        active_evidence,
        history_ready & (medium <= 0.70 * active_threshold),
        max(1, int(round(5 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(30 * 60 / max(cadence_s, 1.0)))),
    )
    storm = _hysteresis_mask(
        storm_evidence,
        history_ready & (medium <= 0.65 * storm_threshold),
        max(1, int(round(10 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(45 * 60 / max(cadence_s, 1.0)))),
    )
    storm = _bridge_short_false_runs(storm, max(1, int(round(15 * 60 / max(cadence_s, 1.0)))))

    major_evidence = history_ready & (
        (medium >= major_threshold)
        | ((upper_30m >= 0.90 * major_threshold) & (medium >= 0.70 * major_threshold))
        | ((fast >= 1.15 * major_threshold) & (medium >= 0.70 * major_threshold))
    )
    severe_evidence = history_ready & (
        (medium >= severe_threshold)
        | ((upper_30m >= 0.90 * severe_threshold) & (medium >= 0.70 * severe_threshold))
        | ((fast >= 1.10 * severe_threshold) & (medium >= 0.70 * severe_threshold))
    )
    major = _hysteresis_mask(
        major_evidence,
        history_ready & (medium <= 0.75 * major_threshold),
        max(1, int(round(10 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(45 * 60 / max(cadence_s, 1.0)))),
    ) & storm
    severe = _hysteresis_mask(
        severe_evidence,
        history_ready & (medium <= 0.75 * severe_threshold),
        max(1, int(round(10 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(45 * 60 / max(cadence_s, 1.0)))),
    ) & major

    active &= valid & history_ready
    storm &= valid & history_ready
    major &= valid & history_ready
    severe &= valid & history_ready

    anomaly, anomaly_median_threshold, anomaly_threshold = _causal_anomaly_mask(x, cadence_s)

    envelopes = {
        "fast_5m_nt": fast,
        "medium_15m_nt": medium,
        "upper_30m_p75_nt": upper_30m,
        "slow_60m_nt": slow,
        "slow_3h_nt": slow_3h,
        "history_ready": history_ready,
        "unsettled_threshold_nt": np.full(len(x), unsettled_threshold, dtype=float),
        "anomaly_threshold_nt": anomaly_threshold,
        "anomaly_median_threshold_nt": np.full(len(x), anomaly_median_threshold, dtype=float),
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
