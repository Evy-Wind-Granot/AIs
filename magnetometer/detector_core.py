#!/usr/bin/env python3
"""Deterministic production detector shared by demo and certification paths.

The detector deliberately remains non-ML.  It operates only on the local QDC
residual and uses robust multi-timescale envelopes plus explicit stateful
hysteresis.  Thresholds are supplied by the calibration layer; no final-test
information is used here.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# These are policy defaults only. Certification supplies calibrated active and
# storm thresholds; the tier thresholds remain the production severity policy.
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


def _hysteresis_mask(
    evidence_on: np.ndarray,
    evidence_off: np.ndarray,
    min_on: int,
    min_off: int,
) -> np.ndarray:
    """Convert noisy evidence into a deterministic state machine.

    A state opens only after ``min_on`` consecutive positive samples.  Once
    open it closes only after ``min_off`` consecutive release samples.  This
    is intentionally stateful: a brief dip inside a geomagnetic disturbance
    does not manufacture a second event.
    """
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


def _persistent_state(
    evidence: np.ndarray,
    cadence_s: float,
    on_minutes: float,
    off_minutes: float,
    release_fraction: float,
) -> np.ndarray:
    min_on = max(1, int(round(on_minutes * 60.0 / max(cadence_s, 1.0))))
    min_off = max(1, int(round(off_minutes * 60.0 / max(cadence_s, 1.0))))
    # Release evidence is deliberately lower than onset evidence.
    return _hysteresis_mask(
        evidence,
        evidence <= release_fraction,
        min_on,
        min_off,
    )


def detect_activity_masks(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: float = 15.0,
    storm_threshold: float = 35.0,
    unsettled_threshold: float = DEFAULT_UNSETTLED_NT,
    major_threshold: float = DEFAULT_MAJOR_STORM_NT,
    severe_threshold: float = DEFAULT_SEVERE_STORM_NT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return active/storm/major/severe masks and diagnostic envelopes.

    The evidence model is intentionally conservative about isolated spikes but
    much less likely than the legacy detector to erase sustained moderate
    disturbances.  Three robust local timescales are used:

    * 5 min: onset / rapid excursion evidence
    * 15 min: primary operational evidence
    * 60 min: sustained storm context

    Storms use hysteresis and a 30-minute release requirement.  This prevents
    one physical disturbance from becoming many artificial events.
    """
    x = np.asarray(residual, dtype=float)
    magnitude = np.abs(x)
    valid = np.isfinite(magnitude)
    safe = np.where(valid, magnitude, np.nan)

    fast = _rolling_median(safe, _odd_window(5 * 60, cadence_s, 31))
    medium = _rolling_median(safe, _odd_window(15 * 60, cadence_s, 61))
    slow = _rolling_median(safe, _odd_window(60 * 60, cadence_s, 121))

    # Fill only missing envelope positions; never manufacture signal from NaNs.
    for arr in (fast, medium, slow):
        arr[~np.isfinite(arr)] = 0.0

    # Sustained evidence: the 60-minute context can lower the required
    # instantaneous amplitude, but only when the 15-minute envelope also shows
    # a real excursion.  Fast evidence catches rapidly developing events.
    active_evidence = (
        (medium >= active_threshold)
        | ((slow >= 0.75 * active_threshold) & (medium >= 0.60 * active_threshold))
        | (fast >= 1.35 * active_threshold)
    )
    storm_evidence = (
        (medium >= storm_threshold)
        | ((slow >= 0.65 * storm_threshold) & (medium >= 0.70 * storm_threshold))
        | (fast >= 1.30 * storm_threshold)
    )

    # Release is based on the 15-minute envelope and is lower than onset.  The
    # state machine itself supplies temporal hysteresis.
    active_release = 0.75 * active_threshold
    storm_release = 0.70 * storm_threshold

    active = _hysteresis_mask(
        active_evidence,
        medium <= active_release,
        max(1, int(round(5 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(20 * 60 / max(cadence_s, 1.0)))),
    )
    storm = _hysteresis_mask(
        storm_evidence,
        medium <= storm_release,
        max(1, int(round(8 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(30 * 60 / max(cadence_s, 1.0)))),
    )

    # Severity is derived from the same physical envelopes, but remains nested
    # inside the storm state.  Major/severe states use the 15-minute envelope
    # plus a fast-excursion path so a sharp storm is not flattened away.
    major_evidence = (medium >= major_threshold) | (fast >= 1.15 * major_threshold)
    severe_evidence = (medium >= severe_threshold) | (fast >= 1.10 * severe_threshold)
    major = _hysteresis_mask(
        major_evidence,
        medium <= 0.75 * major_threshold,
        max(1, int(round(8 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(30 * 60 / max(cadence_s, 1.0)))),
    ) & storm
    severe = _hysteresis_mask(
        severe_evidence,
        medium <= 0.75 * severe_threshold,
        max(1, int(round(8 * 60 / max(cadence_s, 1.0)))),
        max(1, int(round(30 * 60 / max(cadence_s, 1.0)))),
    ) & major

    # Do not report activity across unusable samples.
    active &= valid
    storm &= valid
    major &= valid
    severe &= valid

    # Robust step anomaly: a fixed threshold remains the safety floor, while a
    # local MAD scale catches station-specific telemetry discontinuities.
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
        "slow_60m_nt": slow,
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
    """Classify residuals with deterministic multi-timescale hysteresis."""
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
