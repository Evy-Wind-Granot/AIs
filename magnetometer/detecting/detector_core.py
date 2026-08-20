#!/usr/bin/env python3
"""Deterministic, strictly-causal production magnetometer detector.

The detector uses trailing statistics, explicit warm-up, causal hysteresis,
and a separate short-peak path so brief real disturbances are not lost by
long-window medians. Missing data always resets detector state.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_UNSETTLED_NT = 10.0
DEFAULT_ACTIVE_NT = 15.0
DEFAULT_STORM_NT = 35.0
DEFAULT_MAJOR_STORM_NT = 100.0
DEFAULT_SEVERE_STORM_NT = 200.0
DEFAULT_ANOMALY_DELTA_NT = 100.0

PROFILE_ENV = "MAGNETOMETER_DETECTOR_PROFILE"
PROFILE_PATH = Path(__file__).resolve().with_name("detector_profile.json")


@dataclass(frozen=True)
class DetectorProfile:
    active_nt: float = DEFAULT_ACTIVE_NT
    storm_nt: float = DEFAULT_STORM_NT
    unsettled_nt: float = DEFAULT_UNSETTLED_NT
    major_nt: float = DEFAULT_MAJOR_STORM_NT
    severe_nt: float = DEFAULT_SEVERE_STORM_NT

    active_slow_ratio: float = 0.65
    active_slow_3h_ratio: float = 0.55
    active_upper_ratio: float = 1.00
    active_fast_ratio: float = 1.25
    active_medium_slow_ratio: float = 0.40
    active_medium_upper_ratio: float = 0.35
    active_peak_ratio: float = 1.80
    active_peak_medium_ratio: float = 0.25

    storm_fast_ratio: float = 1.80
    storm_fast_medium_ratio: float = 0.55
    storm_upper_ratio: float = 1.10
    storm_upper_medium_ratio: float = 0.70
    storm_medium_ratio: float = 0.80
    storm_release_ratio: float = 0.65
    storm_peak_ratio: float = 1.70
    storm_peak_medium_ratio: float = 0.20

    peak_window_minutes: float = 5.0
    active_on_minutes: float = 5.0
    active_off_minutes: float = 30.0
    storm_on_minutes: float = 10.0
    storm_off_minutes: float = 180.0

    major_upper_ratio: float = 0.90
    major_fast_ratio: float = 1.15
    major_medium_ratio: float = 0.80
    severe_upper_ratio: float = 0.90
    severe_fast_ratio: float = 1.10
    severe_medium_ratio: float = 0.80

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "DetectorProfile":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        profile = cls(**{k: value[k] for k in allowed if k in value})
        profile.validate()
        return profile

    def validate(self) -> None:
        numeric = asdict(self)
        if not all(np.isfinite(float(v)) for v in numeric.values()):
            raise ValueError("detector profile contains non-finite values")
        if not (0 < self.active_nt < self.storm_nt <= self.major_nt <= self.severe_nt):
            raise ValueError("profile thresholds must satisfy active < storm <= major <= severe")
        if not (0 < self.unsettled_nt <= self.active_nt):
            raise ValueError("unsettled threshold must be positive and <= active threshold")
        if not (0 < self.peak_window_minutes <= 30):
            raise ValueError("peak window must be between 0 and 30 minutes")
        if not (0 < self.active_on_minutes <= self.active_off_minutes <= 24 * 60):
            raise ValueError("invalid active hysteresis durations")
        if not (0 < self.storm_on_minutes <= self.storm_off_minutes <= 24 * 60):
            raise ValueError("invalid storm hysteresis durations")
        bounded = (
            self.active_slow_ratio, self.active_slow_3h_ratio, self.active_upper_ratio,
            self.active_fast_ratio, self.active_medium_slow_ratio, self.active_medium_upper_ratio,
            self.active_peak_ratio, self.active_peak_medium_ratio,
            self.storm_fast_ratio, self.storm_fast_medium_ratio, self.storm_upper_ratio,
            self.storm_upper_medium_ratio, self.storm_medium_ratio, self.storm_release_ratio,
            self.storm_peak_ratio, self.storm_peak_medium_ratio,
            self.major_upper_ratio, self.major_fast_ratio, self.major_medium_ratio,
            self.severe_upper_ratio, self.severe_fast_ratio, self.severe_medium_ratio,
        )
        if any(v <= 0 or v > 3.0 for v in bounded):
            raise ValueError("detector evidence multipliers are outside safe bounds")


@lru_cache(maxsize=8)
def _load_profile_cached(path_text: str) -> DetectorProfile:
    candidate = Path(path_text)
    if not candidate.exists():
        return DetectorProfile()
    try:
        payload = json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load detector profile {candidate}: {exc}") from exc
    if payload.get("status") != "certified":
        raise RuntimeError(f"detector profile {candidate} is not certified")
    return DetectorProfile.from_dict(payload.get("profile", payload))


def load_detector_profile(path: Optional[Path | str] = None) -> DetectorProfile:
    candidate = Path(path or os.environ.get(PROFILE_ENV, PROFILE_PATH)).resolve()
    return _load_profile_cached(str(candidate))


def _window(seconds: float, cadence_s: float, cap: int = 0) -> int:
    n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
    return min(n, cap) if cap else n


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).median().to_numpy(dtype=float, copy=False)


def _rolling_quantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).quantile(quantile).to_numpy(dtype=float, copy=False)


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).max().to_numpy(dtype=float, copy=False)


def _hysteresis_mask(
    evidence_on: np.ndarray,
    evidence_off: np.ndarray,
    min_on: int,
    min_off: int,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    on = np.asarray(evidence_on, dtype=bool)
    off = np.asarray(evidence_off, dtype=bool)
    if on.shape != off.shape:
        raise ValueError("evidence_on and evidence_off must have identical shapes")
    valid_mask = np.ones(on.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid_mask.shape != on.shape:
        raise ValueError("valid mask must have identical shape")
    out = np.zeros(on.size, dtype=bool)
    state = False
    candidate = 0
    min_on = max(1, int(min_on))
    min_off = max(1, int(min_off))
    for i in range(on.size):
        if not valid_mask[i]:
            state = False
            candidate = 0
            continue
        if not state:
            candidate = candidate + 1 if on[i] else 0
            if candidate >= min_on:
                state = True
                candidate = 0
        else:
            candidate = candidate + 1 if off[i] else 0
            if candidate >= min_off:
                state = False
                candidate = 0
        out[i] = state
    return out


def _causal_anomaly_mask(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, float, np.ndarray]:
    diff = np.diff(x, prepend=np.nan)
    w = _window(3 * 3600, cadence_s, 721)
    d = pd.Series(diff, copy=False)
    med = d.rolling(w, min_periods=w).median().to_numpy(dtype=float, copy=False)
    mad = pd.Series(np.abs(diff - med), copy=False).rolling(w, min_periods=w).median().to_numpy(dtype=float, copy=False)
    threshold = np.maximum(DEFAULT_ANOMALY_DELTA_NT, 8.0 * 1.4826 * mad)
    threshold[~np.isfinite(threshold)] = DEFAULT_ANOMALY_DELTA_NT
    anomaly = np.isfinite(diff) & np.isfinite(med) & (np.abs(diff - med) >= threshold)
    finite_threshold = threshold[np.isfinite(threshold)]
    median_threshold = float(np.median(finite_threshold)) if finite_threshold.size else DEFAULT_ANOMALY_DELTA_NT
    return anomaly, median_threshold, threshold


def detect_activity_masks(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: Optional[float] = None,
    storm_threshold: Optional[float] = None,
    unsettled_threshold: Optional[float] = None,
    major_threshold: Optional[float] = None,
    severe_threshold: Optional[float] = None,
    profile: Optional[DetectorProfile] = None,
    *,
    include_anomaly: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Return strictly-causal activity/severity masks and diagnostics."""
    p = profile or load_detector_profile()
    active_threshold = p.active_nt if active_threshold is None else float(active_threshold)
    storm_threshold = p.storm_nt if storm_threshold is None else float(storm_threshold)
    unsettled_threshold = p.unsettled_nt if unsettled_threshold is None else float(unsettled_threshold)
    major_threshold = p.major_nt if major_threshold is None else float(major_threshold)
    severe_threshold = p.severe_nt if severe_threshold is None else float(severe_threshold)

    x = np.asarray(residual, dtype=float)
    if x.ndim != 1:
        raise ValueError("residual must be a one-dimensional array")
    if cadence_s <= 0 or not np.isfinite(cadence_s):
        raise ValueError("cadence_s must be a positive finite number")
    if not (0 < active_threshold < storm_threshold <= major_threshold <= severe_threshold):
        raise ValueError("thresholds must satisfy active < storm <= major <= severe")

    magnitude = np.abs(x)
    valid = np.isfinite(magnitude)
    safe = np.where(valid, magnitude, np.nan)

    fast = _rolling_median(safe, _window(5 * 60, cadence_s, 31))
    medium = _rolling_median(safe, _window(15 * 60, cadence_s, 61))
    upper_30m = _rolling_quantile(safe, _window(30 * 60, cadence_s, 121), 0.75)
    slow = _rolling_median(safe, _window(60 * 60, cadence_s, 181))
    slow_3h = _rolling_median(safe, _window(3 * 3600, cadence_s, 361))
    peak = _rolling_max(safe, _window(p.peak_window_minutes * 60, cadence_s, 61))
    history_ready = np.isfinite(fast) & np.isfinite(medium) & np.isfinite(upper_30m) & np.isfinite(slow) & np.isfinite(slow_3h) & np.isfinite(peak) & valid
    for arr in (fast, medium, upper_30m, slow, slow_3h, peak):
        arr[~np.isfinite(arr)] = 0.0

    active_evidence = history_ready & (
        (medium >= active_threshold)
        | ((slow >= p.active_slow_ratio * active_threshold) & (medium >= p.active_medium_slow_ratio * active_threshold))
        | ((slow_3h >= p.active_slow_3h_ratio * active_threshold) & (medium >= p.active_medium_slow_ratio * active_threshold))
        | ((upper_30m >= p.active_upper_ratio * active_threshold) & (medium >= p.active_medium_upper_ratio * active_threshold))
        | ((fast >= p.active_fast_ratio * active_threshold) & (medium >= p.active_medium_slow_ratio * active_threshold))
        | ((peak >= p.active_peak_ratio * active_threshold) & (medium >= p.active_peak_medium_ratio * active_threshold))
    )

    strong_short = (fast >= p.storm_fast_ratio * storm_threshold) & (medium >= p.storm_fast_medium_ratio * storm_threshold)
    strong_peak = (peak >= p.storm_peak_ratio * storm_threshold) & (medium >= p.storm_peak_medium_ratio * storm_threshold)
    strong_30m = (upper_30m >= p.storm_upper_ratio * storm_threshold) & (medium >= p.storm_upper_medium_ratio * storm_threshold)
    sustained = (
        (medium >= storm_threshold)
        | ((slow >= storm_threshold) & (medium >= p.storm_medium_ratio * storm_threshold))
        | ((slow_3h >= storm_threshold) & (medium >= p.storm_medium_ratio * storm_threshold))
    )
    storm_evidence = history_ready & (sustained | strong_short | strong_peak | strong_30m)

    active = _hysteresis_mask(
        active_evidence,
        history_ready & (medium <= 0.60 * active_threshold),
        _window(p.active_on_minutes * 60, cadence_s),
        _window(p.active_off_minutes * 60, cadence_s),
        valid=history_ready,
    )
    storm = _hysteresis_mask(
        storm_evidence,
        history_ready & (medium <= p.storm_release_ratio * storm_threshold),
        _window(p.storm_on_minutes * 60, cadence_s),
        _window(p.storm_off_minutes * 60, cadence_s),
        valid=history_ready,
    )

    major_evidence = history_ready & (
        (medium >= major_threshold)
        | ((upper_30m >= p.major_upper_ratio * major_threshold) & (medium >= p.major_medium_ratio * major_threshold))
        | ((peak >= p.major_fast_ratio * major_threshold) & (medium >= p.major_medium_ratio * major_threshold))
    )
    severe_evidence = history_ready & (
        (medium >= severe_threshold)
        | ((upper_30m >= p.severe_upper_ratio * severe_threshold) & (medium >= p.severe_medium_ratio * severe_threshold))
        | ((peak >= p.severe_fast_ratio * severe_threshold) & (medium >= p.severe_medium_ratio * severe_threshold))
    )
    major = _hysteresis_mask(
        major_evidence,
        history_ready & (medium <= 0.75 * major_threshold),
        _window(10 * 60, cadence_s),
        _window(30 * 60, cadence_s),
        valid=history_ready,
    ) & storm
    severe = _hysteresis_mask(
        severe_evidence,
        history_ready & (medium <= 0.75 * severe_threshold),
        _window(10 * 60, cadence_s),
        _window(30 * 60, cadence_s),
        valid=history_ready,
    ) & major

    active &= valid & history_ready
    storm &= valid & history_ready
    major &= valid & history_ready
    severe &= valid & history_ready

    diagnostics: Dict[str, object] = {
        "fast_5m_nt": fast,
        "medium_15m_nt": medium,
        "upper_30m_p75_nt": upper_30m,
        "slow_60m_nt": slow,
        "slow_3h_nt": slow_3h,
        "peak_window_nt": peak,
        "storm_evidence": storm_evidence,
        "storm_strong_short_evidence": history_ready & strong_short,
        "storm_strong_peak_evidence": history_ready & strong_peak,
        "storm_strong_30m_evidence": history_ready & strong_30m,
        "storm_sustained_evidence": history_ready & sustained,
        "history_ready": history_ready,
        "invalid_input": ~valid,
        "warmup_samples": int(np.argmax(history_ready)) if np.any(history_ready) else int(x.size),
        "unsettled_threshold_nt": np.full(x.size, unsettled_threshold, dtype=float),
    }
    if include_anomaly:
        anomaly, anomaly_median_threshold, anomaly_threshold = _causal_anomaly_mask(x, cadence_s)
        diagnostics.update({
            "anomaly_threshold_nt": anomaly_threshold,
            "anomaly_median_threshold_nt": np.full(x.size, anomaly_median_threshold, dtype=float),
            "anomaly": anomaly & valid,
        })
    return active, storm, major, severe, diagnostics


def flag_activity(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    active_threshold: Optional[float] = None,
    storm_threshold: Optional[float] = None,
    unsettled_threshold: Optional[float] = None,
    major_threshold: Optional[float] = None,
    severe_threshold: Optional[float] = None,
    profile: Optional[DetectorProfile] = None,
) -> np.ndarray:
    """Classify residuals using the certified profile."""
    x = np.asarray(residual, dtype=float)
    p = profile or load_detector_profile()
    active, storm, major, severe, diagnostics = detect_activity_masks(
        x,
        cadence_s=cadence_s,
        active_threshold=active_threshold,
        storm_threshold=storm_threshold,
        unsettled_threshold=unsettled_threshold,
        major_threshold=major_threshold,
        severe_threshold=severe_threshold,
        profile=p,
        include_anomaly=True,
    )
    medium = diagnostics["medium_15m_nt"]
    ready = diagnostics["history_ready"]
    unsettled = p.unsettled_nt if unsettled_threshold is None else float(unsettled_threshold)
    flags = np.full(x.size, "quiet", dtype=object)
    flags[ready & (medium >= unsettled)] = "unsettled"
    flags[active] = "active"
    flags[storm] = "minor_storm"
    flags[major] = "major_storm"
    flags[severe] = "severe_storm"
    anomaly = diagnostics["anomaly"]
    flags[anomaly & ready & ~active & ~storm] = "anomaly"
    flags[~np.isfinite(x)] = "quiet"
    return flags


__all__ = [
    "DetectorProfile",
    "detect_activity_masks",
    "flag_activity",
    "load_detector_profile",
]
