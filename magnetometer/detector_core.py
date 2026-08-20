#!/usr/bin/env python3
"""Causal production magnetometer detector.

The live detector and historical evaluator share this implementation. All
statistics are trailing-only, missing data resets detector state, and local
normalization is corroborating evidence only: it can never promote a small
absolute disturbance into an activity/storm alert by itself.
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

DETECTOR_VERSION = "causal-disturbance-v2.1"
DEFAULT_UNSETTLED_NT = 10.0
DEFAULT_ACTIVE_NT = 15.0
DEFAULT_STORM_NT = 35.0
DEFAULT_MAJOR_STORM_NT = 100.0
DEFAULT_SEVERE_STORM_NT = 200.0
DEFAULT_NOISE_FLOOR_NT = 2.0
DEFAULT_ACTIVE_Z = 3.0
DEFAULT_STORM_Z = 4.5
DEFAULT_ACTIVE_ON_MINUTES = 3.0
DEFAULT_ACTIVE_OFF_MINUTES = 30.0
DEFAULT_STORM_ON_MINUTES = 10.0
DEFAULT_STORM_OFF_MINUTES = 180.0
DEFAULT_PEAK_WINDOW_MINUTES = 5.0

PROFILE_ENV = "MAGNETOMETER_DETECTOR_PROFILE"
PROFILE_PATH = Path(__file__).resolve().with_name("detector_profile.json")


@dataclass(frozen=True)
class DetectorProfile:
    active_nt: float = DEFAULT_ACTIVE_NT
    storm_nt: float = DEFAULT_STORM_NT
    unsettled_nt: float = DEFAULT_UNSETTLED_NT
    major_nt: float = DEFAULT_MAJOR_STORM_NT
    severe_nt: float = DEFAULT_SEVERE_STORM_NT

    active_z: float = DEFAULT_ACTIVE_Z
    storm_z: float = DEFAULT_STORM_Z
    noise_floor_nt: float = DEFAULT_NOISE_FLOOR_NT
    active_on_minutes: float = DEFAULT_ACTIVE_ON_MINUTES
    active_off_minutes: float = DEFAULT_ACTIVE_OFF_MINUTES
    storm_on_minutes: float = DEFAULT_STORM_ON_MINUTES
    storm_off_minutes: float = DEFAULT_STORM_OFF_MINUTES
    peak_window_minutes: float = DEFAULT_PEAK_WINDOW_MINUTES

    active_slow_ratio: float = 0.65
    active_slow_3h_ratio: float = 0.55
    active_upper_ratio: float = 1.0
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
        values = asdict(self)
        if not all(np.isfinite(float(v)) for v in values.values()):
            raise ValueError("detector profile contains non-finite values")
        if not (0 < self.unsettled_nt <= self.active_nt < self.storm_nt <= self.major_nt <= self.severe_nt):
            raise ValueError("thresholds must satisfy unsettled <= active < storm <= major <= severe")
        if self.noise_floor_nt <= 0:
            raise ValueError("noise_floor_nt must be positive")
        if not (0 < self.active_z < self.storm_z):
            raise ValueError("normalized evidence thresholds must satisfy active_z < storm_z")
        if not (0 < self.active_on_minutes <= self.active_off_minutes <= 24 * 60):
            raise ValueError("invalid active persistence/release durations")
        if not (0 < self.storm_on_minutes <= self.storm_off_minutes <= 24 * 60):
            raise ValueError("invalid storm persistence/release durations")
        if not (0 < self.peak_window_minutes <= 30):
            raise ValueError("peak window must be between 0 and 30 minutes")


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
    version = payload.get("detector_version")
    if version not in (None, DETECTOR_VERSION):
        raise RuntimeError(
            f"detector profile {candidate} targets {version!r}; "
            f"expected {DETECTOR_VERSION}"
        )
    return DetectorProfile.from_dict(payload.get("profile", payload))


def load_detector_profile(path: Optional[Path | str] = None) -> DetectorProfile:
    candidate = Path(path or os.environ.get(PROFILE_ENV, PROFILE_PATH)).resolve()
    return _load_profile_cached(str(candidate))


def _window(seconds: float, cadence_s: float, cap: int = 0) -> int:
    n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
    return min(n, cap) if cap else n


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).median().to_numpy(dtype=float, copy=False)


def _rolling_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).quantile(q).to_numpy(dtype=float, copy=False)


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).max().to_numpy(dtype=float, copy=False)


def _hysteresis_mask(evidence_on: np.ndarray, evidence_off: np.ndarray, min_on: int, min_off: int, valid: Optional[np.ndarray] = None) -> np.ndarray:
    on = np.asarray(evidence_on, dtype=bool)
    off = np.asarray(evidence_off, dtype=bool)
    valid_mask = np.ones(on.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if not (on.shape == off.shape == valid_mask.shape):
        raise ValueError("hysteresis inputs must have identical shapes")
    out = np.zeros(on.size, dtype=bool)
    state = False
    candidate = 0
    for i in range(on.size):
        if not valid_mask[i]:
            state = False
            candidate = 0
            continue
        if not state:
            candidate = candidate + 1 if on[i] else 0
            if candidate >= max(1, int(min_on)):
                state = True
                candidate = 0
        else:
            candidate = candidate + 1 if off[i] else 0
            if candidate >= max(1, int(min_off)):
                state = False
                candidate = 0
        out[i] = state
    return out


def _causal_robust_features(x: np.ndarray, cadence_s: float, p: DetectorProfile) -> Dict[str, np.ndarray]:
    magnitude = np.abs(x)
    valid = np.isfinite(magnitude)
    safe = np.where(valid, magnitude, np.nan)
    fast = _rolling_median(safe, _window(5 * 60, cadence_s, 31))
    medium = _rolling_median(safe, _window(15 * 60, cadence_s, 61))
    upper_30m = _rolling_quantile(safe, _window(30 * 60, cadence_s, 121), 0.75)
    slow_60m = _rolling_median(safe, _window(60 * 60, cadence_s, 181))
    slow_3h = _rolling_median(safe, _window(3 * 3600, cadence_s, 361))
    peak = _rolling_max(safe, _window(p.peak_window_minutes * 60, cadence_s, 61))
    center = _rolling_median(x, _window(60 * 60, cadence_s, 181))
    mad = _rolling_median(np.abs(x - center), _window(60 * 60, cadence_s, 181))
    scale = np.maximum(p.noise_floor_nt, 1.4826 * mad)
    ready = valid & np.isfinite(fast) & np.isfinite(medium) & np.isfinite(upper_30m) & np.isfinite(slow_60m) & np.isfinite(slow_3h) & np.isfinite(peak) & np.isfinite(scale)
    return {"valid": valid, "ready": ready, "fast": fast, "medium": medium, "upper_30m": upper_30m, "slow_60m": slow_60m, "slow_3h": slow_3h, "peak": peak, "scale": scale, "z_fast": fast / scale, "z_medium": medium / scale, "z_slow": slow_60m / scale, "z_peak": peak / scale}


def _causal_anomaly_mask(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    diff = np.diff(x, prepend=np.nan)
    window = _window(3 * 3600, cadence_s, 361)
    center = _rolling_median(diff, window)
    mad = _rolling_median(np.abs(diff - center), window)
    threshold = np.maximum(100.0, 8.0 * 1.4826 * np.maximum(mad, 1e-6))
    anomaly = np.isfinite(diff) & np.isfinite(center) & (np.abs(diff - center) >= threshold)
    return anomaly, threshold


def detect_activity_masks(residual: np.ndarray, cadence_s: float = 60.0, active_threshold: Optional[float] = None, storm_threshold: Optional[float] = None, unsettled_threshold: Optional[float] = None, major_threshold: Optional[float] = None, severe_threshold: Optional[float] = None, profile: Optional[DetectorProfile] = None, *, include_anomaly: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Return strictly causal activity/severity masks and diagnostics."""
    p = profile or load_detector_profile()
    active_threshold = p.active_nt if active_threshold is None else float(active_threshold)
    storm_threshold = p.storm_nt if storm_threshold is None else float(storm_threshold)
    unsettled_threshold = p.unsettled_nt if unsettled_threshold is None else float(unsettled_threshold)
    major_threshold = p.major_nt if major_threshold is None else float(major_threshold)
    severe_threshold = p.severe_nt if severe_threshold is None else float(severe_threshold)
    x = np.asarray(residual, dtype=float)
    if x.ndim != 1:
        raise ValueError("residual must be one-dimensional")
    if cadence_s <= 0 or not np.isfinite(cadence_s):
        raise ValueError("cadence_s must be positive and finite")
    if not (0 < unsettled_threshold <= active_threshold < storm_threshold <= major_threshold <= severe_threshold):
        raise ValueError("thresholds must satisfy unsettled <= active < storm <= major <= severe")
    f = _causal_robust_features(x, cadence_s, p)
    valid = f["valid"]
    ready = f["ready"]
    active_evidence = ready & ((f["medium"] >= active_threshold) | ((f["fast"] >= active_threshold) & (f["medium"] >= 0.80 * active_threshold) & (f["z_fast"] >= p.active_z)) | ((f["peak"] >= 1.10 * active_threshold) & (f["medium"] >= 0.70 * active_threshold) & (f["z_peak"] >= p.active_z)))
    storm_evidence = ready & ((f["medium"] >= storm_threshold) | ((f["slow_60m"] >= storm_threshold) & (f["medium"] >= 0.80 * storm_threshold)) | ((f["fast"] >= storm_threshold) & (f["medium"] >= 0.80 * storm_threshold) & (f["z_fast"] >= p.storm_z)) | ((f["peak"] >= 1.10 * storm_threshold) & (f["medium"] >= 0.70 * storm_threshold) & (f["z_peak"] >= p.storm_z)))
    active = _hysteresis_mask(active_evidence, ready & (f["medium"] <= 0.55 * active_threshold), _window(p.active_on_minutes * 60, cadence_s), _window(p.active_off_minutes * 60, cadence_s), ready)
    storm = _hysteresis_mask(storm_evidence, ready & (f["medium"] <= 0.60 * storm_threshold), _window(p.storm_on_minutes * 60, cadence_s), _window(p.storm_off_minutes * 60, cadence_s), ready)
    major_evidence = ready & ((f["medium"] >= major_threshold) | ((f["upper_30m"] >= 0.90 * major_threshold) & (f["medium"] >= 0.70 * major_threshold)) | ((f["peak"] >= major_threshold) & (f["medium"] >= 0.60 * major_threshold)))
    severe_evidence = ready & ((f["medium"] >= severe_threshold) | ((f["upper_30m"] >= 0.90 * severe_threshold) & (f["medium"] >= 0.70 * severe_threshold)) | ((f["peak"] >= severe_threshold) & (f["medium"] >= 0.60 * severe_threshold)))
    major = _hysteresis_mask(major_evidence, ready & (f["medium"] <= 0.70 * major_threshold), _window(10 * 60, cadence_s), _window(30 * 60, cadence_s), ready) & storm
    severe = _hysteresis_mask(severe_evidence, ready & (f["medium"] <= 0.70 * severe_threshold), _window(10 * 60, cadence_s), _window(30 * 60, cadence_s), ready) & major
    active &= valid & ready
    storm &= valid & ready
    major &= valid & ready
    severe &= valid & ready
    diagnostics: Dict[str, object] = {"detector_version": DETECTOR_VERSION, "fast_5m_nt": f["fast"], "medium_15m_nt": f["medium"], "upper_30m_p75_nt": f["upper_30m"], "slow_60m_nt": f["slow_60m"], "slow_3h_nt": f["slow_3h"], "peak_nt": f["peak"], "local_scale_nt": f["scale"], "z_fast": f["z_fast"], "z_medium": f["z_medium"], "z_slow": f["z_slow"], "z_peak": f["z_peak"], "ready": ready}
    if include_anomaly:
        anomaly, anomaly_threshold = _causal_anomaly_mask(x, cadence_s)
        diagnostics["anomaly"] = anomaly
        diagnostics["anomaly_threshold_nt"] = anomaly_threshold
    return active, storm, major, severe, diagnostics
