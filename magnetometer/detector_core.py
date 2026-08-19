#!/usr/bin/env python3
"""Production causal magnetometer detector.

The live detector and the historical evaluator use this exact implementation.
It never reads future samples. Detection is based on causal, robust
multi-scale residual statistics rather than a large collection of independently
tuned ratio rules.

A detector profile is intentionally small: absolute disturbance thresholds,
robust-normalized thresholds, and persistence/release timings. The remaining
legacy profile fields are accepted for backwards compatibility but are not used
by the v2 scoring path.
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

DETECTOR_VERSION = "causal-disturbance-v2"
DEFAULT_UNSETTLED_NT = 10.0
DEFAULT_ACTIVE_NT = 15.0
DEFAULT_STORM_NT = 35.0
DEFAULT_MAJOR_STORM_NT = 100.0
DEFAULT_SEVERE_STORM_NT = 200.0
DEFAULT_NOISE_FLOOR_NT = 1.5
DEFAULT_ACTIVE_Z = 3.0
DEFAULT_STORM_Z = 4.5
DEFAULT_ACTIVE_PERSISTENCE_MINUTES = 3.0
DEFAULT_STORM_PERSISTENCE_MINUTES = 10.0
DEFAULT_ACTIVE_RELEASE_MINUTES = 20.0
DEFAULT_STORM_RELEASE_MINUTES = 120.0
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
    normalized_active_fraction: float = 0.50
    normalized_storm_fraction: float = 0.70
    active_on_minutes: float = DEFAULT_ACTIVE_PERSISTENCE_MINUTES
    active_off_minutes: float = DEFAULT_ACTIVE_RELEASE_MINUTES
    storm_on_minutes: float = DEFAULT_STORM_PERSISTENCE_MINUTES
    storm_off_minutes: float = DEFAULT_STORM_RELEASE_MINUTES
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
    peak_window_minutes: float = 5.0
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
        if not (0 < self.normalized_active_fraction <= 1 and 0 < self.normalized_storm_fraction <= 1):
            raise ValueError("normalized evidence fractions must be in (0, 1]")
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
        raise RuntimeError(f"detector profile {candidate} targets {version!r}, expected {DETECTOR_VERSION!r}")
    return DetectorProfile.from_dict(payload.get("profile", payload))


def load_detector_profile(path: Optional[Path | str] = None) -> DetectorProfile:
    candidate = Path(path or os.environ.get(PROFILE_ENV, PROFILE_PATH)).resolve()
    return _load_profile_cached(str(candidate))


def _window(seconds: float, cadence_s: float, cap: int = 0) -> int:
    n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
    return min(n, cap) if cap else n


def _trailing_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).median().to_numpy(dtype=float, copy=False)


def _trailing_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).quantile(q).to_numpy(dtype=float, copy=False)


def _trailing_max(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).max().to_numpy(dtype=float, copy=False)


def _hysteresis_mask(evidence_on: np.ndarray, evidence_off: np.ndarray, min_on: int, min_off: int, valid: np.ndarray) -> np.ndarray:
    on = np.asarray(evidence_on, dtype=bool)
    off = np.asarray(evidence_off, dtype=bool)
    valid_mask = np.asarray(valid, dtype=bool)
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
    fast = _trailing_median(safe, _window(5 * 60, cadence_s, 31))
    medium = _trailing_median(safe, _window(15 * 60, cadence_s, 61))
    upper = _trailing_quantile(safe, _window(60 * 60, cadence_s, 181), 0.75)
    slow = _trailing_median(safe, _window(60 * 60, cadence_s, 181))
    slow_3h = _trailing_median(safe, _window(3 * 3600, cadence_s, 361))
    peak = _trailing_max(safe, _window(p.peak_window_minutes * 60, cadence_s, 61))
    center = _trailing_median(x, _window(60 * 60, cadence_s, 181))
    mad = _trailing_median(np.abs(x - center), _window(60 * 60, cadence_s, 181))
    scale = np.maximum(p.noise_floor_nt, 1.4826 * mad)
    ready = valid & np.isfinite(fast) & np.isfinite(medium) & np.isfinite(upper) & np.isfinite(slow) & np.isfinite(slow_3h) & np.isfinite(peak) & np.isfinite(scale)
    return {
        "valid": valid, "ready": ready, "fast": fast, "medium": medium,
        "upper": upper, "slow": slow, "slow_3h": slow_3h, "peak": peak,
        "scale": scale, "z_fast": fast / scale, "z_medium": medium / scale,
        "z_upper": upper / scale, "z_slow": slow / scale, "z_peak": peak / scale,
    }


def _causal_anomaly_mask(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    diff = np.diff(x, prepend=np.nan)
    w = _window(3 * 3600, cadence_s, 361)
    center = _trailing_median(diff, w)
    mad = _trailing_median(np.abs(diff - center), w)
    threshold = np.maximum(100.0, 8.0 * 1.4826 * np.maximum(mad, 1e-6))
    anomaly = np.isfinite(diff) & np.isfinite(center) & (np.abs(diff - center) >= threshold)
    return anomaly, threshold


def detect_activity_masks(residual: np.ndarray, cadence_s: float = 60.0, active_threshold: Optional[float] = None, storm_threshold: Optional[float] = None, unsettled_threshold: Optional[float] = None, major_threshold: Optional[float] = None, severe_threshold: Optional[float] = None, profile: Optional[DetectorProfile] = None, *, include_anomaly: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
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
    valid, ready = f["valid"], f["ready"]
    active_evidence = ready & (
        (f["medium"] >= active_threshold)
        | ((f["z_medium"] >= p.active_z) & (f["medium"] >= p.normalized_active_fraction * active_threshold))
        | ((f["z_fast"] >= p.active_z) & (f["fast"] >= p.normalized_active_fraction * active_threshold) & (f["medium"] >= 0.35 * active_threshold))
        | ((f["z_peak"] >= p.active_z) & (f["peak"] >= 1.25 * active_threshold) & (f["medium"] >= 0.35 * active_threshold))
    )
    storm_evidence = ready & (
        (f["medium"] >= storm_threshold)
        | ((f["z_medium"] >= p.storm_z) & (f["medium"] >= p.normalized_storm_fraction * storm_threshold))
        | ((f["z_slow"] >= p.storm_z) & (f["slow"] >= p.normalized_storm_fraction * storm_threshold) & (f["medium"] >= 0.55 * storm_threshold))
        | ((f["z_peak"] >= p.storm_z) & (f["peak"] >= 1.10 * storm_threshold) & (f["medium"] >= 0.50 * storm_threshold))
    )
    active = _hysteresis_mask(active_evidence, ready & (f["medium"] <= 0.55 * active_threshold), _window(p.active_on_minutes * 60, cadence_s), _window(p.active_off_minutes * 60, cadence_s), ready)
    storm = _hysteresis_mask(storm_evidence, ready & (f["medium"] <= 0.60 * storm_threshold), _window(p.storm_on_minutes * 60, cadence_s), _window(p.storm_off_minutes * 60, cadence_s), ready)
    major_evidence = ready & ((f["medium"] >= major_threshold) | ((f["upper"] >= 0.90 * major_threshold) & (f["medium"] >= 0.65 * major_threshold)) | ((f["peak"] >= major_threshold) & (f["medium"] >= 0.50 * major_threshold)))
    severe_evidence = ready & ((f["medium"] >= severe_threshold) | ((f["upper"] >= 0.90 * severe_threshold) & (f["medium"] >= 0.65 * severe_threshold)) | ((f["peak"] >= severe_threshold) & (f["medium"] >= 0.50 * severe_threshold)))
    major = _hysteresis_mask(major_evidence, ready & (f["medium"] <= 0.70 * major_threshold), _window(10 * 60, cadence_s), _window(30 * 60, cadence_s), ready) & storm
    severe = _hysteresis_mask(severe_evidence, ready & (f["medium"] <= 0.70 * severe_threshold), _window(10 * 60, cadence_s), _window(30 * 60, cadence_s), ready) & major
    active &= valid & ready
    storm &= valid & ready
    major &= valid & ready
    severe &= valid & ready
    diagnostics: Dict[str, object] = {
        "detector_version": DETECTOR_VERSION,
        "fast_5m_nt": f["fast"], "medium_15m_nt": f["medium"],
        "upper_60m_p75_nt": f["upper"], "slow_60m_nt": f["slow"],
        "slow_3h_nt": f["slow_3h"], "peak_window_nt": f["peak"],
        "local_scale_nt": f["scale"], "z_fast": f["z_fast"],
        "z_medium": f["z_medium"], "z_upper": f["z_upper"],
        "z_slow": f["z_slow"], "z_peak": f["z_peak"],
        "active_evidence": active_evidence, "storm_evidence": storm_evidence,
        "history_ready": ready, "invalid_input": ~valid,
        "warmup_samples": int(np.argmax(ready)) if np.any(ready) else int(x.size),
        "unsettled_threshold_nt": np.full(x.size, unsettled_threshold, dtype=float),
    }
    if include_anomaly:
        anomaly, anomaly_threshold = _causal_anomaly_mask(x, cadence_s)
        diagnostics["anomaly"] = anomaly & valid
        diagnostics["anomaly_threshold_nt"] = anomaly_threshold
    return active, storm, major, severe, diagnostics


def flag_activity(residual: np.ndarray, cadence_s: float = 60.0, active_threshold: Optional[float] = None, storm_threshold: Optional[float] = None, unsettled_threshold: Optional[float] = None, major_threshold: Optional[float] = None, severe_threshold: Optional[float] = None, profile: Optional[DetectorProfile] = None) -> np.ndarray:
    p = profile or load_detector_profile()
    active, storm, major, severe, diagnostics = detect_activity_masks(residual, cadence_s, active_threshold, storm_threshold, unsettled_threshold, major_threshold, severe_threshold, p, include_anomaly=True)
    medium = np.asarray(diagnostics["medium_15m_nt"], dtype=float)
    ready = np.asarray(diagnostics["history_ready"], dtype=bool)
    unsettled = p.unsettled_nt if unsettled_threshold is None else float(unsettled_threshold)
    x = np.asarray(residual, dtype=float)
    flags = np.full(x.size, "quiet", dtype=object)
    flags[ready & (medium >= unsettled)] = "unsettled"
    flags[active] = "active"
    flags[storm] = "minor_storm"
    flags[major] = "major_storm"
    flags[severe] = "severe_storm"
    anomaly = np.asarray(diagnostics.get("anomaly", np.zeros(x.size, dtype=bool)))
    flags[anomaly & ready & ~active & ~storm] = "anomaly"
    flags[~np.isfinite(x)] = "quiet"
    return flags
