"""Station-local adaptive event detector independent of Kp/Dst."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


EVENT_FLAGS = frozenset(
    {"unsettled", "active", "minor_storm", "major_storm", "severe_storm"}
)


def event_mask_from_flags(flags: np.ndarray) -> np.ndarray:
    """Return the authoritative sustained-event mask."""
    f = np.asarray(flags, dtype=object)
    return np.isin(f, list(EVENT_FLAGS))


@dataclass(frozen=True)
class AdaptiveConfig:
    # Pointwise onset remains conservative and is useful for impulsive events.
    onset_sigma: float = 6.0
    # Sustained moderate excursions provide the recall path for gradual
    # disturbances, but must be materially above the local noise floor.
    sustained_sigma: float = 6.0
    sustained_minutes: float = 8.0
    sustained_window_minutes: float = 15.0
    # Confirmation prevents a single moderate sample from opening an event.
    sustained_confirm_sigma: float = 3.5
    active_sigma: float = 10.0
    storm_sigma: float = 18.0
    major_sigma: float = 30.0
    severe_sigma: float = 55.0
    clear_sigma: float = 3.0
    derivative_onset_sigma: float = 7.0
    derivative_clear_sigma: float = 3.5
    derivative_confirm_sigma: float = 3.5
    rolling_scale_minutes: float = 180.0
    quiet_scale_quantile: float = 0.35
    # Keep an upper cap so event-contaminated rolling windows cannot inflate
    # their own threshold enough to hide a sustained disturbance.
    max_scale_multiplier: float = 2.0
    onset_minutes: float = 5.0
    clear_minutes: float = 10.0
    min_event_minutes: float = 10.0
    merge_gap_minutes: float = 20.0
    derivative_hold_minutes: float = 3.0


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = float(np.std(x))
    return max(float(sigma), 1e-6)


def _quiet_floor(x: np.ndarray, quantile: float) -> float:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 20:
        return 1.0
    med = float(np.median(finite))
    amp = np.abs(finite - med)
    cutoff = float(np.quantile(amp, np.clip(quantile, 0.05, 0.8)))
    return _robust_sigma(finite[amp <= cutoff])


def _rolling_robust_scale(x: np.ndarray, window: int, floor: float, cap: float) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    min_periods = min(max(10, window // 10), window)
    med = s.rolling(window=window, min_periods=min_periods).median()
    deviation = (s - med).abs()
    mad = deviation.rolling(window=window, min_periods=min_periods).median()
    scale = 1.4826 * mad
    fallback = s.rolling(window=window, min_periods=min_periods).std(ddof=0)
    scale = scale.where(np.isfinite(scale) & (scale > 1e-9), fallback)
    scale = scale.ffill().bfill().fillna(floor)
    return np.clip(scale.to_numpy(float), floor, cap)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    x = np.asarray(mask, dtype=bool)
    if not x.size:
        return []
    p = np.concatenate(([False], x, [False]))
    starts = np.flatnonzero(~p[:-1] & p[1:])
    ends = np.flatnonzero(p[:-1] & ~p[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in runs:
        if out and s - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _rolling_rms(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    rms = np.sqrt(s.pow(2).rolling(window=window, min_periods=min_periods).mean())
    return rms.ffill().bfill().to_numpy(float)


def detect_adaptive(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    config: AdaptiveConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect sustained local disturbances using adaptive robust thresholds.

    Onset uses three independent paths: conservative pointwise amplitude,
    persistent short-window energy, and fast derivative change with amplitude
    confirmation. Once accepted, the state machine owns the event mask and
    severity labels cannot fragment it.
    """
    cfg = config or AdaptiveConfig()
    r = np.asarray(residual, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty(0, dtype=object), {"config": asdict(cfg)}

    finite = np.isfinite(r)
    if not finite.all():
        good = np.flatnonzero(finite)
        if len(good) < 2:
            flags = np.full(n, "quiet", dtype=object)
            return flags, {"config": asdict(cfg), "noise_sigma_nt": None, "event_count": 0}
        r = np.interp(np.arange(n), good, r[good])

    center = float(np.median(r))
    centered = r - center

    global_floor = _quiet_floor(centered, cfg.quiet_scale_quantile)
    scale_window = max(11, int(round(cfg.rolling_scale_minutes * 60.0 / cadence_s)))
    scale = _rolling_robust_scale(
        centered,
        scale_window,
        global_floor,
        global_floor * max(cfg.max_scale_multiplier, 1.0),
    )
    amplitude_z = np.abs(centered) / scale

    sustained_window_n = max(3, int(round(cfg.sustained_window_minutes * 60.0 / cadence_s)))
    sustained_min_n = max(3, sustained_window_n // 3)
    sustained_rms = _rolling_rms(centered, sustained_window_n, sustained_min_n)
    sustained_scale = np.maximum(scale, global_floor)
    sustained_z = sustained_rms / sustained_scale

    diff = np.diff(r, prepend=r[0]) / max(cadence_s, 1.0)
    derivative_floor = _quiet_floor(diff, cfg.quiet_scale_quantile)
    derivative_window = max(11, int(round(60.0 * 60.0 / cadence_s)))
    derivative_scale = _rolling_robust_scale(
        diff,
        derivative_window,
        derivative_floor,
        derivative_floor * max(cfg.max_scale_multiplier, 1.0),
    )
    derivative_z = np.abs(diff) / derivative_scale

    derivative_hold_n = max(1, int(round(cfg.derivative_hold_minutes * 60.0 / cadence_s)))
    derivative_trigger = np.zeros(n, dtype=bool)
    for s, e in _runs(derivative_z >= cfg.derivative_onset_sigma):
        if e - s >= derivative_hold_n:
            derivative_trigger[s:e] = True

    point_onset = amplitude_z >= cfg.onset_sigma
    sustained_onset = (
        (sustained_z >= cfg.sustained_sigma)
        & (amplitude_z >= cfg.sustained_confirm_sigma)
    )
    derivative_onset = derivative_trigger & (amplitude_z >= cfg.derivative_confirm_sigma)

    sustained_n = max(1, int(round(cfg.sustained_minutes * 60.0 / cadence_s)))
    sustained_onset_persistent = np.zeros(n, dtype=bool)
    for s, e in _runs(sustained_onset):
        if e - s >= sustained_n:
            sustained_onset_persistent[s:e] = True

    strong_onset = point_onset | derivative_onset
    onset_signal = strong_onset | sustained_onset_persistent

    clear_signal = (
        (amplitude_z <= cfg.clear_sigma)
        & (sustained_z <= cfg.sustained_sigma * 0.75)
        & (derivative_z <= cfg.derivative_clear_sigma)
    )

    onset_n = max(1, int(round(cfg.onset_minutes * 60.0 / cadence_s)))
    clear_n = max(1, int(round(cfg.clear_minutes * 60.0 / cadence_s)))
    min_n = max(1, int(round(cfg.min_event_minutes * 60.0 / cadence_s)))
    merge_n = max(0, int(round(cfg.merge_gap_minutes * 60.0 / cadence_s)))

    active = np.zeros(n, dtype=bool)
    above = below = 0
    state = False
    for i in range(n):
        if onset_signal[i]:
            above += 1
            below = 0
        elif clear_signal[i]:
            below += 1
            above = 0
        else:
            above = below = 0
        if not state and above >= onset_n:
            state = True
            below = 0
        elif state and below >= clear_n:
            state = False
            above = 0
        active[i] = state

    runs = _merge_runs(_runs(active), merge_n)
    clean = np.zeros(n, dtype=bool)
    for s, e in runs:
        if e - s >= min_n:
            clean[s:e] = True

    flags = np.full(n, "quiet", dtype=object)
    flags[clean] = "unsettled"
    flags[clean & (amplitude_z >= cfg.active_sigma)] = "active"
    flags[clean & (amplitude_z >= cfg.storm_sigma)] = "minor_storm"
    flags[clean & (amplitude_z >= cfg.major_sigma)] = "major_storm"
    flags[clean & (amplitude_z >= cfg.severe_sigma)] = "severe_storm"

    anomaly = (derivative_z >= cfg.derivative_onset_sigma * 1.8) & ~clean
    flags[anomaly] = "anomaly"

    event_mask = event_mask_from_flags(flags)
    event_runs = _runs(event_mask)
    anomaly_runs = _runs(flags == "anomaly")
    diagnostics: Dict[str, Any] = {
        "config": asdict(cfg),
        "noise_sigma_nt": float(global_floor),
        "derivative_sigma_nt_per_s": float(derivative_floor),
        "rolling_scale_min_nt": float(np.min(scale)),
        "rolling_scale_median_nt": float(np.median(scale)),
        "rolling_scale_max_nt": float(np.max(scale)),
        "onset_threshold_nt_median": float(cfg.onset_sigma * np.median(scale)),
        "sustained_threshold_nt_median": float(cfg.sustained_sigma * np.median(scale)),
        "sustained_confirm_threshold_nt_median": float(cfg.sustained_confirm_sigma * np.median(scale)),
        "active_threshold_nt_median": float(cfg.active_sigma * np.median(scale)),
        "storm_threshold_nt_median": float(cfg.storm_sigma * np.median(scale)),
        "major_threshold_nt_median": float(cfg.major_sigma * np.median(scale)),
        "severe_threshold_nt_median": float(cfg.severe_sigma * np.median(scale)),
        "flagged_fraction": float(np.mean(event_mask)),
        "event_count": len(event_runs),
        "event_durations_minutes": [round((e - s) * cadence_s / 60.0, 2) for s, e in event_runs],
        "anomaly_count": len(anomaly_runs),
        "anomaly_sample_fraction": float(np.mean(flags == "anomaly")),
        "max_amplitude_z": float(np.nanmax(amplitude_z)),
        "max_sustained_z": float(np.nanmax(sustained_z)),
        "max_derivative_z": float(np.nanmax(derivative_z)),
        "onset_path_counts": {
            "pointwise": int(np.sum(point_onset)),
            "sustained": int(np.sum(sustained_onset_persistent)),
            "derivative": int(np.sum(derivative_onset)),
        },
    }
    return flags, diagnostics
