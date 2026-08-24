"""Station-local adaptive event detector independent of Kp/Dst.

The detector is deliberately local: it estimates a quiet noise floor from the
residual itself, uses rolling robust scale for changing station conditions,
and applies hysteresis/persistence so individual spikes do not become events.
Kp and Dst are not inputs to detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    # Pointwise entry/exit levels in units of local robust sigma.
    onset_sigma: float = 6.0
    active_sigma: float = 10.0
    storm_sigma: float = 18.0
    major_sigma: float = 30.0
    severe_sigma: float = 55.0
    clear_sigma: float = 3.0

    # Independent derivative trigger, in robust sigma of d(residual)/dt.
    derivative_onset_sigma: float = 7.0
    derivative_clear_sigma: float = 3.5

    # Adaptive scale windows. The quiet floor prevents the rolling scale from
    # inflating indefinitely during a long storm.
    rolling_scale_minutes: float = 180.0
    quiet_scale_quantile: float = 0.35
    max_scale_multiplier: float = 3.0

    # Temporal logic.
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


def _quiet_floor(residual: np.ndarray, quantile: float) -> float:
    r = np.asarray(residual, dtype=float)
    finite = r[np.isfinite(r)]
    if finite.size < 20:
        return 1.0
    med = float(np.median(finite))
    amp = np.abs(finite - med)
    cutoff = float(np.quantile(amp, np.clip(quantile, 0.05, 0.8)))
    quiet = finite[amp <= cutoff]
    return _robust_sigma(quiet)


def _rolling_mad_scale(x: np.ndarray, window: int, floor: float, cap: float) -> np.ndarray:
    """Compute a causal rolling robust scale without pandas dependencies."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.empty(n, dtype=float)
    half = max(2, window // 2)
    # A strided implementation is unnecessary for the minute cadence sizes
    # used here; this is intentionally simple and auditable.
    for i in range(n):
        s = max(0, i - half + 1)
        w = x[s : i + 1]
        w = w[np.isfinite(w)]
        if w.size < 10:
            out[i] = floor
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 1e-9:
            sigma = np.std(w)
        out[i] = np.clip(float(sigma), floor, cap)
    return out


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


def detect_adaptive(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    config: AdaptiveConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect sustained local disturbances using adaptive robust thresholds.

    Important design property: thresholds are based only on the local
    residual. Global indices are deliberately excluded so this remains a
    genuine local detector and can operate when Kp/Dst are unavailable.
    """
    cfg = config or AdaptiveConfig()
    r = np.asarray(residual, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty(0, dtype=object), {"config": asdict(cfg)}

    finite = np.isfinite(r)
    if not finite.all():
        # Missing samples cannot silently create an event. They are filled only
        # for the short gaps normally allowed by the upstream pipeline.
        good = np.flatnonzero(finite)
        if len(good) < 2:
            return np.full(n, "quiet", dtype=object), {"config": asdict(cfg), "noise_sigma_nt": None}
        r = np.interp(np.arange(n), good, r[good])

    global_floor = _quiet_floor(r, cfg.quiet_scale_quantile)
    scale_window = max(11, int(round(cfg.rolling_scale_minutes * 60.0 / cadence_s)))
    scale = _rolling_mad_scale(
        r,
        scale_window,
        global_floor,
        global_floor * max(cfg.max_scale_multiplier, 1.0),
    )

    centered = r - np.median(r)
    amplitude_z = np.abs(centered) / scale

    diff = np.diff(r, prepend=r[0]) / max(cadence_s, 1.0)
    derivative_floor = _quiet_floor(diff, cfg.quiet_scale_quantile)
    derivative_window = max(11, int(round(60.0 * 60.0 / cadence_s)))
    derivative_scale = _rolling_mad_scale(
        diff,
        derivative_window,
        derivative_floor,
        derivative_floor * max(cfg.max_scale_multiplier, 1.0),
    )
    derivative_z = np.abs(diff) / derivative_scale

    onset_signal = (amplitude_z >= cfg.onset_sigma) | (
        derivative_z >= cfg.derivative_onset_sigma
    )
    clear_signal = (amplitude_z <= cfg.clear_sigma) & (
        derivative_z <= cfg.derivative_clear_sigma
    )
    derivative_hold_n = max(1, int(round(cfg.derivative_hold_minutes * 60.0 / cadence_s)))
    derivative_trigger = np.zeros(n, dtype=bool)
    for s, e in _runs(derivative_z >= cfg.derivative_onset_sigma):
        if e - s >= derivative_hold_n:
            derivative_trigger[s:e] = True
    onset_signal &= (amplitude_z >= cfg.onset_sigma) | derivative_trigger

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
    flags[clean & (amplitude_z >= cfg.onset_sigma)] = "unsettled"
    flags[clean & (amplitude_z >= cfg.active_sigma)] = "active"
    flags[clean & (amplitude_z >= cfg.storm_sigma)] = "minor_storm"
    flags[clean & (amplitude_z >= cfg.major_sigma)] = "major_storm"
    flags[clean & (amplitude_z >= cfg.severe_sigma)] = "severe_storm"

    # A short, high dB/dt pulse is reported as an anomaly rather than being
    # promoted to a sustained event.
    anomaly = (derivative_z >= cfg.derivative_onset_sigma * 1.8) & ~clean
    flags[anomaly] = "anomaly"

    event_runs = _runs(clean)
    diagnostics: Dict[str, Any] = {
        "config": asdict(cfg),
        "noise_sigma_nt": float(global_floor),
        "derivative_sigma_nt_per_s": float(derivative_floor),
        "rolling_scale_min_nt": float(np.min(scale)),
        "rolling_scale_median_nt": float(np.median(scale)),
        "rolling_scale_max_nt": float(np.max(scale)),
        "onset_threshold_nt_median": float(cfg.onset_sigma * np.median(scale)),
        "active_threshold_nt_median": float(cfg.active_sigma * np.median(scale)),
        "storm_threshold_nt_median": float(cfg.storm_sigma * np.median(scale)),
        "major_threshold_nt_median": float(cfg.major_sigma * np.median(scale)),
        "severe_threshold_nt_median": float(cfg.severe_sigma * np.median(scale)),
        "flagged_fraction": float(np.mean(clean)),
        "event_count": len(event_runs),
        "event_durations_minutes": [round((e - s) * cadence_s / 60.0, 2) for s, e in event_runs],
        "max_amplitude_z": float(np.nanmax(amplitude_z)),
        "max_derivative_z": float(np.nanmax(derivative_z)),
    }
    return flags, diagnostics
