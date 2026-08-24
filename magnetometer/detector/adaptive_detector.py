"""Local adaptive event detector independent of Kp/Dst."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    onset_sigma: float = 5.0
    active_sigma: float = 8.0
    storm_sigma: float = 15.0
    major_sigma: float = 25.0
    severe_sigma: float = 50.0
    clear_sigma: float = 3.5
    derivative_sigma: float = 5.0
    onset_minutes: float = 5.0
    clear_minutes: float = 10.0
    min_event_minutes: float = 10.0
    merge_gap_minutes: float = 15.0


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = np.std(x)
    return float(max(sigma, 1e-6))


def _calibration_sigma(residual: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float)
    finite = r[np.isfinite(r)]
    if finite.size < 20:
        return 1.0
    med = np.median(finite)
    abs_r = np.abs(finite - med)
    quiet = finite[abs_r <= np.quantile(abs_r, 0.60)]
    return _robust_sigma(quiet)


def _runs(mask: np.ndarray) -> list[Tuple[int, int]]:
    x = np.asarray(mask, dtype=bool)
    if not x.size:
        return []
    p = np.concatenate(([False], x, [False]))
    starts = np.flatnonzero(~p[:-1] & p[1:])
    ends = np.flatnonzero(p[:-1] & ~p[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_runs(runs: list[Tuple[int, int]], max_gap: int) -> list[Tuple[int, int]]:
    out: list[Tuple[int, int]] = []
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
    """Detect sustained local disturbances using station-local robust scale.

    Kp/Dst are deliberately absent. The local noise scale is estimated from
    the lower 60 percent of the residual amplitude distribution so active
    periods cannot simply redefine the threshold upward.
    """
    cfg = config or AdaptiveConfig()
    r = np.asarray(residual, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty(0, dtype=object), {"config": asdict(cfg), "noise_sigma_nt": None}

    sigma = _calibration_sigma(r)
    diff = np.diff(r, prepend=r[0])
    diff_sigma = _robust_sigma(diff)
    amplitude = np.abs(r) / sigma
    rate = np.abs(diff) / max(diff_sigma, 1e-6)

    onset = (amplitude >= cfg.onset_sigma) | (rate >= cfg.derivative_sigma)
    onset_n = max(1, int(round(cfg.onset_minutes * 60.0 / cadence_s)))
    clear_n = max(1, int(round(cfg.clear_minutes * 60.0 / cadence_s)))
    min_n = max(1, int(round(cfg.min_event_minutes * 60.0 / cadence_s)))
    merge_n = max(0, int(round(cfg.merge_gap_minutes * 60.0 / cadence_s)))

    active = np.zeros(n, dtype=bool)
    above = below = 0
    state = False
    for i in range(n):
        if onset[i]:
            above += 1; below = 0
        elif amplitude[i] <= cfg.clear_sigma:
            below += 1; above = 0
        else:
            above = below = 0
        if not state and above >= onset_n:
            state = True
        elif state and below >= clear_n:
            state = False
        active[i] = state

    runs = _merge_runs(_runs(active), merge_n)
    clean = np.zeros(n, dtype=bool)
    for s, e in runs:
        if e - s >= min_n:
            clean[s:e] = True

    flags = np.full(n, "quiet", dtype=object)
    flags[clean & (amplitude >= cfg.onset_sigma)] = "unsettled"
    flags[clean & (amplitude >= cfg.active_sigma)] = "active"
    flags[clean & (amplitude >= cfg.storm_sigma)] = "minor_storm"
    flags[clean & (amplitude >= cfg.major_sigma)] = "major_storm"
    flags[clean & (amplitude >= cfg.severe_sigma)] = "severe_storm"
    anomaly = (rate >= max(cfg.derivative_sigma * 2.0, 10.0)) & ~clean
    flags[anomaly] = "anomaly"

    diagnostics = {
        "config": asdict(cfg),
        "noise_sigma_nt": float(sigma),
        "derivative_sigma_nt_per_sample": float(diff_sigma),
        "onset_threshold_nt": float(cfg.onset_sigma * sigma),
        "active_threshold_nt": float(cfg.active_sigma * sigma),
        "storm_threshold_nt": float(cfg.storm_sigma * sigma),
        "major_threshold_nt": float(cfg.major_sigma * sigma),
        "severe_threshold_nt": float(cfg.severe_sigma * sigma),
        "flagged_fraction": float(np.mean(clean)),
        "event_count": int(len(_runs(clean))),
    }
    return flags, diagnostics
