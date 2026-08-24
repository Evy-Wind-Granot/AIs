"""Station-local event reference independent of Kp/Dst and detector settings.

This is a local *reference*, not an absolute physical ground truth: the only
observations used are the station's own magnetic residual. It intentionally
uses multi-scale disturbance energy calibrated against a robust quiet subset
rather than the adaptive detector's pointwise sigma thresholds. That makes it
useful for detector development without pretending that global Kp is local
truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LocalReferenceConfig:
    short_minutes: float = 15.0
    long_minutes: float = 180.0
    short_percentile: float = 99.0
    long_percentile: float = 99.5
    derivative_percentile: float = 99.0
    persistence_minutes: float = 10.0
    merge_gap_minutes: float = 30.0
    min_event_minutes: float = 10.0
    quiet_baseline_quantile: float = 0.40


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    x = np.asarray(mask, dtype=bool)
    if not len(x):
        return []
    p = np.concatenate(([False], x, [False]))
    s = np.flatnonzero(~p[:-1] & p[1:])
    e = np.flatnonzero(p[:-1] & ~p[1:])
    return list(zip(s.tolist(), e.tolist()))


def _merge(runs: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in runs:
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-9:
        s = np.std(x)
    return max(float(s), 1e-6)


def _quiet_scale(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) < 20:
        return 1.0
    med = np.median(finite)
    a = np.abs(finite - med)
    cutoff = np.quantile(a, np.clip(q, 0.1, 0.8))
    return _robust_sigma(finite[a <= cutoff])


def _robust_feature_threshold(values: np.ndarray, percentile: float) -> float:
    """Estimate a quiet-feature upper threshold without letting events set it.

    The previous implementation took a percentile over the entire benchmark
    period. Because that same period contains the events we want to detect,
    the threshold could sit at the event's own maximum, making a 10-minute
    persistence requirement mathematically impossible. We now use a robust
    location/scale estimate of the quiet subset and map the configured
    percentile to a Gaussian-equivalent robust z threshold.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return float("inf")
    med = float(np.median(v))
    sigma = _robust_sigma(v)
    p = float(np.clip(percentile, 90.0, 99.9)) / 100.0
    z = NormalDist().inv_cdf(p)
    return float(med + z * sigma)


def build_local_reference(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    config: LocalReferenceConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a frozen local event reference from multi-scale disturbance energy."""
    cfg = config or LocalReferenceConfig()
    r = np.asarray(residual, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty(0, dtype=bool), {"config": asdict(cfg), "event_count": 0}

    finite = np.isfinite(r)
    if not finite.all():
        good = np.flatnonzero(finite)
        if len(good) < 2:
            return np.zeros(n, dtype=bool), {"config": asdict(cfg), "event_count": 0}
        r = np.interp(np.arange(n), good, r[good])

    center = float(np.median(r))
    centered = r - center
    quiet_sigma = _quiet_scale(centered, cfg.quiet_baseline_quantile)

    # A point is part of the quiet calibration subset when its instantaneous
    # residual amplitude is in the lower quiet-baseline quantile. Feature
    # thresholds are then estimated robustly from that subset rather than from
    # the full period, preventing the event itself from defining its threshold.
    amplitude = np.abs(centered)
    quiet_cutoff = float(np.quantile(amplitude[np.isfinite(amplitude)], np.clip(cfg.quiet_baseline_quantile, 0.1, 0.8)))
    quiet_mask = np.isfinite(amplitude) & (amplitude <= quiet_cutoff)

    short_n = max(3, int(round(cfg.short_minutes * 60.0 / cadence_s)))
    long_n = max(short_n + 1, int(round(cfg.long_minutes * 60.0 / cadence_s)))

    s = pd.Series(centered)
    short_min = max(3, short_n // 3)
    long_min = max(short_n, long_n // 3)
    short_rms = np.sqrt(s.pow(2).rolling(short_n, min_periods=short_min).mean())
    short_range = s.rolling(short_n, min_periods=short_min).max() - s.rolling(short_n, min_periods=short_min).min()
    long_range = s.rolling(long_n, min_periods=long_min).max() - s.rolling(long_n, min_periods=long_min).min()

    diff = np.diff(r, prepend=r[0]) / max(cadence_s, 1.0)
    d = pd.Series(np.abs(diff))
    derivative_rms = np.sqrt(d.pow(2).rolling(short_n, min_periods=short_min).mean())

    valid_short = short_rms.to_numpy(float)
    valid_range = short_range.to_numpy(float)
    valid_long = long_range.to_numpy(float)
    valid_deriv = derivative_rms.to_numpy(float)

    short_quiet = valid_short[quiet_mask & np.isfinite(valid_short)]
    range_quiet = valid_range[quiet_mask & np.isfinite(valid_range)]
    long_quiet = valid_long[quiet_mask & np.isfinite(valid_long)]
    deriv_quiet = valid_deriv[quiet_mask & np.isfinite(valid_deriv)]

    short_thr = max(_robust_feature_threshold(short_quiet, cfg.short_percentile), quiet_sigma * 2.5)
    range_thr = max(_robust_feature_threshold(range_quiet, cfg.short_percentile), quiet_sigma * 5.0)
    long_thr = max(_robust_feature_threshold(long_quiet, cfg.long_percentile), quiet_sigma * 10.0)
    deriv_thr = max(_robust_feature_threshold(deriv_quiet, cfg.derivative_percentile), _robust_sigma(diff) * 3.0)

    # Two independent local signatures are required. This remains deliberately
    # harder to satisfy than the adaptive detector's OR-based onset rule.
    candidate = (
        ((valid_short >= short_thr) & (valid_range >= range_thr))
        | ((valid_long >= long_thr) & (valid_deriv >= deriv_thr))
    )
    candidate &= np.isfinite(valid_short) & np.isfinite(valid_long)

    persistence_n = max(1, int(round(cfg.persistence_minutes * 60 / cadence_s)))
    min_n = max(1, int(round(cfg.min_event_minutes * 60 / cadence_s)))
    gap_n = max(0, int(round(cfg.merge_gap_minutes * 60 / cadence_s)))

    persistent = np.zeros(n, dtype=bool)
    for start, end in _runs(candidate):
        if end - start >= persistence_n:
            persistent[start:end] = True

    events = _merge(_runs(persistent), gap_n)
    reference = np.zeros(n, dtype=bool)
    kept: List[Tuple[int, int]] = []
    for start, end in events:
        if end - start >= min_n:
            reference[start:end] = True
            kept.append((start, end))

    severity = np.zeros(n, dtype=np.int8)
    long_z = valid_long / max(quiet_sigma, 1e-6)
    severity[reference & (long_z >= 12)] = 1
    severity[reference & (long_z >= 20)] = 2
    severity[reference & (long_z >= 35)] = 3
    severity[reference & (long_z >= 60)] = 4

    return reference, {
        "config": asdict(cfg),
        "reference_definition": "local multi-scale disturbance energy; robust quiet-subset calibration; no Kp/Dst",
        "quiet_sigma_nt": float(quiet_sigma),
        "quiet_sample_fraction": float(np.mean(quiet_mask)),
        "short_rms_threshold_nt": float(short_thr),
        "short_range_threshold_nt": float(range_thr),
        "long_range_threshold_nt": float(long_thr),
        "derivative_rms_threshold_nt_per_s": float(deriv_thr),
        "threshold_calibration": "robust quiet-subset location/scale with configured percentile as z target",
        "event_count": len(kept),
        "event_durations_minutes": [round((e - s) * cadence_s / 60.0, 2) for s, e in kept],
        "severity_counts": {str(i): int(np.sum(severity == i)) for i in range(5)},
    }


def compare_events(
    reference: np.ndarray,
    prediction: np.ndarray,
    cadence_s: float = 60.0,
    tolerance_minutes: float = 30.0,
) -> Dict[str, Any]:
    """Event-level one-to-one matching with latency and false-event rate."""
    rr = _runs(reference)
    pp = _runs(prediction)
    tol = int(round(tolerance_minutes * 60 / cadence_s))
    used = set()
    latencies: List[float] = []
    matches = 0
    for rs, re in rr:
        candidates = []
        for j, (ps, pe) in enumerate(pp):
            if j in used:
                continue
            if ps <= re + tol and pe >= rs - tol:
                overlap = max(0, min(re, pe) - max(rs, ps))
                distance = 0 if overlap else abs(ps - re)
                candidates.append((distance, j, ps))
        if candidates:
            _, j, ps = min(candidates)
            used.add(j)
            matches += 1
            latencies.append(max(0, ps - rs) * cadence_s / 60.0)
    precision = matches / len(pp) if pp else 0.0
    recall = matches / len(rr) if rr else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    days = max(len(reference) * cadence_s / 86400.0, 1e-9)
    return {
        "reference_events": len(rr),
        "predicted_events": len(pp),
        "matched_events": matches,
        "missed_events": len(rr) - matches,
        "false_positive_events": len(pp) - matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_events_per_day": (len(pp) - matches) / days,
        "mean_latency_minutes": float(np.mean(latencies)) if latencies else None,
        "median_latency_minutes": float(np.median(latencies)) if latencies else None,
        "max_latency_minutes": float(np.max(latencies)) if latencies else None,
    }
