"""Station-local event benchmark.

This benchmark deliberately does not use Kp or Dst as labels.  It constructs a
local reference event series from independent physical signatures in the
magnetometer residual: sustained amplitude, first-difference rate, and
persistence.  The reference is frozen before detector evaluation and is useful
for measuring local sensitivity, false events, latency, and event duration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class LocalReferenceConfig:
    amplitude_sigma: float = 6.0
    derivative_sigma: float = 6.0
    persistence_minutes: float = 10.0
    merge_gap_minutes: float = 20.0
    min_event_minutes: float = 10.0


def _sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    return float(max(s if np.isfinite(s) else 0.0, np.std(x), 1e-6))


def _local_noise(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    med = np.median(x)
    a = np.abs(x - med)
    q = np.quantile(a, 0.50)
    return _sigma(x[a <= q])


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    p = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
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


def build_local_reference(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    config: LocalReferenceConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a local binary event reference and auditable diagnostics."""
    cfg = config or LocalReferenceConfig()
    r = np.asarray(residual, dtype=float)
    sigma = _local_noise(r)
    d = np.diff(r, prepend=r[0])
    dsigma = _sigma(d)
    amp = np.abs(r) / sigma
    rate = np.abs(d) / dsigma

    candidate = (amp >= cfg.amplitude_sigma) & (rate >= cfg.derivative_sigma / 2.0)
    persistence = max(1, int(round(cfg.persistence_minutes * 60 / cadence_s)))
    min_n = max(1, int(round(cfg.min_event_minutes * 60 / cadence_s)))
    gap_n = max(0, int(round(cfg.merge_gap_minutes * 60 / cadence_s)))

    persistent = np.zeros(len(r), dtype=bool)
    for s, e in _runs(candidate):
        if e - s >= persistence:
            persistent[s:e] = True

    events = _merge(_runs(persistent), gap_n)
    reference = np.zeros(len(r), dtype=bool)
    kept = []
    for s, e in events:
        if e - s >= min_n:
            reference[s:e] = True
            kept.append((s, e))

    # Severity is local and continuous; this is not a Kp-derived class.
    severity = np.zeros(len(r), dtype=np.int8)
    severity[reference & (amp >= 6)] = 1
    severity[reference & (amp >= 12)] = 2
    severity[reference & (amp >= 20)] = 3
    severity[reference & (amp >= 40)] = 4

    return reference, {
        "config": asdict(cfg),
        "noise_sigma_nt": float(sigma),
        "derivative_sigma_nt_per_sample": float(dsigma),
        "event_count": len(kept),
        "event_durations_minutes": [round((e - s) * cadence_s / 60, 2) for s, e in kept],
        "severity_counts": {str(i): int(np.sum(severity == i)) for i in range(5)},
    }


def compare_events(
    reference: np.ndarray,
    prediction: np.ndarray,
    cadence_s: float = 60.0,
    tolerance_minutes: float = 30.0,
) -> Dict[str, Any]:
    """Event-level precision/recall with one-to-one matching and latency."""
    rr = _runs(reference)
    pp = _runs(prediction)
    tol = int(round(tolerance_minutes * 60 / cadence_s))
    used = set(); latencies = []; matches = 0
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
            used.add(j); matches += 1
            latencies.append(max(0, ps - rs) * cadence_s / 60.0)
    precision = matches / len(pp) if pp else 0.0
    recall = matches / len(rr) if rr else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_events": len(rr), "predicted_events": len(pp),
        "matched_events": matches, "missed_events": len(rr) - matches,
        "false_positive_events": len(pp) - matches,
        "precision": precision, "recall": recall, "f1": f1,
        "false_events_per_day": (len(pp) - matches) / max(len(reference) * cadence_s / 86400, 1e-9),
        "mean_latency_minutes": float(np.mean(latencies)) if latencies else None,
        "median_latency_minutes": float(np.median(latencies)) if latencies else None,
        "max_latency_minutes": float(np.max(latencies)) if latencies else None,
    }
