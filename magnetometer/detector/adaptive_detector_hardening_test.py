#!/usr/bin/env python3
"""Regression tests for production invariants of the adaptive detector."""
from __future__ import annotations

import numpy as np

from adaptive_detector import detect_adaptive, event_mask_from_flags


def _runs(mask: np.ndarray):
    x = np.asarray(mask, dtype=bool)
    if not len(x):
        return []
    p = np.concatenate(([False], x, [False]))
    starts = np.flatnonzero(~p[:-1] & p[1:])
    ends = np.flatnonzero(p[:-1] & ~p[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def main() -> int:
    rng = np.random.default_rng(7)
    cadence_s = 60.0
    n = 24 * 60
    residual = rng.normal(0.0, 2.0, n)

    # Gradual event: deliberately below the pointwise onset threshold at the
    # nominal scale, but large enough to exercise the sustained path.
    start, end = 4 * 60, 10 * 60
    residual[start : start + 20] += np.linspace(0.0, 18.0, 20)
    residual[start + 20 : end - 20] += 18.0
    residual[end - 20 : end] += np.linspace(18.0, 0.0, 20)

    # Separate impulsive event well after a quiet gap. These must remain
    # separate detector events rather than being merged into one giant run.
    impulse_start = 15 * 60
    residual[impulse_start : impulse_start + 12] += 90.0

    flags, diag = detect_adaptive(residual, cadence_s)
    events = event_mask_from_flags(flags)
    runs = _runs(events)

    assert diag["event_count"] == len(runs), diag
    assert len(runs) >= 2, (diag, runs)
    assert all(
        duration >= diag["config"]["min_event_minutes"]
        for duration in diag["event_durations_minutes"]
    ), diag
    assert diag["onset_path_counts"]["sustained"] > 0, diag
    assert diag["onset_path_counts"]["pointwise"] > 0, diag

    # The detector must never emit contradictory event/anomaly labels.
    assert not np.any(events & (flags == "anomaly"))

    # Normal noisy baseline must not be classified wholesale as an event.
    baseline = rng.normal(0.0, 2.0, n)
    baseline_flags, baseline_diag = detect_adaptive(baseline, cadence_s)
    baseline_events = event_mask_from_flags(baseline_flags)
    assert baseline_diag["flagged_fraction"] < 0.10, baseline_diag
    assert np.sum(baseline_events) < n * 0.10, baseline_diag

    # A slow baseline shift should be treated as local baseline motion rather
    # than a sustained disturbance. This guards against global-median leakage.
    drift = rng.normal(0.0, 1.5, n)
    drift += np.linspace(0.0, 30.0, n)
    drift_flags, drift_diag = detect_adaptive(drift, cadence_s)
    drift_events = event_mask_from_flags(drift_flags)
    assert drift_diag["flagged_fraction"] < 0.10, drift_diag
    assert np.sum(drift_events) < n * 0.10, drift_diag

    # Clearing no longer depends on derivative noise, so a quiet local signal
    # must eventually re-arm the detector even when derivative jitter is high.
    quiet_then_jitter = np.zeros(n, dtype=float)
    quiet_then_jitter[5 * 60 : 25 * 60] = 20.0
    jitter = rng.normal(0.0, 5.0, n)
    quiet_then_jitter += jitter
    jitter_flags, jitter_diag = detect_adaptive(quiet_then_jitter, cadence_s)
    jitter_events = _runs(event_mask_from_flags(jitter_flags))
    assert jitter_events, jitter_diag
    assert all(d >= jitter_diag["config"]["min_event_minutes"] for d in jitter_diag["event_durations_minutes"])

    print("Adaptive detector hardening tests: PASS")
    print(f"  detector events       : {diag['event_count']}")
    print(f"  sustained onset marks : {diag['onset_path_counts']['sustained']}")
    print(f"  baseline flagged frac : {baseline_diag['flagged_fraction']:.4f}")
    print(f"  drift flagged frac    : {drift_diag['flagged_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
