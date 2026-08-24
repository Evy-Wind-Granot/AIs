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

    # 20-minute gradual event: deliberately below the 6-sigma pointwise
    # threshold at the median adaptive scale, but large enough to be a
    # sustained disturbance.
    start, end = 4 * 60, 24 * 60
    residual[start : start + 20] += np.linspace(0.0, 18.0, 20)
    residual[start + 20 : end - 20] += 18.0
    residual[end - 20 : end] += np.linspace(18.0, 0.0, 20)

    # Separate impulsive event to ensure the fast path remains operational.
    impulse_start = 16 * 60
    residual[impulse_start : impulse_start + 12] += 90.0

    flags, diag = detect_adaptive(residual, cadence_s)
    events = event_mask_from_flags(flags)
    runs = _runs(events)

    assert diag["event_count"] == len(runs), diag
    assert runs, diag
    assert all(
        duration >= diag["config"]["min_event_minutes"]
        for duration in diag["event_durations_minutes"]
    ), diag
    assert diag["onset_path_counts"]["sustained"] > 0, diag

    # The detector must never emit contradictory event/anomaly labels.
    assert not np.any(events & (flags == "anomaly"))

    # A normal noisy baseline must not be classified wholesale as an event.
    baseline = rng.normal(0.0, 2.0, n)
    baseline_flags, baseline_diag = detect_adaptive(baseline, cadence_s)
    baseline_events = event_mask_from_flags(baseline_flags)
    assert baseline_diag["flagged_fraction"] < 0.10, baseline_diag
    assert np.sum(baseline_events) < n * 0.10, baseline_diag

    print("Adaptive detector hardening tests: PASS")
    print(f"  detector events       : {diag['event_count']}")
    print(f"  sustained onset marks : {diag['onset_path_counts']['sustained']}")
    print(f"  baseline flagged frac : {baseline_diag['flagged_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
