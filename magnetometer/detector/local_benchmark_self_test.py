#!/usr/bin/env python3
"""Deterministic self-test for the local benchmark and adaptive detector."""
from __future__ import annotations

import numpy as np

from adaptive_detector import detect_adaptive, event_mask_from_flags
from local_event_benchmark import build_local_reference, compare_events


def _runs(mask: np.ndarray):
    x = np.asarray(mask, dtype=bool)
    if not len(x):
        return []
    p = np.concatenate(([False], x, [False]))
    starts = np.flatnonzero(~p[:-1] & p[1:])
    ends = np.flatnonzero(p[:-1] & ~p[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def _overlap_exists(reference: np.ndarray, expected: np.ndarray) -> bool:
    for rs, re in _runs(reference):
        for es, ee in _runs(expected):
            if min(re, ee) > max(rs, es):
                return True
    return False


def main() -> int:
    rng = np.random.default_rng(42)
    cadence_s = 60.0
    n = 24 * 60
    residual = rng.normal(0.0, 2.0, n)
    expected = np.zeros(n, dtype=bool)

    # Two deliberately obvious local disturbances with different shapes.
    # The second is oscillatory so both short-window energy and range remain
    # elevated for much longer than the 10-minute persistence requirement.
    residual[7 * 60 : 9 * 60] += np.linspace(0.0, 180.0, 120)
    residual[9 * 60 : 11 * 60] += np.linspace(180.0, 0.0, 120)
    expected[7 * 60 : 11 * 60] = True

    t = np.arange(4 * 60)
    residual[17 * 60 : 21 * 60] += 160.0 * np.sin(2.0 * np.pi * t / 30.0)
    expected[17 * 60 : 21 * 60] = True

    reference, ref_diag = build_local_reference(residual, cadence_s)
    prediction, det_diag = detect_adaptive(residual, cadence_s)
    prediction_mask = event_mask_from_flags(prediction)
    metrics = compare_events(reference, prediction_mask, cadence_s)

    assert len(reference) == n
    assert ref_diag["event_count"] >= 1, ref_diag
    assert det_diag["event_count"] >= 1, det_diag
    assert metrics["reference_events"] == ref_diag["event_count"], metrics
    assert metrics["predicted_events"] == det_diag["event_count"], metrics
    assert metrics["reference_events"] >= 1
    assert metrics["predicted_events"] >= 1
    assert _overlap_exists(reference, expected), (ref_diag, "reference did not overlap injected disturbance")
    assert _overlap_exists(prediction_mask, expected), (det_diag, "detector did not overlap injected disturbance")

    # Anomaly is a transient diagnostic state, not an event.
    anomaly = prediction == "anomaly"
    assert not np.any(prediction_mask & anomaly)

    print("Local benchmark self-test: PASS")
    print(f"  reference events : {ref_diag['event_count']}")
    print(f"  detector events  : {det_diag['event_count']}")
    print(f"  anomaly samples  : {int(anomaly.sum())}")
    print(f"  event F1         : {metrics['f1']:.3f}")
    print(f"  event precision  : {metrics['precision']:.3f}")
    print(f"  event recall     : {metrics['recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
