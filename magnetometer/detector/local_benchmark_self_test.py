#!/usr/bin/env python3
"""Deterministic self-test for the local benchmark and adaptive detector."""
from __future__ import annotations

import numpy as np

from adaptive_detector import detect_adaptive
from local_event_benchmark import build_local_reference, compare_events


def main() -> int:
    rng = np.random.default_rng(42)
    cadence_s = 60.0
    n = 24 * 60
    residual = rng.normal(0.0, 2.0, n)

    # Two synthetic local disturbances with different shapes.
    residual[7 * 60 : 9 * 60] += np.linspace(0.0, 45.0, 120)
    residual[9 * 60 : 11 * 60] += np.linspace(45.0, 0.0, 120)
    residual[17 * 60 : 21 * 60] += 80.0 * np.sin(np.linspace(0, np.pi, 240))

    reference, ref_diag = build_local_reference(residual, cadence_s)
    prediction, det_diag = detect_adaptive(residual, cadence_s)
    metrics = compare_events(reference, prediction != "quiet", cadence_s)

    assert len(reference) == n
    assert ref_diag["event_count"] >= 1, ref_diag
    assert det_diag["event_count"] >= 1, det_diag
    assert metrics["reference_events"] >= 1
    assert metrics["predicted_events"] >= 1

    print("Local benchmark self-test: PASS")
    print(f"  reference events : {ref_diag['event_count']}")
    print(f"  detector events  : {det_diag['event_count']}")
    print(f"  event F1         : {metrics['f1']:.3f}")
    print(f"  event precision  : {metrics['precision']:.3f}")
    print(f"  event recall     : {metrics['recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
