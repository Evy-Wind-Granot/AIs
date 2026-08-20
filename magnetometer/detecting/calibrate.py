"""Canonical production calibrator entry point.

This module keeps calibration and live inference on exactly the same detector
implementation.  It also narrows the search space to profiles that satisfy
basic safety invariants before expensive scoring.
"""
from dataclasses import asdict
from pathlib import Path

from .. import causal_baseline
from .. import production_grade_validation as pg
from .. import detector_core as live_detector

pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v9"

from .. import calibrate_detector as _impl  # noqa: E402

# Search only profiles that are capable of coherent, low-FAR detection.  The
# live detector itself remains authoritative; the calibration predictor below
# delegates directly to it so calibration cannot optimize a different model.
_impl.PARAMETER_GRID.update({
    "active_nt": (20.0, 25.0, 30.0, 35.0, 40.0, 50.0),
    "storm_nt": (50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0),
    "active_fast_ratio": (1.25, 1.40, 1.60, 1.80, 2.00),
    "active_medium_slow_ratio": (0.60, 0.70, 0.80, 0.90),
    "active_medium_upper_ratio": (0.55, 0.65, 0.75, 0.90),
    "active_peak_ratio": (1.75, 2.00, 2.25, 2.50, 3.00),
    "active_peak_medium_ratio": (0.50, 0.60, 0.75, 0.90),
    "storm_fast_ratio": (1.75, 2.00, 2.25, 2.50),
    "storm_fast_medium_ratio": (0.70, 0.80, 0.90, 1.00),
    "storm_upper_ratio": (1.15, 1.25, 1.35, 1.50),
    "storm_upper_medium_ratio": (0.85, 0.95, 1.00),
    "storm_medium_ratio": (0.85, 0.95, 1.00, 1.10),
    "storm_peak_ratio": (1.75, 2.00, 2.25, 2.50, 3.00),
    "storm_peak_medium_ratio": (0.55, 0.65, 0.80, 0.95),
    "active_on_minutes": (3.0, 5.0, 8.0, 12.0),
    "active_off_minutes": (30.0, 45.0, 60.0, 90.0),
    "storm_on_minutes": (8.0, 10.0, 15.0, 20.0),
    "storm_off_minutes": (120.0, 180.0, 240.0, 300.0),
})


def _shared_predict(self, profile):
    """Use the exact live detector for calibration scoring."""
    active, storm, _major, _severe, _diagnostics = live_detector.detect_activity_masks(
        self.residual,
        cadence_s=self.cadence_s,
        profile=profile,
        include_anomaly=False,
    )
    return active, storm


_impl.PreparedCase.predict = _shared_predict

# Run multiple bounded coordinate-descent passes until the profile is stable.
_original_coordinate_descent = _impl._coordinate_descent


def _converged_coordinate_descent(cases, base, max_passes=5):
    profile = base
    for _ in range(max_passes):
        candidate = _original_coordinate_descent(cases, profile)
        if asdict(candidate) == asdict(profile):
            break
        profile = candidate
    return profile


_impl._coordinate_descent = _converged_coordinate_descent
main = _impl.main

if __name__ == "__main__":
    main()
