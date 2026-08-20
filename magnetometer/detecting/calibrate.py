"""Canonical production calibrator entry point with strict provenance isolation."""
from dataclasses import asdict
from pathlib import Path
from .. import causal_baseline
from .. import production_grade_validation as pg

pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v7"

from .. import calibrate_detector as _impl  # noqa: E402

# The previous search space admitted overly permissive evidence combinations
# that produced pathological false alarms. Keep the optimizer broad enough to
# recover recall, but require materially stronger multi-timescale confirmation.
_impl.PARAMETER_GRID.update({
    "active_nt": (20.0, 25.0, 30.0, 35.0, 40.0, 50.0),
    "storm_nt": (50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0),
    "active_fast_ratio": (1.15, 1.25, 1.40, 1.60, 1.80),
    "active_medium_slow_ratio": (0.65, 0.70, 0.80, 0.90),
    "active_medium_upper_ratio": (0.55, 0.65, 0.75, 0.90),
    "active_peak_ratio": (1.75, 2.00, 2.25, 2.50, 3.00),
    "active_peak_medium_ratio": (0.50, 0.60, 0.75, 0.90),
    "storm_fast_ratio": (1.50, 1.75, 2.00, 2.25, 2.50),
    "storm_fast_medium_ratio": (0.75, 0.85, 0.95, 1.00),
    "storm_upper_ratio": (1.10, 1.20, 1.35, 1.50),
    "storm_upper_medium_ratio": (0.80, 0.90, 1.00),
    "storm_medium_ratio": (0.85, 0.95, 1.00, 1.10),
    "storm_peak_ratio": (1.75, 2.00, 2.25, 2.50, 3.00),
    "storm_peak_medium_ratio": (0.50, 0.65, 0.80, 0.95),
    "active_on_minutes": (3.0, 5.0, 8.0, 12.0),
    "active_off_minutes": (60.0, 90.0, 120.0, 180.0),
    "storm_on_minutes": (8.0, 10.0, 15.0, 20.0),
    "storm_off_minutes": (180.0, 240.0, 300.0, 360.0),
})

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
