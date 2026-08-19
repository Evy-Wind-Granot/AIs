"""Canonical production calibrator entry point with strict provenance isolation."""
from dataclasses import asdict
from pathlib import Path
from .. import causal_baseline
from .. import production_grade_validation as pg

pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v6"

from .. import calibrate_detector as _impl  # noqa: E402
_original_coordinate_descent = _impl._coordinate_descent

def _converged_coordinate_descent(cases, base, max_passes=3):
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
