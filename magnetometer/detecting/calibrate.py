"""Canonical production calibrator entry point."""
from pathlib import Path

from .. import production_grade_validation as pg
from .. import calibrate_detector

# Residual generation is already implemented by performance_metrics through
# the shared strictly-causal baseline. Keep calibration on a fresh cache
# namespace whenever detector semantics change.
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v7"

main = calibrate_detector.main

if __name__ == "__main__":
    main()
