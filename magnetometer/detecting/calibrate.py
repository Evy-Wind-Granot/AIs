"""Canonical production calibrator entry point."""
from pathlib import Path

from .. import production_grade_validation as pg
from .. import calibrate_detector

pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v8"

main = calibrate_detector.main

if __name__ == "__main__":
    main()
