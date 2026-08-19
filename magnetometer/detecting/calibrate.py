"""Canonical production calibrator entry point.

This wrapper pins calibration to a new cache namespace because residual generation
is part of the certified computation and changed with the causal-baseline fix.
"""
from pathlib import Path

from .. import production_grade_validation as pg

pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v2"

from ..calibrate_detector import main  # noqa: E402


if __name__ == "__main__":
    main()
