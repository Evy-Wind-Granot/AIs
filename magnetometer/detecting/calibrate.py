"""Canonical production calibrator entry point."""
from pathlib import Path

from .. import causal_baseline
from .. import production_grade_validation as pg
from .. import calibrate_detector

# Calibration and release validation must use the identical strictly causal
# baseline implementation. Install it at module import time so every caller
# of the canonical calibrator receives the production baseline, including
# tests and programmatic integrations that invoke ``main`` directly.
pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v8"

main = calibrate_detector.main

if __name__ == "__main__":
    main()
