"""Canonical detector release-gate entry point."""
from pathlib import Path

from .. import causal_baseline
from .. import production_grade_validation as pg

# The release gate must score the same strictly-causal residual generation used
# by calibration and live inference. Use the same fresh cache generation as
# calibration so certification cannot consume older residual artifacts.
pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v5"

from ..production_release_gate_v4 import main  # noqa: E402

if __name__ == "__main__":
    main()
