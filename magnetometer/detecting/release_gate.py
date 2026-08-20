"""Canonical detector release-gate entry point."""
from pathlib import Path

from .. import causal_baseline
from .. import production_grade_validation as pg

# Release scoring must use the same strictly-causal residual generation as
# calibration/live inference, with a fresh cache namespace for this detector
# generation.
pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v7"

from ..production_release_gate_v5 import main  # noqa: E402

if __name__ == "__main__":
    main()
