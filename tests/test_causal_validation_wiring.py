from pathlib import Path

from magnetometer import causal_baseline
from magnetometer import production_grade_validation as pg


def test_causal_baseline_has_explicit_production_implementation():
    assert callable(causal_baseline.compute_causal_qdc_baseline)
    assert causal_baseline.compute_causal_qdc_baseline.__name__ == "compute_causal_qdc_baseline"


def test_canonical_calibrator_installs_causal_baseline():
    import magnetometer.detecting.calibrate  # noqa: F401

    assert pg.pm.compute_qdc_baseline is causal_baseline.compute_causal_qdc_baseline
    assert str(pg.DEFAULT_CACHE_DIR).endswith("case_cache_causal_v3")


def test_causal_cache_namespace_is_distinct_from_legacy():
    legacy = Path(__file__).resolve().parents[1] / "magnetometer" / "data" / "case_cache"
    fresh = Path(__file__).resolve().parents[1] / "magnetometer" / "data" / "case_cache_causal_v3"
    assert fresh != legacy
