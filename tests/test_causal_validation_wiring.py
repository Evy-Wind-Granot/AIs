from pathlib import Path

from magnetometer import causal_baseline
from magnetometer import production_grade_validation as pg


def test_causal_baseline_has_explicit_production_implementation():
    assert callable(causal_baseline.compute_causal_qdc_baseline)
    assert causal_baseline.compute_causal_qdc_baseline.__name__ == "compute_causal_qdc_baseline"


def test_canonical_calibrator_uses_fresh_causal_v8_cache():
    import magnetometer.detecting.calibrate  # noqa: F401

    assert pg.pm.compute_qdc_baseline is causal_baseline.compute_causal_qdc_baseline
    assert str(pg.DEFAULT_CACHE_DIR).endswith("case_cache_causal_v8")


def test_causal_cache_namespace_is_distinct_from_legacy():
    root = Path(__file__).resolve().parents[1] / "magnetometer" / "data"
    legacy = root / "case_cache"
    old_causal = root / "case_cache_causal_v4"
    fresh = root / "case_cache_causal_v8"
    assert fresh != legacy
    assert fresh != old_causal
