import numpy as np

from magnetometer.calibrate_detector import PreparedCase, _evaluate, _search
from magnetometer.detector_core import DETECTOR_VERSION, DetectorProfile


def _case(n=5000):
    residual = np.zeros(n, dtype=float)
    residual[2200:2800] = 60.0
    refs = {
        "known": np.ones(n, dtype=bool),
        "active": residual >= 15.0,
        "storm": residual >= 35.0,
    }
    return PreparedCase(
        "TEST",
        type("Case", (), {})(),
        {"residual": residual, "cadence_s": 60.0, "refs": refs},
    )


def test_prepared_case_uses_public_detector_path():
    prepared = _case()
    result = _evaluate([prepared], DetectorProfile())
    assert "active" in result
    assert "storm" in result
    assert result["active"]["tp"] >= 0
    assert result["storm"]["tp"] >= 0


def test_search_keeps_threshold_ordering_and_versioned_profile():
    prepared = _case()
    profile = _search([prepared], DetectorProfile())
    profile.validate()
    assert profile.active_nt < profile.storm_nt
    assert DETECTOR_VERSION == "causal-disturbance-v2.1"


def test_detector_profile_search_does_not_create_sub_threshold_storms():
    prepared = _case()
    profile = _search([prepared], DetectorProfile())
    assert profile.storm_nt >= 35.0
