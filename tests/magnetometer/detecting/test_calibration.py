import time

import numpy as np

from magnetometer.calibrate_detector import PreparedCase, _prepare_case, _hysteresis_mask_fast
from magnetometer.detector_core import DetectorProfile


def test_prepare_case_computes_profile_independent_features_once() -> None:
    n = 3 * 60 * 60 + 200
    residual = np.zeros(n, dtype=float)
    residual[200:500] = 60.0
    refs = {
        "known": np.ones(n, dtype=bool),
        "active": residual >= 15.0,
        "storm": residual >= 35.0,
    }
    data = {"residual": residual, "cadence_s": 60.0, "refs": refs}
    prepared = _prepare_case(data)
    assert isinstance(prepared, PreparedCase)
    assert prepared.n == n
    assert prepared.fast_5m.shape == (n,)
    assert prepared.medium_15m.shape == (n,)
    assert prepared.upper_30m.shape == (n,)
    assert prepared.slow_60m.shape == (n,)
    assert prepared.slow_3h.shape == (n,)


def test_fast_hysteresis_matches_expected_state_transition() -> None:
    on = np.array([False, True, True, True, False, False, False])
    off = np.array([True, False, False, False, True, True, True])
    result = _hysteresis_mask_fast(on, off, min_on=2, min_off=2)
    assert np.array_equal(result, np.array([False, False, True, True, True, True, False]))


def test_calibration_profile_evaluation_is_independent_of_profile_io() -> None:
    n = 1000
    residual = np.zeros(n, dtype=float)
    residual[300:600] = 60.0
    refs = {"known": np.ones(n, dtype=bool), "active": residual >= 15.0, "storm": residual >= 35.0}
    prepared = _prepare_case({"residual": residual, "cadence_s": 60.0, "refs": refs})
    profile = DetectorProfile()
    start = time.perf_counter()
    active, storm = prepared.predict(profile)
    elapsed = time.perf_counter() - start
    assert active.shape == (n,)
    assert storm.shape == (n,)
    assert elapsed < 1.0
