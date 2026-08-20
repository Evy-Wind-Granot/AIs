import numpy as np

from magnetometer.detecting.detector_core import DetectorProfile, detect_activity_masks


def test_detector_is_strictly_causal():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 2.0, 1000)
    profile = DetectorProfile(active_nt=25.0, storm_nt=70.0, active_on_minutes=5.0, active_off_minutes=30.0)
    before = detect_activity_masks(x, cadence_s=60.0, profile=profile, include_anomaly=False)
    changed = x.copy()
    changed[700:] += 500.0
    after = detect_activity_masks(changed, cadence_s=60.0, profile=profile, include_anomaly=False)
    for lhs, rhs in zip(before[:4], after[:4]):
        np.testing.assert_array_equal(lhs[:700], rhs[:700])


def test_invalid_samples_reset_state():
    x = np.full(500, 2.0)
    x[250:300] = 80.0
    x[300] = np.nan
    profile = DetectorProfile(active_nt=20.0, storm_nt=60.0, active_on_minutes=2.0, active_off_minutes=10.0, storm_on_minutes=2.0, storm_off_minutes=10.0)
    active, storm, major, severe, diagnostics = detect_activity_masks(x, cadence_s=60.0, profile=profile, include_anomaly=False)
    assert not active[300]
    assert not storm[300]
    assert not major[300]
    assert not severe[300]
    assert diagnostics["invalid_input"][300]


def test_single_short_spike_does_not_create_active_event():
    x = np.zeros(1000, dtype=float)
    x[500] = 500.0
    profile = DetectorProfile(active_nt=20.0, storm_nt=70.0, active_on_minutes=5.0, active_off_minutes=30.0, peak_window_minutes=5.0)
    active, storm, *_ = detect_activity_masks(x, cadence_s=60.0, profile=profile, include_anomaly=False)
    assert not active[500]
    assert not storm[500]
