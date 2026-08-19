import numpy as np

from magnetometer.detector_core import DETECTOR_VERSION, DetectorProfile, detect_activity_masks


def test_detector_is_strictly_causal_with_respect_to_future_samples():
    rng = np.random.default_rng(7)
    prefix = rng.normal(0.0, 2.0, 5000)
    future = np.full(1000, 250.0)
    x1 = np.r_[prefix, np.zeros(1000)]
    x2 = np.r_[prefix, future]
    p = DetectorProfile()
    a1, s1, m1, v1, _ = detect_activity_masks(x1, profile=p, include_anomaly=False)
    a2, s2, m2, v2, _ = detect_activity_masks(x2, profile=p, include_anomaly=False)
    assert np.array_equal(a1[:5000], a2[:5000])
    assert np.array_equal(s1[:5000], s2[:5000])
    assert np.array_equal(m1[:5000], m2[:5000])
    assert np.array_equal(v1[:5000], v2[:5000])


def test_detector_profile_requires_certification_and_version_at_load_time(tmp_path):
    profile_path = tmp_path / "detector_profile.json"
    profile_path.write_text('{"status":"candidate","detector_version":"causal-disturbance-v2","profile":{}}')
    from magnetometer.detector_core import load_detector_profile
    try:
        load_detector_profile(profile_path)
    except RuntimeError as exc:
        assert "not certified" in str(exc)
    else:
        raise AssertionError("uncertified detector profile was loaded")


def test_detector_version_is_explicit():
    assert DETECTOR_VERSION == "causal-disturbance-v2"
