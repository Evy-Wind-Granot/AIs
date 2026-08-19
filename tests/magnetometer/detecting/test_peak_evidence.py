"""Regression tests for causal short-peak detector evidence."""

import numpy as np

from magnetometer.detector_core import DetectorProfile, detect_activity_masks, flag_activity


def test_short_peak_evidence_requires_causal_history_and_persistence():
    residual = np.zeros(8 * 60, dtype=float)
    residual[240:270] = 90.0
    profile = DetectorProfile(
        storm_nt=35.0,
        storm_peak_ratio=1.5,
        storm_peak_medium_ratio=0.15,
        storm_on_minutes=2.0,
        storm_off_minutes=30.0,
        peak_window_minutes=5.0,
    )
    _, storm, _, _, diagnostics = detect_activity_masks(residual, cadence_s=60.0, profile=profile)
    ready = diagnostics["history_ready"]
    first_ready = int(np.flatnonzero(ready)[0])
    assert not np.any(storm[:first_ready])
    assert diagnostics["storm_strong_peak_evidence"][250]
    assert storm[250:270].mean() > 0.5


def test_invalid_gap_resets_peak_evidence_and_hysteresis():
    residual = np.full(12 * 60, 60.0, dtype=float)
    residual[5 * 60:6 * 60] = np.nan
    profile = DetectorProfile(storm_on_minutes=1.0, storm_off_minutes=10.0, peak_window_minutes=5.0)
    _, storm, _, _, diagnostics = detect_activity_masks(residual, cadence_s=60.0, profile=profile)
    assert not np.any(storm[5 * 60:6 * 60])
    assert not storm[6 * 60]
    assert diagnostics["invalid_input"][5 * 60:6 * 60].all()


def test_profile_roundtrip_with_peak_parameters():
    profile = DetectorProfile(
        active_peak_ratio=1.6,
        active_peak_medium_ratio=0.2,
        storm_peak_ratio=1.5,
        storm_peak_medium_ratio=0.2,
        peak_window_minutes=7.0,
    )
    restored = DetectorProfile.from_dict(profile.__dict__)
    assert restored == profile


def test_isolated_large_spike_still_does_not_become_storm():
    residual = np.zeros(8 * 60, dtype=float)
    residual[240] = 250.0
    flags = flag_activity(residual, cadence_s=60.0)
    assert not np.any(np.isin(flags, ["minor_storm", "major_storm", "severe_storm"]))
