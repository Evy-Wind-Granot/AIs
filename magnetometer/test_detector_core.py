#!/usr/bin/env python3
"""Regression tests for the deterministic production detector."""

import json

import numpy as np

from detector_core import DetectorProfile, detect_activity_masks, flag_activity


STORM_LEVELS = ["minor_storm", "major_storm", "severe_storm"]


def test_isolated_spike_does_not_create_storm():
    residual = np.zeros(6 * 60, dtype=float); residual[180] = 250.0
    flags = flag_activity(residual, cadence_s=60.0)
    assert not np.any(np.isin(flags, STORM_LEVELS))


def test_sustained_storm_is_detected_after_causal_warmup():
    residual = np.zeros(6 * 60, dtype=float); residual[60:300] = 60.0
    active, storm, major, severe, diagnostics = detect_activity_masks(residual, cadence_s=60.0, active_threshold=15.0, storm_threshold=35.0)
    ready = diagnostics["history_ready"]; first_ready = int(np.flatnonzero(ready)[0])
    assert not np.any(storm[:first_ready]); assert storm[195:240].mean() > 0.8; assert active[195:240].mean() > 0.8
    assert not np.any(major); assert not np.any(severe)


def test_strong_short_excursion_can_confirm_storm():
    residual = np.zeros(8 * 60, dtype=float); residual[240:270] = 90.0; residual[270:330] = 45.0
    storm = np.isin(flag_activity(residual, cadence_s=60.0), STORM_LEVELS)
    assert storm[255:330].mean() > 0.5


def test_short_dip_does_not_split_storm_event():
    residual = np.zeros(10 * 60, dtype=float); residual[60:540] = 60.0; residual[300:310] = 5.0
    storm = np.isin(flag_activity(residual, cadence_s=60.0), STORM_LEVELS)
    assert storm[195:420].mean() > 0.9


def test_long_moderate_disturbance_remains_active_not_storm():
    residual = np.zeros(8 * 60, dtype=float); residual[60:420] = 27.0; residual[210:225] = 12.0
    flags = flag_activity(residual, cadence_s=60.0, active_threshold=15.0, storm_threshold=35.0)
    assert np.isin(flags, STORM_LEVELS).mean() == 0.0; assert (flags == "active")[195:360].mean() > 0.75


def test_nan_samples_are_safe():
    residual = np.full(120, np.nan, dtype=float); residual[30:90] = 20.0; flags = flag_activity(residual, cadence_s=60.0)
    assert np.all(flags[np.isnan(residual)] == "quiet"); assert np.all(np.isfinite(residual) | (flags == "quiet"))


def test_detector_is_strictly_causal():
    prefix = np.zeros(5 * 3600, dtype=float); prefix[4 * 3600:] = 18.0; future = np.full(3 * 3600, 250.0, dtype=float)
    short = flag_activity(prefix, cadence_s=60.0); long = flag_activity(np.concatenate([prefix, future]), cadence_s=60.0)
    assert np.array_equal(short, long[: len(prefix)])


def test_future_gap_cannot_retroactively_create_a_storm():
    prefix = np.zeros(8 * 60, dtype=float); prefix[3 * 60:5 * 60] = 60.0; future = np.zeros(4 * 60, dtype=float); future[-2 * 60:] = 60.0
    short = flag_activity(prefix, cadence_s=60.0); long = flag_activity(np.concatenate([prefix, future]), cadence_s=60.0)
    assert np.array_equal(short, long[: len(prefix)])


def test_startup_requires_history():
    residual = np.full(4 * 3600, 60.0, dtype=float); active, storm, major, severe, diagnostics = detect_activity_masks(residual, cadence_s=60.0)
    ready = diagnostics["history_ready"]; first_ready = int(np.flatnonzero(ready)[0]); high_severity = np.isin(flag_activity(residual, cadence_s=60.0), STORM_LEVELS)
    assert not np.any(high_severity[:first_ready]); assert not np.any(storm[:first_ready]); assert not np.any(major[:first_ready]); assert not np.any(severe[:first_ready]); assert np.any(high_severity[first_ready:])


def test_invalid_threshold_order_is_rejected():
    residual = np.zeros(100, dtype=float)
    try: flag_activity(residual, cadence_s=60.0, active_threshold=50.0, storm_threshold=35.0)
    except ValueError: pass
    else: raise AssertionError("invalid threshold ordering was accepted")


def test_profile_round_trip_is_deterministic():
    profile = DetectorProfile(active_nt=20.0, storm_nt=50.0, storm_fast_ratio=2.0)
    payload = {"status": "certified", "profile": json.loads(json.dumps(profile.__dict__))}
    restored = DetectorProfile.from_dict(payload["profile"])
    assert restored == profile


def test_profile_rejects_unsafe_values():
    try: DetectorProfile(active_nt=80.0, storm_nt=35.0).validate()
    except ValueError: pass
    else: raise AssertionError("unsafe profile was accepted")
