#!/usr/bin/env python3
"""Regression tests for the deterministic production detector."""

import numpy as np

from detector_core import detect_activity_masks, flag_activity


def test_isolated_spike_does_not_create_storm():
    residual = np.zeros(6 * 60, dtype=float)
    residual[180] = 250.0
    flags = flag_activity(residual, cadence_s=60.0)
    assert not np.any(np.isin(flags, ["minor_storm", "major_storm", "severe_storm"]))


def test_sustained_storm_is_detected():
    residual = np.zeros(6 * 60, dtype=float)
    residual[60:300] = 60.0
    active, storm, major, severe, _ = detect_activity_masks(
        residual, cadence_s=60.0, active_threshold=15.0, storm_threshold=35.0
    )
    assert storm[150:240].mean() > 0.8
    assert active[150:240].mean() > 0.8
    assert not np.any(major)
    assert not np.any(severe)


def test_short_dip_does_not_split_storm_event():
    residual = np.zeros(10 * 60, dtype=float)
    residual[60:540] = 60.0
    residual[300:310] = 5.0
    flags = flag_activity(residual, cadence_s=60.0)
    storm = np.isin(flags, ["minor_storm", "major_storm", "severe_storm"])
    assert storm[180:420].mean() > 0.9


def test_long_moderate_disturbance_uses_long_context():
    # A long disturbance can be physically meaningful without remaining above
    # the storm threshold at every short timescale.
    residual = np.zeros(8 * 60, dtype=float)
    residual[60:420] = 27.0
    residual[210:225] = 12.0
    flags = flag_activity(residual, cadence_s=60.0, active_threshold=15.0, storm_threshold=35.0)
    storm = np.isin(flags, ["minor_storm", "major_storm", "severe_storm"])
    assert storm[150:360].mean() > 0.75


def test_nan_samples_are_safe():
    residual = np.full(120, np.nan, dtype=float)
    residual[30:90] = 20.0
    flags = flag_activity(residual, cadence_s=60.0)
    assert np.all(flags[np.isnan(residual)] == "quiet")
    assert np.all(np.isfinite(residual) | (flags == "quiet"))


def test_detector_is_strictly_causal():
    """Adding future samples must not change already-emitted classifications."""
    prefix = np.zeros(5 * 3600, dtype=float)
    prefix[4 * 3600:] = 18.0
    future = np.full(3 * 3600, 250.0, dtype=float)
    short = flag_activity(prefix, cadence_s=60.0)
    long = flag_activity(np.concatenate([prefix, future]), cadence_s=60.0)
    assert np.array_equal(short, long[: len(prefix)])


def test_startup_requires_history():
    """The detector does not fabricate classifications before all windows exist."""
    residual = np.full(4 * 3600, 60.0, dtype=float)
    flags = flag_activity(residual, cadence_s=60.0)
    # Three-hour context is the slowest feature, so the initial warm-up remains quiet.
    assert np.all(flags[: 3 * 3600] == "quiet")
    assert np.any(np.isin(flags[3 * 3600 :], ["minor_storm", "major_storm", "severe_storm"]))
