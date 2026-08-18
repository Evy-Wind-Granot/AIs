"""Offline tests for the historical calibration workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from magnetometer.calibration.calibrate_historical_magnetometer import (
    calibrated_config_from_best,
    candidate_thresholds,
    choose_best,
    score_threshold,
)
from magnetometer.demos import magnetometer_demo as md


class CalibrationTests(unittest.TestCase):
    def test_score_threshold(self) -> None:
        amplitude = np.array([1.0, 2.0, 10.0, 12.0, 20.0, 25.0])
        truth = np.array([0.0, 0.0, 3.0, 3.0, 4.0, 0.0])
        metrics = score_threshold(amplitude, truth, 10.0)
        # pred = amplitude >= 10 -> indices 2,3,4,5
        # truth_storm = truth >= 3 -> indices 2,3,4
        # TP=3, FP=1, FN=0, TN=2
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tn"], 2)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 * 0.75 * 1.0 / (0.75 + 1.0))

    def test_candidate_thresholds_are_positive_and_unique(self) -> None:
        thresholds = candidate_thresholds(np.array([0.0, 1.0, 2.0, 2.0, 5.0]))
        self.assertTrue(np.all(thresholds > 0))
        self.assertEqual(len(thresholds), len(np.unique(thresholds)))

    def test_choose_best_prefers_false_alarm_budget(self) -> None:
        truth = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0])
        amplitudes = {
            (60.0, "range"): np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]),
            (180.0, "max"): np.array([1.0, 2.0, 20.0, 21.0, 22.0, 23.0]),
        }
        best = choose_best(amplitudes, truth, max_false_alarm_rate=0.01)
        self.assertTrue(best["eligible"])
        self.assertEqual(best["fp"], 0)
        self.assertAlmostEqual(best["f1"], 1.0)

    def test_calibrated_config_is_nested_and_loadable(self) -> None:
        best = {
            "window_min": 120.0,
            "mode": "range",
            "threshold_nt": 83.5,
        }
        config = calibrated_config_from_best(best)
        self.assertIn("thresholds", config)
        self.assertEqual(config["thresholds"]["amplitude_window_min"], 120.0)
        self.assertEqual(config["thresholds"]["amplitude_mode"], "range")
        self.assertEqual(config["thresholds"]["amplitude_centered"], False)
        self.assertEqual(config["thresholds"]["minor_storm"], 83.5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrated.json"
            path.write_text(json.dumps(config))
            # Snapshot defaults, load, assert override, then restore.
            prev = {
                "FLAG_AMPLITUDE_WINDOW_MIN": md.FLAG_AMPLITUDE_WINDOW_MIN,
                "FLAG_AMPLITUDE_MODE": md.FLAG_AMPLITUDE_MODE,
                "FLAG_AMPLITUDE_CENTERED": md.FLAG_AMPLITUDE_CENTERED,
                "FLAG_THRESHOLD_MINOR_STORM_NT": md.FLAG_THRESHOLD_MINOR_STORM_NT,
            }
            try:
                md.load_config(str(path))
                self.assertEqual(md.FLAG_AMPLITUDE_WINDOW_MIN, 120.0)
                self.assertEqual(md.FLAG_AMPLITUDE_MODE, "range")
                self.assertEqual(md.FLAG_AMPLITUDE_CENTERED, False)
                self.assertEqual(md.FLAG_THRESHOLD_MINOR_STORM_NT, 83.5)
            finally:
                for k, v in prev.items():
                    setattr(md, k, v)
                    # Also restore on the legacy module globals used by core wrappers.
                    import magnetometer.legacy_core as legacy

                    setattr(legacy, k, v)

    def test_loaded_threshold_changes_classification(self) -> None:
        """Prove that changing the calibrated minor_storm threshold changes flags."""
        residual = np.array([0.0, 50.0, 90.0, 110.0, 0.0], dtype=float)
        # Use a short instant window so amplitude == |residual|.
        import magnetometer.legacy_core as legacy

        prev = {
            "FLAG_AMPLITUDE_WINDOW_MIN": legacy.FLAG_AMPLITUDE_WINDOW_MIN,
            "FLAG_AMPLITUDE_MODE": legacy.FLAG_AMPLITUDE_MODE,
            "FLAG_AMPLITUDE_CENTERED": legacy.FLAG_AMPLITUDE_CENTERED,
            "FLAG_THRESHOLD_MINOR_STORM_NT": legacy.FLAG_THRESHOLD_MINOR_STORM_NT,
            "FLAG_THRESHOLD_UNSETTLED_NT": legacy.FLAG_THRESHOLD_UNSETTLED_NT,
            "FLAG_THRESHOLD_ACTIVE_NT": legacy.FLAG_THRESHOLD_ACTIVE_NT,
        }
        try:
            legacy.FLAG_AMPLITUDE_WINDOW_MIN = 0.0
            legacy.FLAG_AMPLITUDE_MODE = "instant"
            legacy.FLAG_AMPLITUDE_CENTERED = False
            legacy.FLAG_THRESHOLD_UNSETTLED_NT = 15.0
            legacy.FLAG_THRESHOLD_ACTIVE_NT = 35.0

            legacy.FLAG_THRESHOLD_MINOR_STORM_NT = 100.0
            flags_high = md.flag_activity(residual, 60.0)

            legacy.FLAG_THRESHOLD_MINOR_STORM_NT = 80.0
            flags_low = md.flag_activity(residual, 60.0)

            # Sample at index 3 has |r|=110: storm under both thresholds.
            # Sample at index 2 has |r|=90: storm only under the lower threshold.
            self.assertNotEqual(list(flags_high), list(flags_low))
            self.assertIn(flags_low[2], ("minor_storm", "major_storm", "severe_storm"))
            self.assertNotIn(flags_high[2], ("minor_storm", "major_storm", "severe_storm"))
        finally:
            for k, v in prev.items():
                setattr(legacy, k, v)
                setattr(md, k, v)


if __name__ == "__main__":
    unittest.main()
