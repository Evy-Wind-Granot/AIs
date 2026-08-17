"""Offline tests for the historical calibration workflow."""

from __future__ import annotations

import unittest

import numpy as np

from calibrate_historical_magnetometer import candidate_thresholds, choose_best, score_threshold


class CalibrationTests(unittest.TestCase):
    def test_score_threshold(self) -> None:
        amplitude = np.array([1.0, 2.0, 10.0, 12.0, 20.0, 25.0])
        truth = np.array([0.0, 0.0, 3.0, 3.0, 4.0, 0.0])
        metrics = score_threshold(amplitude, truth, 10.0)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tn"], 4)
        self.assertAlmostEqual(metrics["f1"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
