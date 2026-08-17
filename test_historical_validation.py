"""Offline tests for the historical validation benchmark."""

from __future__ import annotations

import unittest

import numpy as np

from validate_historical_magnetometer import binary_metrics, global_levels_from_indices


class HistoricalValidationTests(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        metrics = binary_metrics(tp=80, fp=10, fn=20, tn=890)
        self.assertAlmostEqual(metrics["detection_rate"], 0.8)
        self.assertAlmostEqual(metrics["recall"], 0.8)
        self.assertAlmostEqual(metrics["precision"], 80 / 90)
        self.assertAlmostEqual(metrics["f1"], 2 * (80 / 90) * 0.8 / ((80 / 90) + 0.8))
        self.assertAlmostEqual(metrics["false_alarm_rate"], 10 / 900)
        self.assertAlmostEqual(metrics["missed_event_rate"], 0.2)

    def test_global_levels_match_production_boundaries(self) -> None:
        kp = np.array([2.0, 4.0, 5.9, 6.0, 7.9, 8.0, np.nan])
        dst = np.full_like(kp, np.nan)
        np.testing.assert_array_equal(
            global_levels_from_indices(kp, dst)[:6],
            np.array([0.0, 1.0, 2.0, 3.0, 3.0, 4.0]),
        )

    def test_dst_can_raise_global_level(self) -> None:
        kp = np.array([2.0, 4.0, 5.0, np.nan, np.nan])
        dst = np.array([0.0, -35.0, -60.0, -101.0, -20.0])
        np.testing.assert_array_equal(
            global_levels_from_indices(kp, dst),
            np.array([0.0, 2.0, 3.0, 4.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
