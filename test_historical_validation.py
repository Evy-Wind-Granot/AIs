"""Offline tests for the historical validation benchmark."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from validate_historical_magnetometer import Aggregator, binary_metrics, global_levels_from_indices


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

    def test_confusion_matrix_ignores_nan_local_levels(self) -> None:
        aggregate = Aggregator()
        index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
        flags = np.array(["quiet", "minor_storm", None, "severe_storm"], dtype=object)
        kp = np.array([1.0, 6.0, 7.0, 8.0])
        dst = np.full(4, np.nan)
        aggregate.add("VIC", index, flags, kp, dst, global_levels_from_indices(kp, dst))
        np.testing.assert_array_equal(
            aggregate.confusion,
            np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=np.int64),
        )

    def test_coverage_reports_excluded_failed_chunks(self) -> None:
        aggregate = Aggregator(requested_samples=10)
        index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
        flags = np.array(["quiet", "quiet", "minor_storm", "quiet"], dtype=object)
        kp = np.array([1.0, 1.0, 6.0, 1.0])
        dst = np.full(4, np.nan)
        aggregate.add("VIC", index, flags, kp, dst, global_levels_from_indices(kp, dst))
        aggregate.chunks_ok = 1
        aggregate.chunks_failed = 1
        coverage = aggregate.report()["coverage"]
        self.assertEqual(coverage["requested_samples"], 10)
        self.assertEqual(coverage["evaluated_samples"], 4)
        self.assertEqual(coverage["excluded_samples"], 6)
        self.assertAlmostEqual(coverage["evaluation_coverage"], 0.4)

    def test_event_level_detection_tracks_contiguous_global_storm(self) -> None:
        aggregate = Aggregator()
        index = pd.date_range("2024-01-01", periods=7, freq="min", tz="UTC")
        flags = np.array(["quiet", "minor_storm", "major_storm", "quiet", "quiet", "severe_storm", "quiet"], dtype=object)
        kp = np.array([1.0, 6.0, 7.0, 1.0, 1.0, 8.0, 1.0])
        dst = np.full(7, np.nan)
        aggregate.add("VIC", index, flags, kp, dst, global_levels_from_indices(kp, dst))
        events = aggregate.report()["event_level_performance"]
        self.assertEqual(events["global_events"], 2)
        self.assertEqual(events["events_detected"], 2)
        self.assertEqual(events["events_missed"], 0)
        self.assertEqual(events["severe_global_events"], 1)


if __name__ == "__main__":
    unittest.main()
