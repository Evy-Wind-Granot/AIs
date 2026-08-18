"""Offline tests for replay event matching and diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from magnetometer.validation.replay.replay_live_detector import _event_diagnostics, _match_events


class ReplayEventDiagnosticTests(unittest.TestCase):
    def test_early_warning_matches_and_reports_lead(self) -> None:
        start = pd.Timestamp(datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc))
        end = start + pd.Timedelta(minutes=30)
        detector_start = start - pd.Timedelta(minutes=20)
        detector_end = end + pd.Timedelta(minutes=10)
        matched, deltas, leads, mapping = _match_events(
            [(start, end)],
            [{"event_id": "e1", "start": detector_start, "end": detector_end}],
        )
        self.assertEqual(matched, 1)
        self.assertEqual(deltas, [-20.0])
        self.assertEqual(leads, [20.0])
        self.assertEqual(mapping, {1: "e1"})

    def test_missed_event_is_explicit_and_actionable(self) -> None:
        start = pd.Timestamp(datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc))
        end = start + pd.Timedelta(minutes=4)
        timestamps = pd.date_range(start, end, freq="min", tz="UTC")
        predictions = {
            ts.isoformat(): {
                "level": "active",
                "amplitude_nt": 72.5,
                "fast_amplitude_nt": 10.0,
            }
            for ts in timestamps
        }
        diagnostics = _event_diagnostics([(start, end)], [], predictions, {})
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["status"], "missed")
        self.assertEqual(diagnostics[0]["truth_event"], 1)
        self.assertEqual(diagnostics[0]["peak_amplitude_nt"], 72.5)
        self.assertIn("never classified a storm", diagnostics[0]["reason"])


if __name__ == "__main__":
    unittest.main()
