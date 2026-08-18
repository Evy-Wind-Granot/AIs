"""Tests for the causal streaming magnetometer detector."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from magnetometer.live import LiveConfig, LiveDetector


class LiveDetectorTests(unittest.TestCase):
    def _detector(self) -> LiveDetector:
        return LiveDetector(
            LiveConfig(
                cadence_s=60.0,
                baseline_window_min=60.0,
                baseline_update_min=30.0,
                amplitude_window_min=10.0,
                amplitude_mode="range",
                minor_storm_nt=80.0,
                major_storm_nt=150.0,
                severe_storm_nt=300.0,
                event_start_samples=3,
                event_clear_samples=3,
                escalation_samples=2,
            )
        )

    def test_warmup_does_not_classify(self) -> None:
        detector = self._detector()
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(30):
            result = detector.update(t + timedelta(minutes=i), 100.0)
        self.assertEqual(result["status"], "warming_up")
        self.assertIsNone(result["event"])

    def test_event_starts_after_debounce(self) -> None:
        detector = self._detector()
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(80):
            detector.update(t + timedelta(minutes=i), 100.0)

        events = []
        for i in range(80, 95):
            result = detector.update(t + timedelta(minutes=i), 200.0)
            if result["event"] is not None:
                events.append(result["event"])

        self.assertTrue(events)
        self.assertEqual(events[0]["type"], "event_started")
        self.assertIn(events[0]["level"], {"minor_storm", "major_storm"})

    def test_gradual_storm_starts_without_fast_trigger(self) -> None:
        """A slow amplitude build must not depend on the fast anomaly path."""
        detector = self._detector()
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(80):
            detector.update(t + timedelta(minutes=i), 100.0)

        events = []
        # The 10-minute range crosses the storm threshold gradually, while no
        # single sample is large enough to satisfy the fast anomaly threshold.
        values = [140.0, 150.0, 160.0, 170.0, 180.0]
        for offset, value in enumerate(values, start=80):
            result = detector.update(t + timedelta(minutes=offset), value)
            if result["event"] is not None:
                events.append(result["event"])

        self.assertTrue(events)
        self.assertEqual(events[0]["type"], "event_started")
        self.assertEqual(events[0]["trigger"], "storm")

    def test_gap_ends_active_event(self) -> None:
        detector = self._detector()
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(80):
            detector.update(t + timedelta(minutes=i), 100.0)
        started = None
        for i in range(80, 85):
            result = detector.update(t + timedelta(minutes=i), 200.0)
            if result["event"] and result["event"]["type"] == "event_started":
                started = result["event"]
        self.assertIsNotNone(started)

        result = detector.update(t + timedelta(minutes=95), 100.0)
        self.assertTrue(result["gap"])
        self.assertIsNotNone(result["event"])
        self.assertEqual(result["event"]["type"], "event_ended")
        self.assertEqual(result["event"]["reason"], "gap")
        self.assertIsNone(result["active_event_id"])

    def test_gap_does_not_bridge_amplitude_window(self) -> None:
        detector = self._detector()
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(80):
            detector.update(t + timedelta(minutes=i), 100.0)
        detector.update(t + timedelta(minutes=80), 200.0)
        result = detector.update(t + timedelta(minutes=90), 100.0)
        self.assertTrue(result["gap"])
        self.assertIsNone(result["active_event_id"])

    def test_timestamps_must_increase(self) -> None:
        detector = self._detector()
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        detector.update(t, 100.0)
        with self.assertRaises(ValueError):
            detector.update(t, 100.0)


if __name__ == "__main__":
    unittest.main()
