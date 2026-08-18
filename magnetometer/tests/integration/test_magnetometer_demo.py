"""Offline regression tests for the production magnetometer package.

The suite intentionally avoids network access so it can run in CI and during
local refactors without depending on INTERMAGNET, GFZ, or Kyoto availability.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from magnetometer.acquisition import fetch_dst_kyoto
from magnetometer.baseline import (
    build_design_matrix,
    handle_gaps,
    robust_harmonic_baseline,
)
from magnetometer.cache import ResponseCache
from magnetometer.classification import (
    cross_validate_flags,
    disturbance_amplitude,
    flag_activity,
)
from magnetometer.parsing import parse_iaga2002_to_dataframe


class BaselineTests(unittest.TestCase):
    def test_design_matrix_shape_and_constant(self) -> None:
        t = np.arange(24.0)
        matrix = build_design_matrix(t)
        self.assertEqual(matrix.shape, (24, 9))
        np.testing.assert_allclose(matrix[:, 0], 1.0)

    def test_harmonic_baseline_recovers_clean_signal(self) -> None:
        t = np.arange(1440) / 60.0
        expected = 100.0 + 8.0 * np.sin(2.0 * np.pi * t / 24.0)
        observed = expected.copy()
        observed[300:305] += 500.0
        baseline, coeffs = robust_harmonic_baseline(observed, cadence_s=60.0)
        self.assertEqual(len(coeffs), 9)
        error = float(np.sqrt(np.mean((baseline - expected) ** 2)))
        self.assertLess(error, 2.0)

    def test_handle_gaps_respects_interpolation_limit(self) -> None:
        index = pd.date_range("2024-01-01", periods=7, freq="min", tz="UTC")
        series = pd.Series(
            [0.0, 1.0, np.nan, 3.0, np.nan, np.nan, 6.0],
            index=index,
        )
        result = handle_gaps(series, max_gap_samples=1)

        # A one-sample gap is fully filled.
        self.assertAlmostEqual(float(result.iloc[2]), 2.0)

        # For a two-sample gap with limit=1, only the first missing
        # sample is interpolated; the remainder stays missing.
        self.assertAlmostEqual(float(result.iloc[4]), 4.0)
        self.assertTrue(np.isnan(result.iloc[5]))


class ClassificationTests(unittest.TestCase):
    def _kwargs(self) -> dict[str, Any]:
        return {
            "window_min": 1.0,
            "mode": "instant",
            "centered": False,
            "unsettled_nt": 10.0,
            "active_nt": 20.0,
            "minor_storm_nt": 50.0,
            "major_storm_nt": 100.0,
            "severe_storm_nt": 200.0,
            "anomaly_jump_nt": 300.0,
            "max_plausible_nt": 3000.0,
            "min_plausible_nt": -3000.0,
        }

    def test_amplitude_modes_preserve_shape(self) -> None:
        residual = np.array([0.0, 10.0, -20.0, 30.0])
        for mode in ("instant", "range", "hybrid", "max"):
            amplitude = disturbance_amplitude(
                residual,
                60.0,
                window_min=1.0,
                mode=mode,
                centered=False,
            )
            self.assertEqual(amplitude.shape, residual.shape)
            self.assertTrue(np.all(np.isfinite(amplitude)))

    def test_activity_tiers_and_invalid_values(self) -> None:
        residual = np.array(
            [0.0, 15.0, 25.0, 75.0, 150.0, 250.0, 4000.0, np.nan]
        )
        flags = flag_activity(residual, cadence_s=60.0, **self._kwargs())
        self.assertEqual(
            flags.tolist(),
            [
                "quiet",
                "unsettled",
                "active",
                "minor_storm",
                "major_storm",
                "severe_storm",
                "invalid",
                "invalid",
            ],
        )

    def test_cross_validation_marks_global_events(self) -> None:
        local = np.array(
            ["quiet", "quiet", "major_storm", "severe_storm"], dtype=object
        )
        dst = np.array([-80.0, -20.0, 0.0, 0.0])
        kp = np.array([2.0, 5.0, 2.0, 7.0])
        validation = cross_validate_flags(local, dst, kp)
        self.assertEqual(
            validation.tolist(),
            [
                "missed_global_event",
                "under_reacting",
                "unconfirmed_storm",
                "ok",
            ],
        )


class AcquisitionTests(unittest.TestCase):
    def test_dst_wrapper_accepts_tuple_and_two_arguments(self) -> None:
        # AcquisitionClient uses slots, so patch the class method rather than
        # trying to replace a read-only instance attribute.
        with patch("magnetometer.acquisition.AcquisitionClient.fetch_dst") as fetch:
            fetch.return_value = None
            self.assertIsNone(fetch_dst_kyoto((2024, 5)))
            self.assertIsNone(fetch_dst_kyoto(2024, 5))
            self.assertEqual(fetch.call_args_list[0].args, (2024, 5))
            self.assertEqual(fetch.call_args_list[1].args, (2024, 5))


class ParsingTests(unittest.TestCase):
    SAMPLE = """# IAGA-2002
DATE       TIME         DOY     VICX      VICY      VICZ      VICF   |
2024-05-08 00:00:00.000 129     123.4     456.7     789.0     910.1
2024-05-08 00:01:00.000 129     99999.0   450.0     780.0     905.0
"""

    def test_iaga_parser_normalizes_schema_and_sentinels(self) -> None:
        frame = parse_iaga2002_to_dataframe(self.SAMPLE)
        self.assertEqual(list(frame.columns), ["x_nt", "y_nt", "z_nt", "f_nt"])
        self.assertEqual(len(frame), 2)
        self.assertTrue(pd.isna(frame.iloc[1]["x_nt"]))
        self.assertEqual(str(frame.index.tz), "UTC")

    def test_html_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_iaga2002_to_dataframe("<html><body>server error</body></html>")


class CacheTests(unittest.TestCase):
    def test_response_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(Path(tmp), ttl_hours=24.0)
            key = ResponseCache.key("https://example.test/data", {"b": 2, "a": 1})
            cache.put(key, 200, "payload")
            self.assertEqual(cache.get(key), (200, "payload"))
            self.assertTrue(cache.contains(key))


if __name__ == "__main__":
    unittest.main()
