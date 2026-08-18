"""Tests for resilient historical upstream acquisition."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

import requests

from magnetometer.acquisition import AcquisitionClient, AcquisitionError


IAGA = """Format IAGA-2002 |
IAGA CODE VIC |
DATE TIME DOY VICH VICE VICZ VICF |
2024-01-01 00:00:00.000 001 1.0 2.0 3.0 4.0
"""


class AcquisitionTests(unittest.TestCase):
    def test_incomplete_chunked_response_is_retried_and_not_cached(self) -> None:
        session = Mock()
        response = Mock(status_code=200, text=IAGA)
        session.get.side_effect = [
            requests.exceptions.ChunkedEncodingError("truncated body"),
            response,
        ]
        client = AcquisitionClient(session=session, cache_enabled=False)

        status, body = client.get_text("https://example.test/data", transport_retries=1)

        self.assertEqual(status, 200)
        self.assertEqual(body, IAGA)
        self.assertEqual(session.get.call_count, 2)

    def test_transport_failure_is_bounded(self) -> None:
        session = Mock()
        session.get.side_effect = requests.exceptions.ConnectionError("offline")
        client = AcquisitionClient(session=session, cache_enabled=False)

        with self.assertRaises(AcquisitionError):
            client.get_text("https://example.test/data", transport_retries=1)

        self.assertEqual(session.get.call_count, 2)

    def test_long_station_request_is_split_into_small_chunks(self) -> None:
        session = Mock()
        session.get.side_effect = [
            Mock(status_code=200, text=IAGA),
            Mock(status_code=200, text=IAGA),
            Mock(status_code=200, text=IAGA),
        ]
        client = AcquisitionClient(session=session, cache_enabled=False)

        body = client.fetch_station(
            "VIC", "2024-01-01", duration_days=15, samples_per_day="Minute"
        )

        self.assertEqual(body.count("IAGA CODE VIC"), 3)
        self.assertEqual(session.get.call_count, 3)
        durations = [call.kwargs["params"]["dataDuration"] for call in session.get.call_args_list]
        starts = [call.kwargs["params"]["dataStartDate"] for call in session.get.call_args_list]
        self.assertEqual(durations, [7, 7, 1])
        self.assertEqual(starts, ["2024-01-01", "2024-01-08", "2024-01-15"])


if __name__ == "__main__":
    unittest.main()
