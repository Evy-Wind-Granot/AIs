from datetime import datetime, timezone
import math

import pytest

from magnetometer.message_schema import magnetic_field_magnitude_nt, validate_magnetometer_message
from magnetometer.detecting.live_detector import MagnetometerDetector


def make_message(seq: int, seconds: int) -> dict:
    return {
        "sequence_number": seq,
        "timestamp": [{"seconds": seconds, "nanoseconds": 0, "source": "NTP-LOCAL"}],
        "payload": {
            "x_nt": 21000.5,
            "y_nt": -3400.2,
            "z_nt": 48000.0,
            "local_temperature_c": 24.9375,
            "remote_temperature_c": 22.5625,
        },
    }


def test_canonical_message_is_valid_and_total_field_is_xyz_magnitude():
    message = make_message(67890, 1783123456)
    normalized = validate_magnetometer_message(message)
    expected = math.sqrt(21000.5**2 + (-3400.2)**2 + 48000.0**2)
    assert normalized["sequence_number"] == 67890
    assert normalized["timestamp"].tzinfo == timezone.utc
    assert magnetic_field_magnitude_nt(message) == pytest.approx(expected)


def test_payload_rejects_unknown_fields():
    message = make_message(1, 1783123456)
    message["payload"]["f_nt"] = 54000.0
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_magnetometer_message(message)


def test_live_detector_accepts_wire_messages_and_is_causal():
    detector = MagnetometerDetector(min_samples_per_bucket=1)
    outputs = []
    start = 1783123200
    for sequence in range(1, 365):
        result = detector.process_message(make_message(sequence, start + sequence * 60))
        if result is not None:
            outputs.append(result)
    assert outputs
    assert outputs[-1].input_schema == "magnetometer.v1"
    assert outputs[-1].detector_version == "causal-disturbance-v2.1"
    assert outputs[-1].ready
    assert outputs[-1].classification == "quiet"
    assert abs(outputs[-1].residual_nt) < 1e-6


def test_live_detector_rejects_out_of_order_timestamps():
    detector = MagnetometerDetector(min_samples_per_bucket=1)
    detector.process_message(make_message(1, 1783123200))
    with pytest.raises(ValueError, match="strictly increasing"):
        detector.process_message(make_message(2, 1783123199))
