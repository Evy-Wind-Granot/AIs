"""Validation and normalization for the canonical magnetometer.v1 wire message."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping

MESSAGE_SCHEMA_VERSION = "magnetometer.v1"


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_magnetometer_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one magnetometer.v1 envelope and return a safe normalized copy.

    The detector accepts the wire format directly; no positional-array or legacy
    field mapping is performed here. This keeps the production input contract
    explicit and prevents silent x/y/z swaps.
    """
    if not isinstance(message, Mapping):
        raise ValueError("magnetometer message must be an object")
    if "sequence_number" not in message:
        raise ValueError("missing sequence_number")
    sequence = message["sequence_number"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence_number must be a non-negative integer")

    timestamps = message.get("timestamp")
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError("timestamp must be a non-empty array")
    ts = timestamps[0]
    if not isinstance(ts, Mapping):
        raise ValueError("timestamp[0] must be an object")
    seconds = ts.get("seconds")
    nanos = ts.get("nanoseconds")
    source = ts.get("source")
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise ValueError("timestamp[0].seconds must be an integer")
    if isinstance(nanos, bool) or not isinstance(nanos, int) or not 0 <= nanos < 1_000_000_000:
        raise ValueError("timestamp[0].nanoseconds must be in [0, 1e9)")
    if not isinstance(source, str) or not source:
        raise ValueError("timestamp[0].source must be a non-empty string")

    payload = message.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    allowed = {"x_nt", "y_nt", "z_nt", "local_temperature_c", "remote_temperature_c"}
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"payload contains unsupported fields: {sorted(extra)}")
    values = {axis: _finite_number(payload[axis], f"payload.{axis}") for axis in ("x_nt", "y_nt", "z_nt") if axis in payload}
    if set(values) != {"x_nt", "y_nt", "z_nt"}:
        raise ValueError("payload requires x_nt, y_nt and z_nt")
    for name in ("local_temperature_c", "remote_temperature_c"):
        if name in payload:
            values[name] = _finite_number(payload[name], f"payload.{name}")

    timestamp = datetime.fromtimestamp(seconds + nanos / 1e9, tz=timezone.utc)
    return {
        "sequence_number": sequence,
        "timestamp": timestamp,
        "timestamp_seconds": seconds + nanos / 1e9,
        "timestamp_source": source,
        "payload": values,
    }


def magnetic_field_magnitude_nt(message: Mapping[str, Any]) -> float:
    """Return total-field magnitude F from a validated magnetometer message."""
    normalized = validate_magnetometer_message(message)
    p = normalized["payload"]
    return math.sqrt(p["x_nt"] ** 2 + p["y_nt"] ** 2 + p["z_nt"] ** 2)
