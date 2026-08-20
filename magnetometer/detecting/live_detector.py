"""Stateful production detector for canonical magnetometer.v1 messages.

Wire messages are ~1 Hz XYZ readings. The live path aggregates them causally into
one-minute total-field samples, maintains a trailing-only harmonic baseline, and
feeds the resulting residuals into the certified detector. No future samples are
used and malformed/out-of-order input is rejected rather than guessed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

import numpy as np

from ..causal_baseline import build_design_matrix, robust_harmonic_fit
from ..detector_core import DETECTOR_VERSION, DetectorProfile, flag_activity
from ..message_schema import MESSAGE_SCHEMA_VERSION, validate_magnetometer_message, magnetic_field_magnitude_nt


@dataclass(frozen=True)
class DetectionResult:
    timestamp: datetime
    sequence_number: int
    field_magnitude_nt: float
    baseline_nt: float
    residual_nt: float
    classification: str
    ready: bool
    detector_version: str = DETECTOR_VERSION
    input_schema: str = MESSAGE_SCHEMA_VERSION
    sequence_gap: bool = False
    reset: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "sequence_number": self.sequence_number,
            "field_magnitude_nt": self.field_magnitude_nt,
            "baseline_nt": self.baseline_nt,
            "residual_nt": self.residual_nt,
            "classification": self.classification,
            "ready": self.ready,
            "detector_version": self.detector_version,
            "input_schema": self.input_schema,
            "sequence_gap": self.sequence_gap,
            "reset": self.reset,
        }


class MagnetometerDetector:
    """Causal, restart-safe detector fed directly by magnetometer.v1 messages."""

    def __init__(
        self,
        *,
        profile: Optional[DetectorProfile] = None,
        cadence_s: float = 60.0,
        history_hours: float = 24.0,
        detector_history_hours: float = 4.0,
        refit_minutes: float = 15.0,
        max_input_gap_s: float = 180.0,
    ) -> None:
        if cadence_s <= 0 or history_hours <= 0 or detector_history_hours <= 0 or refit_minutes <= 0:
            raise ValueError("cadence_s, history_hours, detector_history_hours and refit_minutes must be positive")
        if max_input_gap_s <= 0:
            raise ValueError("max_input_gap_s must be positive")
        self.profile = profile or DetectorProfile()
        self.cadence_s = float(cadence_s)
        self.history_limit = max(32, int(round(history_hours * 3600 / self.cadence_s)) + 2)
        self.detector_limit = max(361, int(round(detector_history_hours * 3600 / self.cadence_s)) + 2)
        self.refit = timedelta(minutes=float(refit_minutes))
        self.max_input_gap = float(max_input_gap_s)
        self._samples: deque[tuple[datetime, float]] = deque(maxlen=self.history_limit)
        self._residuals: deque[float] = deque(maxlen=self.detector_limit)
        self._bucket_start: Optional[datetime] = None
        self._bucket_values: list[float] = []
        self._last_timestamp: Optional[datetime] = None
        self._last_sequence: Optional[int] = None
        self._coeff: Optional[np.ndarray] = None
        self._ref_min: Optional[float] = None
        self._ref_max: Optional[float] = None
        self._last_fit: Optional[datetime] = None

    @property
    def ready(self) -> bool:
        return len(self._residuals) >= 361 and self._residuals[-1] == self._residuals[-1]

    def reset(self) -> None:
        self._samples.clear()
        self._residuals.clear()
        self._bucket_start = None
        self._bucket_values.clear()
        self._coeff = None
        self._ref_min = None
        self._ref_max = None
        self._last_fit = None

    def process_message(self, message: Mapping[str, Any]) -> Optional[DetectionResult]:
        normalized = validate_magnetometer_message(message)
        timestamp = normalized["timestamp"]
        sequence = normalized["sequence_number"]
        if self._last_timestamp is not None:
            delta = (timestamp - self._last_timestamp).total_seconds()
            if delta <= 0:
                raise ValueError("magnetometer timestamps must be strictly increasing")
            if delta > self.max_input_gap:
                self.reset()
        sequence_gap = self._last_sequence is not None and sequence != self._last_sequence + 1
        self._last_timestamp = timestamp
        self._last_sequence = sequence
        magnitude = magnetic_field_magnitude_nt(message)
        bucket = timestamp.replace(second=0, microsecond=0)
        if self._bucket_start is None:
            self._bucket_start = bucket
        if bucket == self._bucket_start:
            self._bucket_values.append(magnitude)
            return None
        result = self._finalize_bucket(sequence=sequence, sequence_gap=sequence_gap)
        self._bucket_start = bucket
        self._bucket_values = [magnitude]
        return result

    def flush(self, *, sequence_number: Optional[int] = None) -> Optional[DetectionResult]:
        if self._bucket_start is None or not self._bucket_values:
            return None
        sequence = self._last_sequence if sequence_number is None else sequence_number
        if sequence is None:
            raise RuntimeError("cannot flush without a sequence number")
        return self._finalize_bucket(sequence=sequence, sequence_gap=False)

    def _finalize_bucket(self, *, sequence: int, sequence_gap: bool) -> DetectionResult:
        assert self._bucket_start is not None
        if not self._bucket_values:
            raise RuntimeError("cannot finalize an empty bucket")
        timestamp = self._bucket_start + timedelta(seconds=self.cadence_s - 1e-3)
        value = float(np.median(np.asarray(self._bucket_values, dtype=float)))
        baseline = self._causal_baseline(timestamp, value)
        residual = value - baseline
        self._samples.append((timestamp, value))
        self._residuals.append(residual)

        residuals = np.asarray(self._residuals, dtype=float)
        flags = flag_activity(residuals, cadence_s=self.cadence_s, profile=self.profile)
        classification = str(flags[-1])
        ready = len(residuals) >= 361 and np.isfinite(residual)
        if not ready:
            classification = "warming_up"
        return DetectionResult(
            timestamp=timestamp,
            sequence_number=sequence,
            field_magnitude_nt=value,
            baseline_nt=baseline,
            residual_nt=residual,
            classification=classification,
            ready=ready,
            sequence_gap=sequence_gap,
            reset=False,
        )

    def _causal_baseline(self, timestamp: datetime, current_value: float) -> float:
        if not self._samples:
            return current_value
        should_fit = self._coeff is None or self._last_fit is None or timestamp - self._last_fit >= self.refit
        if should_fit and len(self._samples) >= 12:
            history = list(self._samples)[-self.history_limit:]
            values = np.asarray([v for _, v in history], dtype=float)
            times = np.asarray([(t - history[0][0]).total_seconds() / 3600.0 for t, _ in history], dtype=float)
            coeff = robust_harmonic_fit(values, self.cadence_s, times)
            if np.all(np.isfinite(coeff)):
                self._coeff = coeff
                self._ref_min = float(times.min())
                self._ref_max = float(times.max())
                self._last_fit = timestamp
        if self._coeff is not None and self._ref_min is not None and self._ref_max is not None:
            t = (timestamp - self._samples[0][0]).total_seconds() / 3600.0
            return float(build_design_matrix(np.asarray([t]), self._ref_min, self._ref_max) @ self._coeff)
        return float(self._samples[-1][1])
