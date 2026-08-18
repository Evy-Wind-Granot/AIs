"""Causal, stateful magnetometer detector for live sensor streams."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional
import json
import math
import uuid

import numpy as np

from .baseline import build_design_matrix, robust_harmonic_baseline


@dataclass(frozen=True)
class LiveConfig:
    cadence_s: float = 60.0
    baseline_window_min: float = 24.0 * 60.0
    baseline_update_min: float = 30.0
    amplitude_window_min: float = 120.0
    fast_window_min: float = 10.0
    amplitude_mode: str = "range"
    amplitude_centered: bool = False
    unsettled_nt: float = 20.0
    active_nt: float = 30.0
    minor_storm_nt: float = 83.5
    major_storm_nt: float = 400.0
    severe_storm_nt: float = 800.0
    anomaly_jump_nt: float = 100.0
    fast_trigger_fraction: float = 0.75
    max_plausible_nt: float = 3000.0
    min_plausible_nt: float = -3000.0
    baseline_min_coverage: float = 0.80
    baseline_max_rms_nt: float = 50.0
    event_start_samples: int = 3
    event_clear_samples: int = 10
    escalation_samples: int = 2
    max_gap_samples: int = 3
    candidate_timeout_min: float = 30.0
    max_event_duration_min: float = 720.0

    def __post_init__(self) -> None:
        if self.cadence_s <= 0:
            raise ValueError("cadence_s must be > 0")
        if self.baseline_window_min < 60:
            raise ValueError("baseline_window_min must be >= 60 minutes")
        if self.amplitude_window_min <= 0 or self.fast_window_min <= 0:
            raise ValueError("amplitude windows must be > 0")
        if self.fast_window_min > self.amplitude_window_min:
            raise ValueError("fast_window_min must not exceed amplitude_window_min")
        if not 0 < self.fast_trigger_fraction <= 1:
            raise ValueError("fast_trigger_fraction must be in (0, 1]")
        if self.amplitude_centered:
            raise ValueError("LiveDetector requires amplitude_centered=False")
        if self.amplitude_mode not in {"range", "hybrid", "max", "instant"}:
            raise ValueError(f"Unsupported amplitude_mode: {self.amplitude_mode}")
        if self.event_start_samples < 1 or self.event_clear_samples < 1:
            raise ValueError("event debounce sample counts must be >= 1")
        if self.candidate_timeout_min <= 0 or self.max_event_duration_min <= 0:
            raise ValueError("event duration limits must be > 0")


@dataclass
class _EventState:
    event_id: str
    started_at: str
    started_ts: float
    level: str
    trigger: str
    peak_amplitude_nt: float
    peak_residual_nt: float
    samples: int = 0


class _RollingExtrema:
    """O(1) amortized rolling min/max for a causal fixed-size window."""

    def __init__(self, size: int) -> None:
        self.size = max(1, int(size))
        self.values: Deque[float] = deque()
        self.minimum: Deque[tuple[int, float]] = deque()
        self.maximum: Deque[tuple[int, float]] = deque()
        self.index = -1

    def clear(self) -> None:
        self.values.clear()
        self.minimum.clear()
        self.maximum.clear()
        self.index = -1

    def append(self, value: float) -> tuple[float, float]:
        self.index += 1
        i = self.index
        self.values.append(value)
        while self.minimum and self.minimum[-1][1] >= value:
            self.minimum.pop()
        while self.maximum and self.maximum[-1][1] <= value:
            self.maximum.pop()
        self.minimum.append((i, value))
        self.maximum.append((i, value))
        cutoff = i - self.size + 1
        while self.minimum and self.minimum[0][0] < cutoff:
            self.minimum.popleft()
        while self.maximum and self.maximum[0][0] < cutoff:
            self.maximum.popleft()
        while len(self.values) > self.size:
            self.values.popleft()
        return self.minimum[0][1], self.maximum[0][1]


class LiveDetector:
    """Stateful causal detector for one-sample-at-a-time sensor streams."""

    LEVELS = ("quiet", "unsettled", "active", "minor_storm", "major_storm", "severe_storm")

    def __init__(self, config: LiveConfig = LiveConfig()) -> None:
        self.config = config
        self._raw: Deque[tuple[float, float]] = deque()
        self._residuals = _RollingExtrema(self._window_size(config.amplitude_window_min))
        self._fast_residuals = _RollingExtrema(self._window_size(config.fast_window_min))
        self._baseline_coeffs: Optional[np.ndarray] = None
        self._baseline_origin_ts: Optional[float] = None
        self._last_baseline_update_ts: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._last_residual: Optional[float] = None
        self._event: Optional[_EventState] = None
        self._above_minor = 0
        self._fast_candidates = 0
        self._below_active = 0
        self._candidate_level: Optional[str] = None
        self._candidate_count = 0
        self._last_result: Optional[Dict[str, Any]] = None

    def _window_size(self, minutes: float) -> int:
        return max(1, round(minutes * 60.0 / self.config.cadence_s))

    @classmethod
    def from_pipeline_defaults(cls) -> "LiveDetector":
        from . import legacy_core as legacy
        return cls(LiveConfig(
            cadence_s=legacy.DEFAULT_CADENCE_S,
            amplitude_window_min=legacy.FLAG_AMPLITUDE_WINDOW_MIN,
            amplitude_mode=legacy.FLAG_AMPLITUDE_MODE,
            amplitude_centered=legacy.FLAG_AMPLITUDE_CENTERED,
            unsettled_nt=legacy.FLAG_THRESHOLD_UNSETTLED_NT,
            active_nt=legacy.FLAG_THRESHOLD_ACTIVE_NT,
            minor_storm_nt=legacy.FLAG_THRESHOLD_MINOR_STORM_NT,
            major_storm_nt=legacy.FLAG_THRESHOLD_MAJOR_STORM_NT,
            severe_storm_nt=legacy.FLAG_THRESHOLD_SEVERE_STORM_NT,
            anomaly_jump_nt=legacy.FLAG_THRESHOLD_ANOMALY_JUMP_NT,
            max_plausible_nt=legacy.MAX_PLAUSIBLE_RESIDUAL_NT,
            min_plausible_nt=legacy.MIN_PLAUSIBLE_RESIDUAL_NT,
        ))

    def reset(self) -> None:
        self._raw.clear()
        self._residuals.clear()
        self._fast_residuals.clear()
        self._baseline_coeffs = None
        self._baseline_origin_ts = None
        self._last_baseline_update_ts = None
        self._last_timestamp = None
        self._last_residual = None
        self._event = None
        self._above_minor = 0
        self._fast_candidates = 0
        self._below_active = 0
        self._candidate_level = None
        self._candidate_count = 0
        self._last_result = None

    def _fit_baseline(self, now_ts: float) -> bool:
        c = self.config
        window = max(2, round(c.baseline_window_min * 60.0 / c.cadence_s))
        guard = max(1, c.max_gap_samples)
        samples = list(self._raw)
        if len(samples) <= guard + 9:
            return False
        history = samples[max(0, len(samples) - window - guard):len(samples) - guard]
        if len(history) < 60:
            return False
        values = np.asarray([v for _, v in history], dtype=float)
        finite = np.isfinite(values)
        if finite.mean() < c.baseline_min_coverage:
            return False
        times = np.asarray([t for t, _ in history], dtype=float)
        origin = float(times[0])
        A = build_design_matrix((times - origin) / 3600.0)
        baseline, coeffs = robust_harmonic_baseline(
            values, c.cadence_s, n_iter=4, outlier_threshold_nt=30.0, design_matrix=A
        )
        residual = values - baseline
        finite_res = residual[np.isfinite(residual)]
        if finite_res.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(finite_res))))
        if not np.isfinite(rms) or rms > c.baseline_max_rms_nt:
            return False
        self._baseline_origin_ts = origin
        self._baseline_coeffs = coeffs
        self._last_baseline_update_ts = now_ts
        return True

    def _baseline_at(self, timestamp: float) -> Optional[float]:
        if self._baseline_coeffs is None or self._baseline_origin_ts is None:
            return None
        rel_hours = (timestamp - self._baseline_origin_ts) / 3600.0
        return float(build_design_matrix(np.asarray([rel_hours]))[0] @ self._baseline_coeffs)

    def _classify(self, amplitude: float, residual: float) -> str:
        c = self.config
        if not np.isfinite(residual) or not np.isfinite(amplitude):
            return "invalid"
        if residual > c.max_plausible_nt or residual < c.min_plausible_nt:
            return "invalid"
        if amplitude >= c.severe_storm_nt:
            return "severe_storm"
        if amplitude >= c.major_storm_nt:
            return "major_storm"
        if amplitude >= c.minor_storm_nt:
            return "minor_storm"
        if amplitude >= c.active_nt:
            return "active"
        if amplitude >= c.unsettled_nt:
            return "unsettled"
        return "quiet"

    @classmethod
    def _level_rank(cls, level: str) -> int:
        return cls.LEVELS.index(level) if level in cls.LEVELS else -1

    def _end_event(self, timestamp: float, reason: str) -> Optional[Dict[str, Any]]:
        event = self._event
        if event is None:
            return None
        ts = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        payload = {
            "type": "event_ended",
            "event_id": event.event_id,
            "level": event.level,
            "trigger": event.trigger,
            "timestamp": ts,
            "started_at": event.started_at,
            "peak_amplitude_nt": event.peak_amplitude_nt,
            "peak_residual_nt": event.peak_residual_nt,
            "samples": event.samples,
            "reason": reason,
        }
        self._event = None
        self._below_active = 0
        self._candidate_level = None
        self._candidate_count = 0
        return payload

    def _event_update(self, level: str, amplitude: float, residual: float,
                      fast_trigger: bool, timestamp: float) -> Optional[Dict[str, Any]]:
        c = self.config
        ts = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        storm = self._level_rank(level) >= self._level_rank("minor_storm")
        self._above_minor = self._above_minor + 1 if storm else 0

        if self._event is None:
            # There are two intentionally independent onset paths:
            # 1. a sustained fast anomaly for early warning;
            # 2. a sustained storm classification for gradual events.
            storm_ready = storm and self._above_minor >= c.event_start_samples
            fast_ready = fast_trigger and self._fast_candidates >= c.event_start_samples
            if not storm_ready and not fast_ready:
                return None
            trigger = "storm" if storm_ready else "fast"
            start_level = level if storm_ready else "candidate"
            self._event = _EventState(
                event_id=uuid.uuid4().hex,
                started_at=ts,
                started_ts=timestamp,
                level=start_level,
                trigger=trigger,
                peak_amplitude_nt=float(amplitude),
                peak_residual_nt=float(abs(residual)),
            )
            return {
                "type": "event_started",
                "event_id": self._event.event_id,
                "level": start_level,
                "trigger": trigger,
                "timestamp": ts,
            }

        event = self._event
        event.samples += 1
        event.peak_amplitude_nt = max(event.peak_amplitude_nt, float(amplitude))
        event.peak_residual_nt = max(event.peak_residual_nt, float(abs(residual)))

        age_min = (timestamp - event.started_ts) / 60.0
        if age_min >= c.max_event_duration_min:
            return self._end_event(timestamp, "max_duration")

        if event.level == "candidate" and not storm and age_min >= c.candidate_timeout_min:
            return self._end_event(timestamp, "candidate_timeout")

        if storm and self._level_rank(level) > self._level_rank(event.level):
            if self._candidate_level == level:
                self._candidate_count += 1
            else:
                self._candidate_level = level
                self._candidate_count = 1
            if self._candidate_count >= c.escalation_samples:
                event.level = level
                event.trigger = "storm_confirmed"
                self._candidate_level = None
                self._candidate_count = 0
                return {
                    "type": "event_escalated",
                    "event_id": event.event_id,
                    "level": level,
                    "timestamp": ts,
                }
        else:
            self._candidate_level = None
            self._candidate_count = 0

        if self._level_rank(level) < self._level_rank("active") and not fast_trigger:
            self._below_active += 1
        else:
            self._below_active = 0
        if self._below_active >= c.event_clear_samples:
            return self._end_event(timestamp, "below_active")
        return None

    def update(self, timestamp: Any, value_nt: float) -> Dict[str, Any]:
        c = self.config
        ts = self._to_timestamp(timestamp)
        value = float(value_nt)
        gap = False
        gap_event = None

        if self._last_timestamp is not None:
            delta = ts - self._last_timestamp
            if delta <= 0:
                raise ValueError("timestamps must be strictly increasing")
            if delta > c.cadence_s * (c.max_gap_samples + 1):
                gap = True
                # Never carry an active event across a data outage. The caller
                # receives the explicit termination and can restart cleanly.
                gap_event = self._end_event(self._last_timestamp, "gap")
                self._residuals.clear()
                self._fast_residuals.clear()
                self._last_residual = None
                self._above_minor = 0
                self._fast_candidates = 0

        self._last_timestamp = ts
        self._raw.append((ts, value))
        max_raw = max(round(c.baseline_window_min * 60.0 / c.cadence_s) + c.max_gap_samples + 20, 120)
        while len(self._raw) > max_raw:
            self._raw.popleft()

        if (self._baseline_coeffs is None or self._last_baseline_update_ts is None or
                ts - self._last_baseline_update_ts >= c.baseline_update_min * 60.0):
            self._fit_baseline(ts)

        baseline = self._baseline_at(ts)
        if baseline is None:
            result = {
                "status": "warming_up", "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                "value_nt": value, "gap": gap, "event": gap_event,
            }
            self._last_result = result
            return result

        residual = value - baseline if np.isfinite(value) else float("nan")
        if gap:
            self._residuals.clear()
            self._fast_residuals.clear()

        if np.isfinite(residual):
            slow_min, slow_max = self._residuals.append(residual)
            fast_min, fast_max = self._fast_residuals.append(residual)
            slow_amplitude = slow_max - slow_min
            fast_amplitude = fast_max - fast_min
            if c.amplitude_mode == "range":
                amplitude = slow_amplitude
            elif c.amplitude_mode == "max":
                amplitude = max(abs(slow_min), abs(slow_max))
            elif c.amplitude_mode == "hybrid":
                amplitude = max(slow_amplitude, 2.0 * abs(float(np.mean(self._residuals.values))))
            else:
                amplitude = abs(residual)
            fast_threshold = min(c.minor_storm_nt, c.anomaly_jump_nt) * c.fast_trigger_fraction
            fast_trigger = max(abs(residual), fast_amplitude) >= fast_threshold
        else:
            amplitude = float("nan")
            fast_amplitude = float("nan")
            fast_trigger = False

        self._fast_candidates = self._fast_candidates + 1 if fast_trigger else 0
        level = self._classify(float(amplitude), residual)
        event = gap_event if gap_event is not None else self._event_update(
            level, float(amplitude), residual, fast_trigger, ts
        )

        result = {
            "status": "ok" if level != "invalid" else "invalid",
            "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
            "value_nt": value,
            "baseline_nt": baseline,
            "residual_nt": residual,
            "amplitude_nt": float(amplitude),
            "fast_amplitude_nt": float(fast_amplitude),
            "fast_trigger": bool(fast_trigger),
            "level": level,
            "gap": gap,
            "event": event,
            "active_event_id": self._event.event_id if self._event else None,
        }
        self._last_residual = residual
        self._last_result = result
        return result

    @staticmethod
    def _to_timestamp(value: Any) -> float:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()
        if isinstance(value, np.datetime64):
            return float(value.astype("datetime64[ns]").astype(np.int64)) / 1e9
        if isinstance(value, (int, float)):
            return float(value)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError as exc:
            raise ValueError(f"Unsupported timestamp: {value!r}") from exc

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "last_timestamp": self._last_timestamp,
            "baseline_coeffs": self._baseline_coeffs.tolist() if self._baseline_coeffs is not None else None,
            "baseline_origin_ts": self._baseline_origin_ts,
            "last_baseline_update_ts": self._last_baseline_update_ts,
            "event": self._event.__dict__ if self._event else None,
        }

    def save_state(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state_dict(), handle, indent=2)


__all__ = ["LiveConfig", "LiveDetector"]
