#!/usr/bin/env python3
"""Causally replay historical magnetometer samples through LiveDetector.

The replay is deliberately independent of future observations. It normalizes
chunk boundaries before scoring so overlapping upstream responses or duplicate
timestamps cannot inflate truth counts. Event scoring is interval-based: an
alert that starts shortly before a reference storm is an early warning, not a
miss simply because its ``event_started`` timestamp precedes the reference
onset.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from magnetometer.acquisition import fetch_intermagnet_iaga2002
from magnetometer.config_strict import load_config
from magnetometer.live import LiveDetector
from magnetometer.parsing import parse_iaga2002_to_dataframe
from validate_historical_magnetometer import align_global_indices, binary_metrics, fetch_global_indices

STORM_FLAGS = {"minor_storm", "major_storm", "severe_storm"}
VALID_LEVELS = {"quiet", "unsettled", "active", *STORM_FLAGS}


@dataclass
class ReplayMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    evaluated: int = 0
    global_truth: int = 0
    warming_up: int = 0
    invalid: int = 0
    errors: int = 0
    gaps: int = 0
    duplicate_samples: int = 0
    missing_samples: int = 0
    events_started: int = 0
    events_escalated: int = 0
    events_ended: int = 0
    truth_events: int = 0
    truth_events_detected: int = 0
    onset_deltas_min: list[float] | None = None
    lead_times_min: list[float] | None = None

    def __post_init__(self) -> None:
        if self.onset_deltas_min is None:
            self.onset_deltas_min = []
        if self.lead_times_min is None:
            self.lead_times_min = []


def _month_windows(start: pd.Timestamp, end: pd.Timestamp):
    cursor = start
    while cursor < end:
        month_end = cursor.normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        nxt = min(month_end, end)
        yield cursor, nxt
        cursor = nxt


def _as_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _truth_events(index: pd.DatetimeIndex, global_level: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: Optional[pd.Timestamp] = None
    last: Optional[pd.Timestamp] = None
    cadence = pd.Timedelta(minutes=1)
    for ts, level in zip(index, global_level):
        storm = np.isfinite(level) and level >= 3
        if storm:
            if start is None or last is None or ts - last > cadence:
                if start is not None and last is not None:
                    events.append((start, last))
                start = ts
            last = ts
        elif start is not None and last is not None:
            events.append((start, last))
            start = last = None
    if start is not None and last is not None:
        events.append((start, last))
    return events


def _safe_float(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _event_type(result: dict) -> Optional[str]:
    event = result.get("event")
    return event.get("type") if isinstance(event, dict) else None


def _close_event_intervals(
    event_intervals: dict[str, dict],
    last_timestamp: Optional[pd.Timestamp],
) -> list[dict]:
    if last_timestamp is not None:
        for event in event_intervals.values():
            if event.get("end") is None:
                event["end"] = last_timestamp
    return [e for e in event_intervals.values() if e.get("start") is not None and e.get("end") is not None]


def _match_events(
    truth_events: list[tuple[pd.Timestamp, pd.Timestamp]],
    detector_events: list[dict],
    max_early_warning_min: float = 180.0,
) -> tuple[int, list[float], list[float]]:
    """Greedily match detector intervals to reference intervals.

    A detector event counts when its interval overlaps a truth event, or when
    it starts before truth onset but no more than ``max_early_warning_min``
    early and remains active into the truth interval. Each detector event is
    used at most once.
    """
    matched = 0
    onset_deltas: list[float] = []
    lead_times: list[float] = []
    used: set[str] = set()

    for truth_start, truth_end in truth_events:
        candidates = []
        for event in detector_events:
            event_id = event["event_id"]
            if event_id in used:
                continue
            event_start = event["start"]
            event_end = event["end"]
            if event_end < truth_start:
                continue
            early_min = (truth_start - event_start).total_seconds() / 60.0
            if early_min > max_early_warning_min:
                continue
            if event_start > truth_end:
                continue
            overlap_start = max(event_start, truth_start)
            overlap_end = min(event_end, truth_end)
            if overlap_start > overlap_end:
                continue
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            candidates.append((overlap_seconds, event_start, event_id))

        if not candidates:
            continue

        # Prefer the detector interval with the greatest overlap; break ties
        # in favour of the earliest onset so early warnings are preserved.
        _, detector_start, event_id = max(candidates, key=lambda item: (item[0], -item[1].value))
        used.add(event_id)
        matched += 1
        delta = (detector_start - truth_start).total_seconds() / 60.0
        onset_deltas.append(delta)
        if delta < 0:
            lead_times.append(-delta)

    return matched, onset_deltas, lead_times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="live_replay_report.json")
    args = parser.parse_args()

    start = _as_utc_timestamp(args.start_date)
    end = _as_utc_timestamp(args.end_date)
    if end <= start:
        raise SystemExit("--end-date must be after --start-date")

    load_config(args.config)
    detector = LiveDetector.from_pipeline_defaults()
    print(
        "Detector config: "
        f"amplitude_window={detector.config.amplitude_window_min:g}min, "
        f"minor_storm={detector.config.minor_storm_nt:g}nT, "
        f"fast_window={detector.config.fast_window_min:g}min, "
        f"start_debounce={detector.config.event_start_samples} samples"
    )

    print(f"Fetching global validation indices for {start.date()} -> {end.date()} ...")
    kp, dst = fetch_global_indices(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(
        "Global sources: "
        f"Kp={'available' if kp is not None else 'unavailable'}, "
        f"Dst={'available' if dst is not None else 'unavailable'}"
    )

    metrics = ReplayMetrics()
    predictions: dict[str, dict] = {}
    truth_parts: list[pd.Series] = []
    event_intervals: dict[str, dict] = {}
    last_seen_timestamp: Optional[pd.Timestamp] = None

    for chunk_start, chunk_end in _month_windows(start, end):
        days = max(1, int((chunk_end - chunk_start).total_seconds() // 86400))
        print(f"[{args.observatory}] {chunk_start.date()} -> {chunk_end.date()} ...")
        try:
            text = fetch_intermagnet_iaga2002(
                observatory=args.observatory,
                start_date=chunk_start.strftime("%Y-%m-%d"),
                duration_days=days,
                samples_per_day="Minute",
            )
            frame = parse_iaga2002_to_dataframe(text)
            frame = frame.loc[(frame.index >= chunk_start) & (frame.index < chunk_end)]
            frame = frame[~frame.index.duplicated(keep="first")]
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if frame.empty:
            print("  FAILED: no parsed samples")
            continue

        _, _, global_level = align_global_indices(frame.index, kp, dst)
        truth_parts.append(pd.Series(global_level, index=frame.index, dtype=float))

        for timestamp, row in frame.iterrows():
            timestamp_utc = _as_utc_timestamp(timestamp)
            last_seen_timestamp = timestamp_utc
            value = _safe_float(row.get("f_nt"))
            try:
                result = detector.update(timestamp_utc, value)
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": str(exc),
                    "timestamp": timestamp_utc.isoformat(),
                    "event": None,
                }

            status = result.get("status")
            if status == "warming_up":
                metrics.warming_up += 1
            elif status == "invalid":
                metrics.invalid += 1
            elif status == "ok":
                metrics.evaluated += 1
            elif status == "error":
                metrics.errors += 1
            if result.get("gap"):
                metrics.gaps += 1

            kind = _event_type(result)
            if kind == "event_started":
                metrics.events_started += 1
                event = result["event"]
                event_intervals[event["event_id"]] = {
                    "event_id": event["event_id"],
                    "start": timestamp_utc,
                    "end": None,
                    "level": event.get("level"),
                    "trigger": event.get("trigger"),
                }
            elif kind == "event_escalated":
                metrics.events_escalated += 1
            elif kind == "event_ended":
                metrics.events_ended += 1
                event = result["event"]
                if event["event_id"] in event_intervals:
                    event_intervals[event["event_id"]]["end"] = timestamp_utc
                    event_intervals[event["event_id"]]["level"] = event.get("level")

            event = result.get("event") if isinstance(result.get("event"), dict) else None
            record = {
                "timestamp": timestamp_utc.isoformat(),
                "level": result.get("level"),
                "status": status,
                "amplitude_nt": result.get("amplitude_nt"),
                "fast_amplitude_nt": result.get("fast_amplitude_nt"),
                "fast_trigger": result.get("fast_trigger", False),
                "event": event,
            }
            key = timestamp_utc.isoformat()
            if key in predictions:
                metrics.duplicate_samples += 1
            predictions[key] = record

    if truth_parts:
        truth_series = pd.concat(truth_parts).sort_index()
        before = len(truth_series)
        truth_series = truth_series[~truth_series.index.duplicated(keep="first")]
        metrics.duplicate_samples += before - len(truth_series)
        truth_index = truth_series.index
        truth_level = truth_series.to_numpy(dtype=float)
    else:
        truth_index = pd.DatetimeIndex([], tz="UTC")
        truth_level = np.asarray([], dtype=float)

    expected_index = pd.date_range(start=start, end=end - pd.Timedelta(minutes=1), freq="min", tz="UTC")
    expected_set = set(expected_index)
    actual_set = set(truth_index)
    metrics.missing_samples = len(expected_set - actual_set)

    flags = np.array(
        [predictions.get(ts.isoformat(), {}).get("level", "invalid") for ts in truth_index],
        dtype=object,
    )
    valid = np.isfinite(truth_level)
    predicted_storm = np.isin(flags, list(STORM_FLAGS))
    truth_storm = valid & (truth_level >= 3)
    mask = valid & np.isin(flags, list(VALID_LEVELS))

    metrics.tp = int(np.sum(predicted_storm & truth_storm & mask))
    metrics.fn = int(np.sum(~predicted_storm & truth_storm & mask))
    metrics.fp = int(np.sum(predicted_storm & ~truth_storm & mask))
    metrics.tn = int(np.sum(~predicted_storm & ~truth_storm & mask))
    metrics.global_truth = int(valid.sum())

    truth_events = _truth_events(truth_index, truth_level)
    detector_events = _close_event_intervals(event_intervals, last_seen_timestamp)
    metrics.truth_events = len(truth_events)
    metrics.truth_events_detected, metrics.onset_deltas_min, metrics.lead_times_min = _match_events(
        truth_events, detector_events
    )

    sample_metrics = binary_metrics(metrics.tp, metrics.fp, metrics.fn, metrics.tn)
    event_recall = metrics.truth_events_detected / metrics.truth_events if metrics.truth_events else float("nan")
    coverage = metrics.global_truth / len(expected_index) if len(expected_index) else float("nan")
    median_delta = float(np.median(metrics.onset_deltas_min)) if metrics.onset_deltas_min else float("nan")
    p95_delta = float(np.percentile(metrics.onset_deltas_min, 95)) if metrics.onset_deltas_min else float("nan")
    median_lead = float(np.median(metrics.lead_times_min)) if metrics.lead_times_min else float("nan")

    report = {
        "mode": "causal_live_replay",
        "observatory": args.observatory,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "config": str(Path(args.config)),
        "samples": {
            "expected": len(expected_index),
            "unique_truth_samples": len(truth_index),
            "evaluated_by_detector": metrics.evaluated,
            "warming_up": metrics.warming_up,
            "invalid": metrics.invalid,
            "errors": metrics.errors,
            "gaps": metrics.gaps,
            "duplicate_samples": metrics.duplicate_samples,
            "missing_expected_samples": metrics.missing_samples,
            "global_truth_samples": metrics.global_truth,
            "truth_coverage": coverage,
        },
        "sample_metrics": {
            "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn, "tn": metrics.tn, **sample_metrics
        },
        "events": {
            "truth_events": metrics.truth_events,
            "truth_events_detected": metrics.truth_events_detected,
            "truth_event_detection_rate": event_recall,
            "live_event_starts": metrics.events_started,
            "live_event_escalations": metrics.events_escalated,
            "live_event_ends": metrics.events_ended,
            "matched_onset_delta_min_median": median_delta,
            "matched_onset_delta_min_p95": p95_delta,
            "early_warning_lead_min_median": median_lead,
            "detector_event_intervals": detector_events,
        },
    }

    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=True, default=str))
    print("\n=== Causal Live Replay ===")
    print(f"Expected samples   : {len(expected_index):,}")
    print(f"Unique truth       : {len(truth_index):,}")
    print(f"Evaluated          : {metrics.evaluated:,}")
    print(f"Warm-up            : {metrics.warming_up:,}")
    print(f"Invalid            : {metrics.invalid:,}")
    print(f"Duplicates         : {metrics.duplicate_samples:,}")
    print(f"Missing            : {metrics.missing_samples:,}")
    print(f"Detector errors    : {metrics.errors:,}")
    print(f"Truth coverage     : {coverage:.4%}")
    print(f"Precision          : {sample_metrics['precision']:.4%}")
    print(f"Recall             : {sample_metrics['recall']:.4%}")
    print(f"F1                 : {sample_metrics['f1']:.4%}")
    print(f"False alarm rate   : {sample_metrics['false_alarm_rate']:.4%}")
    print(f"Truth events       : {metrics.truth_events}")
    print(f"Events detected    : {metrics.truth_events_detected}")
    print(f"Event recall       : {event_recall:.4%}" if metrics.truth_events else "Event recall       : n/a")
    print(f"Median onset delta : {median_delta:.2f} min" if metrics.onset_deltas_min else "Median onset delta : n/a")
    print(f"P95 onset delta    : {p95_delta:.2f} min" if metrics.onset_deltas_min else "P95 onset delta    : n/a")
    print(f"Median early lead  : {median_lead:.2f} min" if metrics.lead_times_min else "Median early lead  : n/a")
    print(f"Report written to  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
