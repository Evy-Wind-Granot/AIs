#!/usr/bin/env python3
"""Replay historical magnetometer samples through the causal live detector.

This is deliberately different from the batch validation benchmark: samples
are fed to ``LiveDetector.update`` one at a time, in timestamp order, with no
future observations available to the detector. Global Kp/Dst data is used only
after local predictions have been produced, for scoring.

Example:
    python3 replay_live_detector.py \
        --observatory VIC \
        --start-date 2024-01-01 \
        --end-date 2025-01-01 \
        --config calibrated_vic.json
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
from validate_historical_magnetometer import (
    align_global_indices,
    binary_metrics,
    fetch_global_indices,
)

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
    events_started: int = 0
    events_escalated: int = 0
    events_ended: int = 0
    truth_events: int = 0
    truth_events_detected: int = 0
    latencies_min: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latencies_min is None:
            self.latencies_min = []


def _month_windows(start: pd.Timestamp, end: pd.Timestamp):
    """Yield non-overlapping calendar-month windows clipped to [start, end)."""
    cursor = start
    while cursor < end:
        month_end = cursor.normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        nxt = min(month_end, end)
        yield cursor, nxt
        cursor = nxt


def _truth_events(
    index: pd.DatetimeIndex, global_level: np.ndarray
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous global storm intervals."""
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
    """Safely extract an event type from a detector result.

    ``LiveDetector`` intentionally uses ``event=None`` when no lifecycle
    transition occurs. Replay/scoring code must treat that as an ordinary
    result rather than assuming every response contains a mapping.
    """
    event = result.get("event")
    return event.get("type") if isinstance(event, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="live_replay_report.json")
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp(args.end_date, tz="UTC")
    if end <= start:
        raise SystemExit("--end-date must be after --start-date")

    # Strict loading also applies the calibrated values to the compatibility
    # layer used by LiveDetector.from_pipeline_defaults().
    load_config(args.config)
    detector = LiveDetector.from_pipeline_defaults()

    print(f"Fetching global validation indices for {start.date()} -> {end.date()} ...")
    kp, dst = fetch_global_indices(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(
        "Global sources: "
        f"Kp={'available' if kp is not None else 'unavailable'}, "
        f"Dst={'available' if dst is not None else 'unavailable'}"
    )

    metrics = ReplayMetrics()
    predictions: list[dict] = []
    truth_index_parts: list[pd.DatetimeIndex] = []
    truth_level_parts: list[np.ndarray] = []

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
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if frame.empty:
            print("  FAILED: no parsed samples")
            continue

        # F is the scalar total-field measurement used by the existing VIC
        # historical pipeline. Each sample is processed immediately.
        for timestamp, row in frame.iterrows():
            value = _safe_float(row.get("f_nt"))
            try:
                result = detector.update(timestamp, value)
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": str(exc),
                    "timestamp": timestamp.isoformat(),
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
            elif kind == "event_escalated":
                metrics.events_escalated += 1
            elif kind == "event_ended":
                metrics.events_ended += 1

            # Normalize absent events to None in the report. Do not assume the
            # detector returns a mapping for every sample.
            event = result.get("event")
            if not isinstance(event, dict):
                event = None

            predictions.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "level": result.get("level"),
                    "status": status,
                    "amplitude_nt": result.get("amplitude_nt"),
                    "event": event,
                }
            )

        # Keep the complete truth index separate; scoring is performed only
        # after all local predictions have been generated.
        _, _, global_level = align_global_indices(frame.index, kp, dst)
        truth_index_parts.append(frame.index)
        truth_level_parts.append(global_level)

    if truth_index_parts:
        truth_index = truth_index_parts[0].append(truth_index_parts[1:])
        truth_level = np.concatenate(truth_level_parts)
    else:
        truth_index = pd.DatetimeIndex([], tz="UTC")
        truth_level = np.asarray([], dtype=float)

    # Predictions are ordered by ingestion. Build arrays for exact timestamp
    # alignment with the global truth series.
    pred_by_ts = {p["timestamp"]: p for p in predictions}
    flags = np.array(
        [pred_by_ts.get(ts.isoformat(), {}).get("level", "invalid") for ts in truth_index],
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
    metrics.truth_events = len(truth_events)

    # Match a truth event to the first event_started at or after its beginning
    # and before its end. This measures live detection latency without giving
    # the detector access to future truth labels.
    starts = [
        pd.Timestamp(p["timestamp"], tz="UTC")
        for p in predictions
        if _event_type(p) == "event_started"
    ]
    for event_start, event_end in truth_events:
        candidates = [s for s in starts if event_start <= s <= event_end]
        if candidates:
            first = min(candidates)
            metrics.truth_events_detected += 1
            metrics.latencies_min.append((first - event_start).total_seconds() / 60.0)

    sample_metrics = binary_metrics(metrics.tp, metrics.fp, metrics.fn, metrics.tn)
    event_recall = (
        metrics.truth_events_detected / metrics.truth_events
        if metrics.truth_events
        else float("nan")
    )
    report = {
        "mode": "causal_live_replay",
        "observatory": args.observatory,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "config": str(Path(args.config)),
        "samples": {
            "evaluated_by_detector": metrics.evaluated,
            "warming_up": metrics.warming_up,
            "invalid": metrics.invalid,
            "errors": metrics.errors,
            "gaps": metrics.gaps,
            "global_truth_samples": metrics.global_truth,
        },
        "sample_metrics": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "tn": metrics.tn,
            **sample_metrics,
        },
        "events": {
            "truth_events": metrics.truth_events,
            "truth_events_detected": metrics.truth_events_detected,
            "truth_event_detection_rate": event_recall,
            "live_event_starts": metrics.events_started,
            "live_event_escalations": metrics.events_escalated,
            "live_event_ends": metrics.events_ended,
            "detection_latency_min_median": (
                float(np.median(metrics.latencies_min))
                if metrics.latencies_min
                else float("nan")
            ),
            "detection_latency_min_p95": (
                float(np.percentile(metrics.latencies_min, 95))
                if metrics.latencies_min
                else float("nan")
            ),
        },
    }

    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=True))
    print("\n=== Causal Live Replay ===")
    print(f"Samples evaluated : {metrics.evaluated:,}")
    print(f"Warm-up samples   : {metrics.warming_up:,}")
    print(f"Invalid samples   : {metrics.invalid:,}")
    print(f"Detector errors   : {metrics.errors:,}")
    print(f"Global truth      : {metrics.global_truth:,}")
    print(f"Precision         : {sample_metrics['precision']:.4%}")
    print(f"Recall            : {sample_metrics['recall']:.4%}")
    print(f"F1                : {sample_metrics['f1']:.4%}")
    print(f"False alarm rate  : {sample_metrics['false_alarm_rate']:.4%}")
    print(f"Truth events      : {metrics.truth_events}")
    print(f"Events detected   : {metrics.truth_events_detected}")
    print(f"Event recall      : {event_recall:.4%}" if metrics.truth_events else "Event recall      : n/a")
    print(
        f"Median latency    : {report['events']['detection_latency_min_median']:.2f} min"
        if metrics.latencies_min
        else "Median latency    : n/a"
    )
    print(
        f"P95 latency       : {report['events']['detection_latency_min_p95']:.2f} min"
        if metrics.latencies_min
        else "P95 latency       : n/a"
    )
    print(f"Report written to : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
