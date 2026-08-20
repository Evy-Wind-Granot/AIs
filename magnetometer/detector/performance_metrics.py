#!/usr/bin/env python3
"""Comprehensive performance metrics for the deterministic magnetometer detector.

The benchmark evaluates the detector against real INTERMAGNET data and the
GFZ Kp index.  It reports sample-level classification quality, event-level
quality, detection latency, false alarms, coverage, and detector state
stability.  Kp is an external geomagnetic reference, not local ground truth;
results should therefore be interpreted as operational agreement with a
standard global activity index.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from magnetometer_demo import (
    fetch_kp_gfz,
    fetch_intermagnet_iaga2002,
    handle_gaps,
    run_analysis,
)


@dataclass
class BinaryMetrics:
    samples: int
    positive_samples: int
    predicted_positive: int
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    false_alarm_rate: float
    balanced_accuracy: float
    prevalence: float


@dataclass
class EventMetrics:
    reference_events: int
    predicted_events: int
    matched_events: int
    missed_events: int
    false_positive_events: int
    precision: float
    recall: float
    f1: float
    mean_detection_latency_minutes: Optional[float]
    median_detection_latency_minutes: Optional[float]
    max_detection_latency_minutes: Optional[float]


@dataclass
class StabilityMetrics:
    samples: int
    state_changes: int
    state_changes_per_day: float
    flagged_samples: int
    flagged_fraction: float
    longest_flagged_minutes: float
    longest_quiet_minutes: float


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_metrics(reference: np.ndarray, prediction: np.ndarray) -> BinaryMetrics:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if len(ref) != len(pred):
        raise ValueError("reference and prediction lengths differ")

    tp = int(np.sum(ref & pred))
    tn = int(np.sum(~ref & ~pred))
    fp = int(np.sum(~ref & pred))
    fn = int(np.sum(ref & ~pred))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    far = _safe_div(fp, fp + tn)
    balanced = 0.5 * (recall + specificity)

    return BinaryMetrics(
        samples=len(ref),
        positive_samples=int(ref.sum()),
        predicted_positive=int(pred.sum()),
        tp=tp, tn=tn, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        specificity=specificity, false_alarm_rate=far,
        balanced_accuracy=balanced,
        prevalence=_safe_div(int(ref.sum()), len(ref)),
    )


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive runs of True values."""
    x = np.asarray(mask, dtype=bool)
    if not len(x):
        return []
    padded = np.concatenate(([False], x, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_close_events(runs: Iterable[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def event_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    cadence_s: float,
    tolerance_minutes: float = 30.0,
    merge_gap_minutes: float = 10.0,
) -> EventMetrics:
    """Match each reference event to at most one predicted event.

    A prediction is considered a hit when it starts within ``tolerance`` of
    the reference start, or begins while the reference event is active.  This
    avoids rewarding repeated minute-by-minute detections as separate events.
    """
    ref_runs = _merge_close_events(_runs(reference), max(0, int(round(merge_gap_minutes * 60 / cadence_s))))
    pred_runs = _merge_close_events(_runs(prediction), max(0, int(round(merge_gap_minutes * 60 / cadence_s))))
    tolerance = max(0, int(round(tolerance_minutes * 60 / cadence_s)))

    used = set()
    latencies: List[float] = []
    matched = 0

    for rs, re in ref_runs:
        candidates = []
        for j, (ps, pe) in enumerate(pred_runs):
            if j in used:
                continue
            if ps <= re and pe >= rs:
                distance = 0
            else:
                distance = min(abs(ps - re), abs(pe - rs))
            if ps <= re + tolerance and pe >= rs - tolerance:
                candidates.append((distance, j, ps))
        if not candidates:
            continue
        _, j, ps = min(candidates)
        used.add(j)
        matched += 1
        latency_samples = max(0, ps - rs)
        latencies.append(latency_samples * cadence_s / 60.0)

    precision = _safe_div(matched, len(pred_runs))
    recall = _safe_div(matched, len(ref_runs))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return EventMetrics(
        reference_events=len(ref_runs),
        predicted_events=len(pred_runs),
        matched_events=matched,
        missed_events=len(ref_runs) - matched,
        false_positive_events=len(pred_runs) - matched,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_detection_latency_minutes=float(np.mean(latencies)) if latencies else None,
        median_detection_latency_minutes=float(np.median(latencies)) if latencies else None,
        max_detection_latency_minutes=float(np.max(latencies)) if latencies else None,
    )


def stability_metrics(flags: np.ndarray, cadence_s: float) -> StabilityMetrics:
    values = np.asarray(flags, dtype=object)
    active = values != "quiet"
    changes = int(np.sum(values[1:] != values[:-1])) if len(values) > 1 else 0
    flagged_runs = _runs(active)
    quiet_runs = _runs(~active)
    hours = len(values) * cadence_s / 3600.0
    days = hours / 24.0
    return StabilityMetrics(
        samples=len(values),
        state_changes=changes,
        state_changes_per_day=_safe_div(changes, days),
        flagged_samples=int(active.sum()),
        flagged_fraction=_safe_div(int(active.sum()), len(values)),
        longest_flagged_minutes=max((e - s for s, e in flagged_runs), default=0) * cadence_s / 60.0,
        longest_quiet_minutes=max((e - s for s, e in quiet_runs), default=0) * cadence_s / 60.0,
    )


def _align_kp(index: pd.DatetimeIndex, kp: pd.Series) -> pd.Series:
    kp = kp.copy()
    kp.index = pd.to_datetime(kp.index, utc=True)
    return kp.reindex(index, method="ffill", tolerance=pd.Timedelta("6h"))


def evaluate_period(
    observatory: str,
    start_date: str,
    days: int,
    column: str = "f_nt",
    cadence_s: int = 60,
) -> Dict[str, Any]:
    text = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
    )
    df = handle_gaps(
        __import__("magnetometer_demo").parse_iaga2002_to_dataframe(text)[column],
        max_gap_samples=3,
    ).to_frame(column)
    df = df.dropna()
    if len(df) < 100:
        raise RuntimeError(f"Insufficient valid magnetometer samples: {len(df)}")

    result = run_analysis(
        df[column].to_numpy(),
        cadence_s,
        label=f"benchmark {observatory} {start_date}",
        start_time=df.index.min().to_pydatetime(),
    )
    flags = np.asarray(result["flags"], dtype=object)

    kp = fetch_kp_gfz(
        pd.Timestamp(df.index.min()).strftime("%Y-%m-%d"),
        pd.Timestamp(df.index.max()).strftime("%Y-%m-%d"),
    )
    kp_aligned = _align_kp(df.index, kp).to_numpy(dtype=float)
    valid = np.isfinite(kp_aligned)
    if valid.sum() < 100:
        raise RuntimeError("Insufficient aligned Kp samples")

    flags = flags[valid]
    kp_aligned = kp_aligned[valid]

    active_ref = kp_aligned >= 4.0
    storm_ref = kp_aligned >= 5.0
    active_pred = np.isin(flags, ["active", "minor_storm", "major_storm", "severe_storm", "anomaly"])
    storm_pred = np.isin(flags, ["minor_storm", "major_storm", "severe_storm"])

    return {
        "observatory": observatory,
        "start_date": start_date,
        "days": days,
        "column": column,
        "cadence_s": cadence_s,
        "valid_samples": int(valid.sum()),
        "kp_range": [float(np.nanmin(kp_aligned)), float(np.nanmax(kp_aligned))],
        "sample": {
            "active_vs_kp_ge_4": asdict(binary_metrics(active_ref, active_pred)),
            "storm_vs_kp_ge_5": asdict(binary_metrics(storm_ref, storm_pred)),
        },
        "event": {
            "active_vs_kp_ge_4": asdict(event_metrics(active_ref, active_pred, cadence_s)),
            "storm_vs_kp_ge_5": asdict(event_metrics(storm_ref, storm_pred, cadence_s)),
        },
        "stability": asdict(stability_metrics(flags, cadence_s)),
        "flag_counts": {
            str(k): int(v)
            for k, v in zip(*np.unique(flags, return_counts=True))
        },
    }


def _print_section(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.4f}")
        else:
            print(f"{key:32s}: {value}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Comprehensive real-data magnetometer detector benchmark")
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--column", default="f_nt", choices=["x_nt", "y_nt", "z_nt", "f_nt"])
    ap.add_argument("--cadence-s", type=int, default=60)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    report = evaluate_period(args.observatory, args.start_date, args.days, args.column, args.cadence_s)
    print("\n=== MAGNETOMETER DETECTOR PERFORMANCE ===")
    print(f"Observatory: {report['observatory']} | Period: {report['start_date']} | Days: {report['days']}")
    print(f"Valid samples: {report['valid_samples']} | Kp range: {report['kp_range']}")
    _print_section("Sample: Active vs Kp >= 4", report["sample"]["active_vs_kp_ge_4"])
    _print_section("Sample: Storm vs Kp >= 5", report["sample"]["storm_vs_kp_ge_5"])
    _print_section("Event: Active vs Kp >= 4", report["event"]["active_vs_kp_ge_4"])
    _print_section("Event: Storm vs Kp >= 5", report["event"]["storm_vs_kp_ge_5"])
    _print_section("Operational stability", report["stability"])
    print("\nFlag counts:")
    for name, count in report["flag_counts"].items():
        print(f"  {name:16s}: {count}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
