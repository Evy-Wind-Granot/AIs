#!/usr/bin/env python3
"""Run a long-range production magnetometer validation benchmark.

The benchmark runs the production pipeline on historical INTERMAGNET data and
compares local activity flags with globally derived Kp/Dst severity labels.
Results are streamed so multi-year one-minute data never has to reside in memory
at once. Failed data-quality chunks are excluded from scored samples and are
reported explicitly rather than silently disappearing from the denominator.

Important: global indices are used only for scoring. They are NOT passed into
run_analysis, so the local baseline/classifier cannot see the truth labels it is
being evaluated against.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from magnetometer.demos import magnetometer_demo as md
from magnetometer.acquisition import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz
from magnetometer.parsing import parse_iaga2002_to_dataframe

STORM_FLAGS = np.array(["minor_storm", "major_storm", "severe_storm"], dtype=object)
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def global_levels_from_indices(kp_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
    kp = np.asarray(kp_vals, dtype=float)
    dst = np.asarray(dst_vals, dtype=float)
    with np.errstate(invalid="ignore"):
        kp_level = np.select([kp <= 2, kp <= 4, kp < 6, kp < 8], [0.0, 1.0, 2.0, 3.0], default=4.0)
        dst_level = np.select([dst >= -10, dst >= -30, dst >= -50, dst >= -100], [0.0, 1.0, 2.0, 3.0], default=4.0)
    kp_level[~np.isfinite(kp)] = np.nan
    dst_level[~np.isfinite(dst)] = np.nan
    return np.fmax(kp_level, dst_level)


def align_global_indices(index: pd.DatetimeIndex, kp: Optional[pd.Series], dst: Optional[pd.Series]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp_aligned = np.full(len(index), np.nan)
    dst_aligned = np.full(len(index), np.nan)
    if kp is not None:
        kp_aligned = kp.reindex(index, method="ffill", tolerance=pd.Timedelta("3h")).to_numpy(dtype=float)
    if dst is not None:
        dst_aligned = dst.reindex(index, method="ffill", tolerance=pd.Timedelta("1h")).to_numpy(dtype=float)
    return kp_aligned, dst_aligned, global_levels_from_indices(kp_aligned, dst_aligned)


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else float("nan")
    false_alarm_rate = fp / (fp + tn) if fp + tn else float("nan")
    missed_event_rate = fn / (tp + fn) if tp + fn else float("nan")
    return {"detection_rate": recall, "precision": precision, "recall": recall, "f1": f1, "false_alarm_rate": false_alarm_rate, "missed_event_rate": missed_event_rate}


def _empty_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _update_counts(counts: Dict[str, int], flags: np.ndarray, global_level: np.ndarray) -> None:
    valid = np.isfinite(global_level)
    pred = np.isin(flags, STORM_FLAGS)
    truth = valid & (global_level >= 3)
    counts["tp"] += int(np.sum(pred & truth))
    counts["fn"] += int(np.sum(~pred & truth))
    counts["fp"] += int(np.sum(pred & valid & ~truth))
    counts["tn"] += int(np.sum(~pred & valid & ~truth))


def _metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    return {**counts, **binary_metrics(counts["tp"], counts["fp"], counts["fn"], counts["tn"])}


@dataclass
class EventState:
    start: Optional[pd.Timestamp] = None
    last: Optional[pd.Timestamp] = None
    samples: int = 0
    detected: bool = False
    max_level: int = 0


@dataclass
class Aggregator:
    """Streaming aggregate for sample-, severity-, seasonal-, and event-level metrics."""

    requested_samples: int = 0
    evaluated_samples: int = 0
    global_samples: int = 0
    kp_samples: int = 0
    dst_samples: int = 0
    chunks_ok: int = 0
    chunks_failed: int = 0
    overall: Dict[str, int] = field(default_factory=_empty_counts)
    by_season: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(_empty_counts))
    by_month: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(_empty_counts))
    by_observatory: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(_empty_counts))
    by_source: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(_empty_counts))
    severity: Dict[str, Dict[str, int]] = field(default_factory=lambda: {"storm": {"samples": 0, "detected": 0, "missed": 0}, "severe_storm": {"samples": 0, "detected": 0, "missed": 0}})
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((5, 5), dtype=np.int64))
    source_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    event_states: Dict[str, EventState] = field(default_factory=dict)
    completed_events: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"total": 0, "detected": 0, "missed": 0, "severe_total": 0, "severe_detected": 0, "severe_missed": 0}))

    def _finish_event(self, observatory: str) -> None:
        state = self.event_states.get(observatory)
        if state is None or state.start is None:
            return
        out = self.completed_events[observatory]
        out["total"] += 1
        out["detected"] += int(state.detected)
        out["missed"] += int(not state.detected)
        if state.max_level >= 4:
            out["severe_total"] += 1
            out["severe_detected"] += int(state.detected)
            out["severe_missed"] += int(not state.detected)
        self.event_states[observatory] = EventState()

    def _update_events(self, observatory: str, index: pd.DatetimeIndex, flags: np.ndarray, global_level: np.ndarray) -> None:
        state = self.event_states.setdefault(observatory, EventState())
        cadence = pd.Timedelta(minutes=1)
        for ts, flag, level in zip(index, flags, global_level):
            if not np.isfinite(level):
                self._finish_event(observatory)
                state = self.event_states[observatory]
                continue
            if level < 3:
                self._finish_event(observatory)
                state = self.event_states[observatory]
                continue
            if state.start is None or state.last is None or ts - state.last > cadence:
                self._finish_event(observatory)
                state = self.event_states[observatory]
                state.start = ts
            state.last = ts
            state.samples += 1
            state.detected = state.detected or bool(flag in STORM_FLAGS)
            state.max_level = max(state.max_level, int(level))

    def add(self, observatory: str, index: pd.DatetimeIndex, flags: np.ndarray, kp_aligned: np.ndarray, dst_aligned: np.ndarray, global_level: np.ndarray) -> None:
        self.evaluated_samples += len(flags)
        valid = np.isfinite(global_level)
        self.global_samples += int(valid.sum())
        self.kp_samples += int(np.isfinite(kp_aligned).sum())
        self.dst_samples += int(np.isfinite(dst_aligned).sum())
        _update_counts(self.overall, flags, global_level)

        source_masks = {
            "Kp+Dst": np.isfinite(kp_aligned) & np.isfinite(dst_aligned),
            "Kp-only": np.isfinite(kp_aligned) & ~np.isfinite(dst_aligned),
            "Dst-only": ~np.isfinite(kp_aligned) & np.isfinite(dst_aligned),
        }
        for name, mask in source_masks.items():
            if np.any(mask):
                _update_counts(self.by_source[name], flags[mask], global_level[mask])
                self.source_counts[name] += int(mask.sum())
        no_global = ~(np.isfinite(kp_aligned) | np.isfinite(dst_aligned))
        self.source_counts["no-global"] += int(no_global.sum())

        for month in sorted(set(index.month)):
            mask = (index.month == month) & valid
            if np.any(mask):
                _update_counts(self.by_month[f"{int(month):02d}"], flags[mask], global_level[mask])
        for season in sorted(set(SEASONS[int(m)] for m in index.month)):
            mask = np.array([SEASONS[int(m)] == season for m in index.month]) & valid
            if np.any(mask):
                _update_counts(self.by_season[season], flags[mask], global_level[mask])

        local = self.by_observatory[observatory]
        local["total"] = local.get("total", 0) + len(flags)
        local["global"] = local.get("global", 0) + int(valid.sum())
        _update_counts(local, flags, global_level)

        for name, mask in (("storm", valid & (global_level >= 3)), ("severe_storm", valid & (global_level >= 4))):
            if np.any(mask):
                pred = np.isin(flags, STORM_FLAGS)
                self.severity[name]["samples"] += int(mask.sum())
                self.severity[name]["detected"] += int(np.sum(pred & mask))
                self.severity[name]["missed"] += int(np.sum(~pred & mask))

        local_level = np.full(len(flags), np.nan)
        for label, level in (("quiet", 0), ("unsettled", 1), ("active", 2), ("minor_storm", 3), ("major_storm", 4), ("severe_storm", 4)):
            local_level[flags == label] = level
        matrix_mask = valid & np.isfinite(local_level)
        if np.any(matrix_mask):
            truth_i = global_level[matrix_mask].astype(np.int64)
            pred_i = local_level[matrix_mask].astype(np.int64)
            in_range = (truth_i >= 0) & (truth_i < 5) & (pred_i >= 0) & (pred_i < 5)
            if np.any(in_range):
                np.add.at(self.confusion, (truth_i[in_range], pred_i[in_range]), 1)

        self._update_events(observatory, index, flags, global_level)

    def finish(self) -> None:
        for observatory in list(self.event_states):
            self._finish_event(observatory)

    def report(self) -> Dict[str, Any]:
        self.finish()
        event_totals = {"total": 0, "detected": 0, "missed": 0, "severe_total": 0, "severe_detected": 0, "severe_missed": 0}
        events_by_obs = {}
        for obs, counts in sorted(self.completed_events.items()):
            events_by_obs[obs] = dict(counts)
            for key in event_totals:
                event_totals[key] += counts[key]
        event_metrics = {
            "global_events": event_totals["total"],
            "events_detected": event_totals["detected"],
            "events_missed": event_totals["missed"],
            "event_detection_rate": event_totals["detected"] / event_totals["total"] if event_totals["total"] else float("nan"),
            "severe_global_events": event_totals["severe_total"],
            "severe_events_detected": event_totals["severe_detected"],
            "severe_events_missed": event_totals["severe_missed"],
            "severe_event_detection_rate": event_totals["severe_detected"] / event_totals["severe_total"] if event_totals["severe_total"] else float("nan"),
            "by_observatory": events_by_obs,
        }
        severity = {}
        for name, c in self.severity.items():
            severity[name] = {**c, "detection_rate": c["detected"] / c["samples"] if c["samples"] else float("nan"), "missed_event_rate": c["missed"] / c["samples"] if c["samples"] else float("nan")}
        obs = {}
        for name, c in sorted(self.by_observatory.items()):
            obs[name] = {"total_samples": c.get("total", 0), "global_samples": c.get("global", 0), **_metrics({k: c.get(k, 0) for k in ("tp", "fp", "fn", "tn")})}
        return {
            "coverage": {
                "requested_samples": self.requested_samples,
                "evaluated_samples": self.evaluated_samples,
                "excluded_samples": max(0, self.requested_samples - self.evaluated_samples),
                "evaluation_coverage": self.evaluated_samples / self.requested_samples if self.requested_samples else float("nan"),
                "global_truth_coverage_of_evaluated": self.global_samples / self.evaluated_samples if self.evaluated_samples else float("nan"),
                "chunks_ok": self.chunks_ok,
                "chunks_failed": self.chunks_failed,
            },
            "overall": {**_metrics(self.overall), "total_evaluated_samples": self.evaluated_samples, "samples_with_global_data": self.global_samples, "kp_samples": self.kp_samples, "dst_samples": self.dst_samples},
            "performance_by_storm_severity": severity,
            "performance_by_season": {k: _metrics(v) for k, v in sorted(self.by_season.items())},
            "performance_by_month": {k: _metrics(v) for k, v in sorted(self.by_month.items())},
            "performance_by_observatory": obs,
            "performance_by_global_source": {k: _metrics(v) for k, v in sorted(self.by_source.items()) if sum(v.values()) > 0},
            "event_level_performance": event_metrics,
            "binary_confusion_matrix": {"rows_truth": ["quiet_or_nonstorm", "storm"], "columns_prediction": ["quiet_or_nonstorm", "storm"], "matrix": [[self.overall["tn"], self.overall["fp"]], [self.overall["fn"], self.overall["tp"]]]},
            "five_level_confusion_matrix": {"rows_truth": ["quiet", "unsettled", "active", "storm", "severe_storm"], "columns_prediction": ["quiet", "unsettled", "active", "storm", "severe_storm"], "matrix": self.confusion.tolist()},
            "global_source_samples": dict(sorted(self.source_counts.items())),
        }


def fetch_global_indices(start_date: str, end_date: str) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    try:
        kp = fetch_kp_gfz(start_date, end_date)
    except Exception as exc:
        print(f"WARNING: Kp unavailable: {exc}")
        kp = None
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    cursor = start.replace(day=1)
    dst_parts = []
    while cursor < end:
        part = fetch_dst_kyoto(cursor.year, cursor.month)
        if part is not None:
            dst_parts.append(part)
        cursor += pd.DateOffset(months=1)
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    return kp, dst


def run_observatory(observatory: str, start: pd.Timestamp, end: pd.Timestamp, chunk_days: int, warmup_days: float, kp: Optional[pd.Series], dst: Optional[pd.Series], aggregate: Aggregator, column: str) -> None:
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        days = int((chunk_end - current).total_seconds() // 86400)
        fetch_start = (current - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        duration = int(days + warmup_days)
        print(f"[{observatory}] {current.date()} -> {chunk_end.date()} ...", flush=True)
        try:
            text = fetch_intermagnet_iaga2002(observatory=observatory, start_date=fetch_start, duration_days=duration)
            df = parse_iaga2002_to_dataframe(text)
            if df is None or df.empty or column not in df:
                raise RuntimeError("No usable magnetometer data returned")
            full_index = pd.date_range(start=df.index.min(), periods=len(df), freq=pd.Timedelta(seconds=60), tz="UTC")
            md.setup_logging(level=logging.WARNING)
            # IMPORTANT: do not pass Kp/Dst into the production pipeline during
            # validation. They are reserved for independent truth scoring below.
            result = md.run_analysis(
                df[column].to_numpy(),
                60,
                label=f"{observatory} {current.date()}",
                start_time=pd.to_datetime(df.index.min()).to_pydatetime(),
                analysis_start_time=current.to_pydatetime(),
                dst_series=None,
                kp_series=None,
                observatory=observatory,
            )
            if result.get("status") != "ok":
                raise RuntimeError(f"pipeline status={result.get('status')}")
            flags = np.asarray(result["flags"], dtype=object)
            analysis_index = full_index[full_index >= current][: len(flags)]
            if len(analysis_index) != len(flags):
                raise RuntimeError("Could not reconstruct the pipeline analysis time grid")
            kp_aligned, dst_aligned, global_level = align_global_indices(analysis_index, kp, dst)
            aggregate.add(observatory, analysis_index, flags, kp_aligned, dst_aligned, global_level)
            aggregate.chunks_ok += 1
        except Exception as exc:
            aggregate.chunks_failed += 1
            print(f"  FAILED: {exc}", flush=True)
        current = chunk_end


def current_classification_settings() -> Dict[str, Any]:
    """Snapshot of the classification knobs the pipeline is actually using."""
    return {
        "FLAG_AMPLITUDE_WINDOW_MIN": float(md.FLAG_AMPLITUDE_WINDOW_MIN),
        "FLAG_AMPLITUDE_MODE": str(md.FLAG_AMPLITUDE_MODE),
        "FLAG_AMPLITUDE_CENTERED": bool(md.FLAG_AMPLITUDE_CENTERED),
        "FLAG_THRESHOLD_UNSETTLED_NT": float(md.FLAG_THRESHOLD_UNSETTLED_NT),
        "FLAG_THRESHOLD_ACTIVE_NT": float(md.FLAG_THRESHOLD_ACTIVE_NT),
        "FLAG_THRESHOLD_MINOR_STORM_NT": float(md.FLAG_THRESHOLD_MINOR_STORM_NT),
        "FLAG_THRESHOLD_MAJOR_STORM_NT": float(md.FLAG_THRESHOLD_MAJOR_STORM_NT),
        "FLAG_THRESHOLD_SEVERE_STORM_NT": float(md.FLAG_THRESHOLD_SEVERE_STORM_NT),
        "FLAG_THRESHOLD_ANOMALY_JUMP_NT": float(md.FLAG_THRESHOLD_ANOMALY_JUMP_NT),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observatories", default="VIC")
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2025-01-01", help="Exclusive end date")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--warmup-days", type=float, default=3.0)
    ap.add_argument("--column", default="x_nt")
    ap.add_argument("--output", default="historical_validation_report.json")
    ap.add_argument("--config", default=None, help="Optional JSON/YAML production config to load before validation")
    args = ap.parse_args()
    start = pd.to_datetime(args.start_date, utc=True)
    end = pd.to_datetime(args.end_date, utc=True)
    if end <= start:
        ap.error("--end-date must be after --start-date")
    if args.chunk_days < 1:
        ap.error("--chunk-days must be >= 1")
    observatories = [x.strip().upper() for x in args.observatories.split(",") if x.strip()]
    if not observatories:
        ap.error("At least one observatory is required")
    if args.config:
        md.load_config(args.config)
    md.setup_logging(level=logging.WARNING)
    loaded_settings = current_classification_settings()
    print("Classification settings in effect:")
    for k, v in loaded_settings.items():
        print(f"  {k}: {v}")
    print(f"Fetching global validation indices for {start.date()} -> {end.date()} ...")
    kp, dst = fetch_global_indices(args.start_date, (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    print(f"Global sources: Kp={'available' if kp is not None and not kp.empty else 'unavailable'}, Dst={'available' if dst is not None and not dst.empty else 'unavailable'}")
    aggregate = Aggregator(requested_samples=len(observatories) * int((end - start).total_seconds() // 60))
    for observatory in observatories:
        run_observatory(observatory, start, end, args.chunk_days, args.warmup_days, kp, dst, aggregate, args.column)
    report = {
        "benchmark": {
            "observatories": observatories,
            "start_date": args.start_date,
            "end_date_exclusive": args.end_date,
            "chunk_days": args.chunk_days,
            "warmup_days": args.warmup_days,
            "column": args.column,
            "config": args.config,
            "loaded_classification_settings": loaded_settings,
            "validation_mode": "local_only_no_global_indices",
            "definition": "sample-level binary storm validation; global storm = severity >= 3; local storm = minor/major/severe; event = contiguous global storm samples at one-minute cadence",
        },
        "results": aggregate.report(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    overall = report["results"]["overall"]
    coverage = report["results"]["coverage"]
    print("\n=== Historical Validation ===")
    print(f"Window: {args.start_date} -> {args.end_date} (exclusive)")
    print(f"Observatories: {', '.join(observatories)}")
    print(f"Chunks: {coverage['chunks_ok']} OK / {coverage['chunks_failed']} failed")
    print(f"Requested samples: {coverage['requested_samples']:,}")
    print(f"Evaluated samples: {coverage['evaluated_samples']:,}")
    print(f"Excluded samples: {coverage['excluded_samples']:,} ({1.0 - coverage['evaluation_coverage']:.2%})")
    print(f"Evaluation coverage: {coverage['evaluation_coverage']:.2%}")
    print(f"Global truth coverage of evaluated samples: {coverage['global_truth_coverage_of_evaluated']:.2%}")
    for key in ("detection_rate", "precision", "recall", "f1", "false_alarm_rate", "missed_event_rate"):
        value = overall[key]
        print(f"{key:20s}: {value:.4%}" if np.isfinite(value) else f"{key:20s}: N/A")
    events = report["results"]["event_level_performance"]
    print(f"Event detection rate: {events['event_detection_rate']:.2%}" if np.isfinite(events["event_detection_rate"]) else "Event detection rate: N/A")
    print("Binary confusion matrix [truth rows x prediction columns]:")
    for row in report["results"]["binary_confusion_matrix"]["matrix"]:
        print(f"  {row}")
    print(f"\nReport written to: {output}")
    return 0 if coverage["chunks_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
