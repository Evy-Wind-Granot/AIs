#!/usr/bin/env python3
"""Run a long-range, offline-aggregated validation benchmark.

The benchmark runs the *production* pipeline on historical INTERMAGNET data,
then aggregates sample-level predictions against the global Kp/Dst indices.
It is intentionally chunked so a multi-year run does not require loading two
years of one-minute magnetometer data into memory at once.

Example:
    python3 validate_historical_magnetometer.py \
        --observatories VIC \
        --start-date 2023-01-01 \
        --end-date 2025-01-01

The default chunk is 30 days with the production pipeline's three-day warmup.
Historical HTTP responses are cached by the production acquisition layer, so
re-running the benchmark after a partial failure does not needlessly redownload
completed historical windows.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

import magnetometer_demo as md
from magnetometer.acquisition import (
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
)
from magnetometer.parsing import parse_iaga2002_to_dataframe

LOCAL_LEVELS = {
    "quiet": 0,
    "unsettled": 1,
    "active": 2,
    "minor_storm": 3,
    "major_storm": 4,
    "severe_storm": 4,
}

SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def global_levels_from_indices(kp_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
    """Return the same 0..4 global severity scale used by MetricsEngine."""
    kp = np.asarray(kp_vals, dtype=float)
    dst = np.asarray(dst_vals, dtype=float)

    with np.errstate(invalid="ignore"):
        kp_level = np.select(
            [kp <= 2, kp <= 4, kp < 6, kp < 8],
            [0.0, 1.0, 2.0, 3.0],
            default=4.0,
        )
        dst_level = np.select(
            [dst >= -10, dst >= -30, dst >= -50, dst >= -100],
            [0.0, 1.0, 2.0, 3.0],
            default=4.0,
        )

    kp_level[~np.isfinite(kp)] = np.nan
    dst_level[~np.isfinite(dst)] = np.nan
    return np.fmax(kp_level, dst_level)


def align_global_indices(
    index: pd.DatetimeIndex,
    kp: Optional[pd.Series],
    dst: Optional[pd.Series],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align Kp/Dst exactly as the production pipeline does."""
    kp_aligned = np.full(len(index), np.nan)
    dst_aligned = np.full(len(index), np.nan)

    if kp is not None:
        kp_aligned = kp.reindex(
            index, method="ffill", tolerance=pd.Timedelta("3h")
        ).to_numpy(dtype=float)
    if dst is not None:
        dst_aligned = dst.reindex(
            index, method="ffill", tolerance=pd.Timedelta("1h")
        ).to_numpy(dtype=float)

    return kp_aligned, dst_aligned, global_levels_from_indices(kp_aligned, dst_aligned)


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute the requested binary validation metrics from confusion counts."""
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    false_alarm_rate = fp / (fp + tn) if fp + tn else float("nan")
    missed_event_rate = fn / (tp + fn) if tp + fn else float("nan")
    detection_rate = recall
    return {
        "detection_rate": detection_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "missed_event_rate": missed_event_rate,
    }


def safe_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    return binary_metrics(
        counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    )


def _empty_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _update_counts(counts: Dict[str, int], local: np.ndarray, global_level: np.ndarray) -> None:
    valid = np.isfinite(global_level)
    pred = np.zeros(len(local), dtype=bool)
    pred[np.isin(local, ["minor_storm", "major_storm", "severe_storm"])] = True
    truth = valid & (global_level >= 3)

    counts["tp"] += int(np.sum(pred & truth))
    counts["fn"] += int(np.sum(~pred & truth))
    counts["fp"] += int(np.sum(pred & valid & ~truth))
    counts["tn"] += int(np.sum(~pred & valid & ~truth))


@dataclass
class Aggregator:
    """Streaming aggregate so the full two-year series stays out of memory."""

    total_samples: int = 0
    global_samples: int = 0
    kp_samples: int = 0
    dst_samples: int = 0
    chunks_ok: int = 0
    chunks_failed: int = 0
    overall: Dict[str, int] = field(default_factory=_empty_counts)
    severity: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "global_storm": _empty_counts(),
            "global_severe": _empty_counts(),
        }
    )
    seasons: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(_empty_counts))
    observatories: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(_empty_counts))
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((5, 5), dtype=np.int64))
    source_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(
        self,
        observatory: str,
        index: pd.DatetimeIndex,
        flags: np.ndarray,
        kp_aligned: np.ndarray,
        dst_aligned: np.ndarray,
        global_level: np.ndarray,
    ) -> None:
        self.total_samples += len(flags)
        valid = np.isfinite(global_level)
        self.global_samples += int(valid.sum())
        self.kp_samples += int(np.isfinite(kp_aligned).sum())
        self.dst_samples += int(np.isfinite(dst_aligned).sum())

        both = valid
        local_levels = np.full(len(flags), np.nan)
        for label, level in LOCAL_LEVELS.items():
            local_levels[flags == label] = level

        for kp_ok, dst_ok in zip(np.isfinite(kp_aligned), np.isfinite(dst_aligned)):
            self.source_counts[
                "Kp+Dst" if kp_ok and dst_ok else "Kp-only" if kp_ok else "Dst-only" if dst_ok else "no-global"
            ] += 1

        _update_counts(self.overall, flags, global_level)

        for key, mask in (
            ("global_storm", both & (global_level >= 3)),
            ("global_severe", both & (global_level >= 4)),
        ):
            # Severity breakdown is primarily recall/detection: every sample
            # in this truth class is a real global event sample.
            if np.any(mask):
                pred = np.isin(flags, ["minor_storm", "major_storm", "severe_storm"])
                self.severity[key]["tp"] += int(np.sum(pred & mask))
                self.severity[key]["fn"] += int(np.sum(~pred & mask))
                self.severity[key]["n"] = self.severity[key].get("n", 0) + int(mask.sum())

        for season in np.unique([SEASONS[int(m)] for m in index.month]):
            mask = np.array([SEASONS[int(m)] == season for m in index.month]) & valid
            if np.any(mask):
                _update_counts(
                    self.seasons[season],
                    flags[mask],
                    global_level[mask],
                )

        self.observatories[observatory]["total"] = self.observatories[observatory].get("total", 0) + len(flags)
        self.observatories[observatory]["global"] = self.observatories[observatory].get("global", 0) + int(valid.sum())
        # Store the same binary confusion counts per observatory.
        local_counts = self.observatories[observatory]
        pred = np.isin(flags, ["minor_storm", "major_storm", "severe_storm"])
        truth = valid & (global_level >= 3)
        local_counts["tp"] = local_counts.get("tp", 0) + int(np.sum(pred & truth))
        local_counts["fn"] = local_counts.get("fn", 0) + int(np.sum(~pred & truth))
        local_counts["fp"] = local_counts.get("fp", 0) + int(np.sum(pred & valid & ~truth))
        local_counts["tn"] = local_counts.get("tn", 0) + int(np.sum(~pred & valid & ~truth))

        # Five-level confusion matrix. Invalid local predictions are excluded;
        # this keeps the matrix scientifically meaningful instead of treating
        # missing samples as quiet predictions.
        local_i = local_levels.astype(np.int64, copy=False)
        global_i = global_level.astype(np.float64, copy=False)
        matrix_mask = valid & np.isfinite(local_i)
        if np.any(matrix_mask):
            np.add.at(
                self.confusion,
                (global_i[matrix_mask].astype(int), local_i[matrix_mask].astype(int)),
                1,
            )

    def report(self) -> Dict[str, Any]:
        def with_metrics(counts: Dict[str, int]) -> Dict[str, Any]:
            payload = dict(counts)
            payload.update(safe_metrics(counts))
            return payload

        severity = {}
        for name, counts in self.severity.items():
            severity[name] = {
                "samples": counts.get("n", 0),
                "detection_rate": (
                    counts.get("tp", 0) / counts.get("n", 0)
                    if counts.get("n", 0)
                    else float("nan")
                ),
            }

        seasons = {name: with_metrics(counts) for name, counts in sorted(self.seasons.items())}
        observatories = {
            name: {
                "total_samples": counts.get("total", 0),
                "global_samples": counts.get("global", 0),
                **with_metrics({k: counts.get(k, 0) for k in ("tp", "fp", "fn", "tn")}),
            }
            for name, counts in sorted(self.observatories.items())
        }

        return {
            "overall": {
                "total_samples": self.total_samples,
                "samples_with_global_data": self.global_samples,
                "global_coverage": self.global_samples / self.total_samples if self.total_samples else float("nan"),
                "kp_samples": self.kp_samples,
                "dst_samples": self.dst_samples,
                "chunks_ok": self.chunks_ok,
                "chunks_failed": self.chunks_failed,
                **with_metrics(self.overall),
            },
            "performance_by_storm_severity": severity,
            "performance_by_season": seasons,
            "performance_by_observatory": observatories,
            "binary_confusion_matrix": {
                "rows_truth": ["quiet_or_nonstorm", "storm"],
                "columns_prediction": ["quiet_or_nonstorm", "storm"],
                "matrix": [
                    [self.overall["tn"], self.overall["fp"]],
                    [self.overall["fn"], self.overall["tp"]],
                ],
            },
            "five_level_confusion_matrix": {
                "rows_truth": ["quiet", "unsettled", "active", "storm", "severe_storm"],
                "columns_prediction": ["quiet", "unsettled", "active", "storm", "severe_storm"],
                "matrix": self.confusion.tolist(),
            },
            "global_source_samples": dict(sorted(self.source_counts.items())),
        }


def fetch_global_indices(start_date: str, end_date: str) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Fetch the full Kp/Dst validation window once, rather than per chunk."""
    kp: Optional[pd.Series]
    try:
        kp = fetch_kp_gfz(start_date, end_date)
    except Exception as exc:
        print(f"WARNING: Kp unavailable: {exc}")
        kp = None

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    months = []
    cursor = start.replace(day=1)
    while cursor < end:
        months.append((cursor.year, cursor.month))
        cursor += pd.DateOffset(months=1)

    dst_parts = []
    for year, month in months:
        part = fetch_dst_kyoto(year, month)
        if part is not None:
            dst_parts.append(part)
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    return kp, dst


def run_observatory(
    observatory: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    warmup_days: float,
    kp: Optional[pd.Series],
    dst: Optional[pd.Series],
    aggregate: Aggregator,
    column: str,
) -> None:
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        days = int((chunk_end - current).total_seconds() // 86400)
        fetch_start = (current - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        duration = int(days + warmup_days)
        print(f"[{observatory}] {current.date()} -> {chunk_end.date()} ...", flush=True)

        try:
            text = fetch_intermagnet_iaga2002(
                observatory=observatory,
                start_date=fetch_start,
                duration_days=duration,
            )
            df = parse_iaga2002_to_dataframe(text)
            if df is None or df.empty or column not in df:
                raise RuntimeError("No usable magnetometer data returned")

            full_index = pd.date_range(
                start=df.index.min(),
                periods=len(df),
                freq=pd.Timedelta(seconds=60),
                tz="UTC",
            )
            analysis_mask = full_index >= current
            analysis_index = full_index[analysis_mask]

            md.setup_logging(level=logging.WARNING)
            result = md.run_analysis(
                df[column].to_numpy(),
                60,
                label=f"{observatory} {current.date()}",
                start_time=pd.to_datetime(df.index.min()).to_pydatetime(),
                analysis_start_time=current.to_pydatetime(),
                dst_series=dst,
                kp_series=kp,
                observatory=observatory,
            )
            if result.get("status") != "ok":
                raise RuntimeError(f"pipeline status={result.get('status')}")

            flags = np.asarray(result["flags"], dtype=object)
            analysis_index = analysis_index[: len(flags)]
            if len(analysis_index) != len(flags):
                raise RuntimeError("Could not reconstruct the pipeline analysis time grid")

            kp_aligned, dst_aligned, global_level = align_global_indices(
                analysis_index, kp, dst
            )
            aggregate.add(
                observatory,
                analysis_index,
                flags,
                kp_aligned,
                dst_aligned,
                global_level,
            )
            aggregate.chunks_ok += 1
        except Exception as exc:
            aggregate.chunks_failed += 1
            print(f"  FAILED: {exc}", flush=True)

        current = chunk_end


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
    ap.add_argument("--observatories", default="VIC", help="Comma-separated IAGA observatory codes")
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2025-01-01", help="Exclusive end date")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--warmup-days", type=float, default=3.0)
    ap.add_argument("--column", default="x_nt")
    ap.add_argument(
        "--output",
        default="historical_validation_report.json",
        help="JSON report path",
    )
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

    md.setup_logging(level=logging.WARNING)
    print(f"Fetching global validation indices for {start.date()} -> {end.date()} ...")
    kp, dst = fetch_global_indices(args.start_date, (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    print(
        "Global sources: "
        f"Kp={'available' if kp is not None and not kp.empty else 'unavailable'}, "
        f"Dst={'available' if dst is not None and not dst.empty else 'unavailable'}"
    )

    aggregate = Aggregator()
    for observatory in observatories:
        run_observatory(
            observatory,
            start,
            end,
            args.chunk_days,
            args.warmup_days,
            kp,
            dst,
            aggregate,
            args.column,
        )

    report = {
        "benchmark": {
            "observatories": observatories,
            "start_date": args.start_date,
            "end_date_exclusive": args.end_date,
            "chunk_days": args.chunk_days,
            "warmup_days": args.warmup_days,
            "column": args.column,
            "definition": "sample-level binary storm validation; global storm = global severity >= 3, local storm = minor/major/severe",
        },
        "results": aggregate.report(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2) + "\n")

    overall = report["results"]["overall"]
    print("\n=== Historical Validation ===")
    print(f"Window: {args.start_date} -> {args.end_date} (exclusive)")
    print(f"Observatories: {', '.join(observatories)}")
    print(f"Chunks: {overall['chunks_ok']} OK / {overall['chunks_failed']} failed")
    print(f"Samples: {overall['total_samples']:,}")
    print(f"Samples with global data: {overall['samples_with_global_data']:,}")
    for key in ("detection_rate", "precision", "recall", "f1", "false_alarm_rate", "missed_event_rate"):
        value = overall[key]
        print(f"{key:20s}: {value:.4%}" if value is not None else f"{key:20s}: N/A")

    print("\nBinary confusion matrix [truth rows x prediction columns]:")
    for row in report["results"]["binary_confusion_matrix"]["matrix"]:
        print(f"  {row}")

    print(f"\nReport written to: {output}")
    return 0 if overall["chunks_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
