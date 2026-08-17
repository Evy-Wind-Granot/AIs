#!/usr/bin/env python3
"""Calibrate storm-detection settings on a historical training window.

The calibration run deliberately does NOT pass Kp/Dst into the production
pipeline. Global indices are used only after local processing, as independent
truth labels. This prevents the baseline fit from seeing the labels it is being
scored against.

The selected settings are written to JSON and, when a holdout window is
provided, evaluated on that unseen period in the same process. Do not use the
holdout interval for calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import magnetometer_demo as md
from magnetometer.acquisition import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz
from magnetometer.parsing import parse_iaga2002_to_dataframe
from magnetometer.classification import disturbance_amplitude
from validate_historical_magnetometer import (
    Aggregator,
    align_global_indices,
    fetch_global_indices,
    global_levels_from_indices,
    run_observatory,
)

MODES = ("range", "hybrid", "max")


def score_threshold(amplitude: np.ndarray, truth: np.ndarray, threshold: float) -> Dict[str, Any]:
    valid = np.isfinite(amplitude) & np.isfinite(truth)
    pred = amplitude >= threshold
    truth_storm = truth >= 3
    tp = int(np.sum(valid & pred & truth_storm))
    fp = int(np.sum(valid & pred & ~truth_storm))
    fn = int(np.sum(valid & ~pred & truth_storm))
    tn = int(np.sum(valid & ~pred & ~truth_storm))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    far = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "missed_event_rate": fn / (tp + fn) if tp + fn else float("nan"),
        "evaluated_samples": int(valid.sum()),
    }


def candidate_thresholds(amplitude: np.ndarray, count: int = 160) -> np.ndarray:
    finite = amplitude[np.isfinite(amplitude)]
    if finite.size == 0:
        return np.array([], dtype=float)
    values = np.unique(np.quantile(finite, np.linspace(0.05, 0.995, count)))
    return values[values > 0]


def choose_best(
    amplitudes: Dict[Tuple[float, str], np.ndarray],
    truth: np.ndarray,
    max_false_alarm_rate: float,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for (window, mode), amplitude in amplitudes.items():
        for threshold in candidate_thresholds(amplitude):
            metrics = score_threshold(amplitude, truth, float(threshold))
            if not np.isfinite(metrics["f1"]):
                continue
            candidate = {
                "window_min": float(window),
                "mode": mode,
                "threshold_nt": float(threshold),
                "eligible": bool(metrics["false_alarm_rate"] <= max_false_alarm_rate),
                **metrics,
            }
            if best is None:
                best = candidate
                continue
            if candidate["eligible"] and not best["eligible"]:
                best = candidate
            elif candidate["eligible"] == best["eligible"]:
                candidate_key = (candidate["f1"], candidate["recall"], -candidate["false_alarm_rate"])
                best_key = (best["f1"], best["recall"], -best["false_alarm_rate"])
                if candidate_key > best_key:
                    best = candidate
    if best is None:
        raise RuntimeError("No finite calibration candidates were produced")
    best["selection_note"] = (
        "Selected highest-F1 candidate within false-alarm budget."
        if best["eligible"]
        else "No candidate met max_false_alarm_rate; selected highest-F1 candidate."
    )
    return best


def collect_training_data(
    observatory: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    warmup_days: float,
    column: str,
    kp: Optional[pd.Series],
    dst: Optional[pd.Series],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    residual_parts = []
    truth_parts = []
    stats = {
        "chunks_ok": 0,
        "chunks_failed": 0,
        "requested_samples": int((end - start).total_seconds() // 60),
    }
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
                start=df.index.min(), periods=len(df), freq=pd.Timedelta(seconds=60), tz="UTC"
            )
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
            residual = np.asarray(result["residual"], dtype=float)
            analysis_index = full_index[full_index >= current][: len(residual)]
            if len(analysis_index) != len(residual):
                raise RuntimeError("Could not reconstruct the pipeline analysis time grid")
            _, _, truth = align_global_indices(analysis_index, kp, dst)
            valid = np.isfinite(residual) & np.isfinite(truth)
            if np.any(valid):
                residual_parts.append(residual[valid])
                truth_parts.append(truth[valid])
            stats["chunks_ok"] += 1
        except Exception as exc:
            stats["chunks_failed"] += 1
            print(f"  FAILED: {exc}", flush=True)
        current = chunk_end
    if not residual_parts:
        raise RuntimeError("No usable calibration samples were collected")
    return np.concatenate(residual_parts), np.concatenate(truth_parts), stats


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
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2024-01-01", help="Exclusive calibration end date")
    ap.add_argument("--holdout-start", default="2024-01-01")
    ap.add_argument("--holdout-end", default="2025-01-01", help="Exclusive unseen holdout end date")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--warmup-days", type=float, default=3.0)
    ap.add_argument("--column", default="x_nt")
    ap.add_argument("--max-false-alarm-rate", type=float, default=0.01)
    ap.add_argument("--windows", default="60,120,180,240,360")
    ap.add_argument("--output-config", default="calibrated_vic.json")
    ap.add_argument("--output-report", default="calibration_report.json")
    args = ap.parse_args()

    start = pd.to_datetime(args.start_date, utc=True)
    end = pd.to_datetime(args.end_date, utc=True)
    holdout_start = pd.to_datetime(args.holdout_start, utc=True)
    holdout_end = pd.to_datetime(args.holdout_end, utc=True)
    if end <= start:
        ap.error("--end-date must be after --start-date")
    if holdout_end <= holdout_start or holdout_start < end:
        ap.error("holdout must be a later, non-overlapping interval")
    if args.chunk_days < 1:
        ap.error("--chunk-days must be >= 1")
    if not 0 < args.max_false_alarm_rate < 1:
        ap.error("--max-false-alarm-rate must be between 0 and 1")
    windows = tuple(float(x.strip()) for x in args.windows.split(",") if x.strip())
    if not windows or any(x <= 0 for x in windows):
        ap.error("--windows must contain positive numbers")

    observatory = args.observatory.upper()
    md.setup_logging(level=logging.WARNING)
    print(f"Fetching global calibration indices for {start.date()} -> {end.date()} ...")
    kp, dst = fetch_global_indices(args.start_date, (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    print(
        f"Global sources: Kp={'available' if kp is not None and not kp.empty else 'unavailable'}, "
        f"Dst={'available' if dst is not None and not dst.empty else 'unavailable'}"
    )

    residual, truth, train_stats = collect_training_data(
        observatory, start, end, args.chunk_days, args.warmup_days, args.column, kp, dst
    )
    amplitudes = {
        (window, mode): disturbance_amplitude(
            residual,
            60.0,
            window_min=window,
            mode=mode,
            centered=False,
        )
        for window in windows
        for mode in MODES
    }
    best = choose_best(amplitudes, truth, args.max_false_alarm_rate)
    config = {
        "FLAG_AMPLITUDE_WINDOW_MIN": best["window_min"],
        "FLAG_AMPLITUDE_MODE": best["mode"],
        "FLAG_AMPLITUDE_CENTERED": False,
        "FLAG_THRESHOLD_MINOR_STORM_NT": best["threshold_nt"],
    }
    Path(args.output_config).write_text(json.dumps(config, indent=2) + "\n")

    report: Dict[str, Any] = {
        "calibration": {
            "observatory": observatory,
            "start_date": args.start_date,
            "end_date_exclusive": args.end_date,
            "objective": "maximize sample-level storm F1 subject to false_alarm_rate <= max_false_alarm_rate",
            "max_false_alarm_rate": args.max_false_alarm_rate,
            "windows_tested_min": list(windows),
            "modes_tested": list(MODES),
            **train_stats,
            "evaluated_samples": int(len(truth)),
            "global_storm_samples": int(np.sum(truth >= 3)),
            "global_source": "Kp+Dst when available; Kp-only when Dst is unavailable",
            "pipeline_validation_mode": "local_only_no_global_indices",
        },
        "selected": best,
        "config": config,
        "holdout_required": True,
    }

    print("\n=== Calibration Result ===")
    for key in ("window_min", "mode", "threshold_nt", "precision", "recall", "f1", "false_alarm_rate", "missed_event_rate"):
        print(f"{key:20s}: {best[key]}")
    print(f"Config written to: {args.output_config}")

    if holdout_end > holdout_start:
        md.load_config(args.output_config)
        holdout_kp, holdout_dst = fetch_global_indices(
            args.holdout_start,
            (holdout_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        aggregate = Aggregator(
            requested_samples=int((holdout_end - holdout_start).total_seconds() // 60)
        )
        run_observatory(
            observatory,
            holdout_start,
            holdout_end,
            args.chunk_days,
            args.warmup_days,
            holdout_kp,
            holdout_dst,
            aggregate,
            args.column,
        )
        holdout_results = aggregate.report()
        report["holdout"] = holdout_results
        overall = holdout_results["overall"]
        coverage = holdout_results["coverage"]
        print("\n=== Unseen Holdout ===")
        print(f"Window: {args.holdout_start} -> {args.holdout_end} (exclusive)")
        print(f"Chunks: {coverage['chunks_ok']} OK / {coverage['chunks_failed']} failed")
        print(f"Evaluation coverage: {coverage['evaluation_coverage']:.2%}")
        for key in ("detection_rate", "precision", "recall", "f1", "false_alarm_rate", "missed_event_rate"):
            value = overall[key]
            print(f"{key:20s}: {value:.4%}" if np.isfinite(value) else f"{key:20s}: N/A")
        events = holdout_results["event_level_performance"]
        print(
            f"Event detection rate: {events['event_detection_rate']:.2%}"
            if np.isfinite(events["event_detection_rate"])
            else "Event detection rate: N/A"
        )

    Path(args.output_report).write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    print(f"Report written to: {args.output_report}")
    return 0 if report.get("holdout", {}).get("coverage", {}).get("chunks_failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
