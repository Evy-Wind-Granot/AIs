#!/usr/bin/env python3
"""
Magnetometer performance benchmark.

This benchmark evaluates the production QDC/activity pipeline without changing
its inference logic. It reports:
  - data completeness and cadence quality
  - residual MAE/RMSE/bias/P95 and quiet-period quality
  - sample-level activity/storm metrics against Kp/Dst when available
  - event-level precision/recall/F1 and onset/duration statistics
  - explicit Kp/Dst reference coverage
  - threshold sweeps for choosing an operating point without silently changing
    the production thresholds
  - daily residual stability and processing throughput
  - deterministic synthetic self-tests for the benchmark implementation

Examples:
    python magnetometer/performance_metrics.py --observatory VIC \
        --start-date 2024-03-15 --days 60

    python magnetometer/performance_metrics.py --observatory VIC \
        --start-date 2024-03-15 --days 60 --sweep-thresholds

    python magnetometer/performance_metrics.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from magnetometer_demo import (  # noqa: E402
    build_design_matrix,
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    flag_activity,
    parse_iaga2002_to_dataframe,
    robust_harmonic_baseline,
)

# Production thresholds from magnetometer_demo.py. These are measured here, not
# modified by the benchmark.
QUIET_LIMIT_NT = 15.0
ACTIVE_LIMIT_NT = 35.0
MINOR_STORM_LIMIT_NT = 70.0
MAJOR_STORM_LIMIT_NT = 150.0
SEVERE_STORM_LIMIT_NT = 300.0
ANOMALY_DELTA_NT = 100.0


def finite_values(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.isfinite(x)]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, Optional[float]]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have the same shape")

    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    total = tp + tn + fp + fn

    def div(a: float, b: float) -> Optional[float]:
        return float(a / b) if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None

    return {
        "samples": total,
        "positive_truth_samples": int(tp + fn),
        "positive_prediction_samples": int(tp + fp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": div(tp + tn, total),
        "balanced_accuracy": (
            float((recall + specificity) / 2)
            if recall is not None and specificity is not None
            else None
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_alarm_rate": div(fp, fp + tn),
        "miss_rate": div(fn, fn + tp),
    }


def compute_qdc_baseline(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate the production rolling QDC construction used by the demo."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    baseline = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)

    window_samples = max(1, int(24 * 3600 / cadence_s))
    step_samples = max(1, window_samples // 2)
    t_global = np.arange(n, dtype=float) * cadence_s / 3600.0
    t_min = float(t_global.min()) if n else 0.0
    t_max = float(t_global.max()) if n else 0.0
    last_good_coeffs = None

    for start in range(0, max(1, n - step_samples), step_samples):
        end = min(start + window_samples, n)
        if end - start < step_samples // 2:
            break

        segment = x[start:end]
        t_seg = t_global[start:end]
        if np.isfinite(segment).sum() < (end - start) * 0.5:
            continue

        seg_base, coeffs = robust_harmonic_baseline(
            segment,
            cadence_s,
            t_hours=t_seg,
            t_ref_min=t_min,
            t_ref_max=t_max,
        )

        seg_res = segment - seg_base
        storm_frac = float(np.mean(np.abs(seg_res) > 50.0))
        if storm_frac > 0.05 and last_good_coeffs is not None:
            seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good_coeffs
        elif storm_frac <= 0.05:
            last_good_coeffs = coeffs

        w_win = np.hanning(end - start)
        baseline[start:end] += seg_base * w_win
        weights[start:end] += w_win

    mask = weights > 0
    finite_x = finite_values(x)
    fallback = float(np.nanmedian(finite_x)) if finite_x.size else 0.0
    baseline[mask] /= weights[mask]
    baseline[~mask] = fallback
    residual = x - baseline
    return baseline, residual


def prepare_global_reference(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    n: int,
    cadence_s: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Fetch Kp/Dst and align them to the magnetometer cadence."""
    kp = pd.Series(dtype=float)
    dst = pd.Series(dtype=float)
    kp_error = None
    dst_failures: List[str] = []

    try:
        kp = fetch_kp_gfz(start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
    except Exception as exc:  # benchmark remains useful if one source fails
        kp_error = str(exc)

    months = pd.period_range(start=start_time.strftime("%Y-%m"), end=end_time.strftime("%Y-%m"), freq="M")
    dst_parts = []
    for period in months:
        try:
            part = fetch_dst_kyoto(int(period.year), int(period.month))
        except Exception as exc:
            part = None
            dst_failures.append(f"{period}: {exc}")
        if part is not None and not part.empty:
            dst_parts.append(part)
        elif part is None:
            dst_failures.append(str(period))
    if dst_parts:
        dst = pd.concat(dst_parts).sort_index()

    target_index = pd.date_range(
        start=start_time,
        periods=n,
        freq=pd.Timedelta(seconds=cadence_s),
        tz="UTC",
    )
    tolerance = pd.Timedelta("3h")
    kp_aligned = (
        kp.reindex(target_index, method="ffill", tolerance=tolerance)
        if not kp.empty
        else pd.Series(np.nan, index=target_index)
    )
    dst_aligned = (
        dst.reindex(target_index, method="ffill", tolerance=tolerance)
        if not dst.empty
        else pd.Series(np.nan, index=target_index)
    )

    return kp_aligned, dst_aligned, {
        "kp_fetch_ok": kp_error is None,
        "kp_error": kp_error,
        "dst_months_requested": len(months),
        "dst_months_with_data": len(dst_parts),
        "dst_failures": dst_failures,
    }


def activity_reference(kp: pd.Series, dst: pd.Series) -> Dict[str, np.ndarray]:
    kp_values = kp.to_numpy(dtype=float)
    dst_values = dst.to_numpy(dtype=float)
    kp_known = np.isfinite(kp_values)
    dst_known = np.isfinite(dst_values)

    # Reference definitions are deliberately conservative and transparent.
    active = ((kp_known & (kp_values >= 4.0)) | (dst_known & (dst_values < -30.0)))
    storm = ((kp_known & (kp_values >= 6.0)) | (dst_known & (dst_values < -50.0)))
    reference_known = kp_known | dst_known
    return {
        "reference_known": reference_known,
        "kp_known": kp_known,
        "dst_known": dst_known,
        "active": active & reference_known,
        "storm": storm & reference_known,
    }


def bool_events(mask: np.ndarray, cadence_s: float, max_gap_s: float = 0.0, min_duration_s: float = 0.0) -> List[Tuple[int, int]]:
    """Convert a boolean series into contiguous [start,end) events."""
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0:
        return []

    if max_gap_s > 0 and mask.size > 1:
        max_gap = max(1, int(round(max_gap_s / cadence_s)))
        false_runs = np.flatnonzero(~mask)
        if false_runs.size:
            padded = np.r_[True, mask, True]
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends = np.flatnonzero(padded[:-1] & ~padded[1:])
            for s, e in zip(starts, ends):
                if e - s <= max_gap:
                    mask[s:e] = True

    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    min_len = max(1, int(math.ceil(min_duration_s / cadence_s))) if min_duration_s > 0 else 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def event_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(
    predicted: Sequence[Tuple[int, int]],
    reference: Sequence[Tuple[int, int]],
    cadence_s: float,
) -> Dict[str, Any]:
    """Greedy one-to-one event matching by overlap."""
    candidates = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            overlap = event_overlap(pred, ref)
            if overlap > 0:
                candidates.append((overlap, pi, ri))
    candidates.sort(reverse=True)

    used_pred = set()
    used_ref = set()
    matches = []
    for overlap, pi, ri in candidates:
        if pi in used_pred or ri in used_ref:
            continue
        used_pred.add(pi)
        used_ref.add(ri)
        pred = predicted[pi]
        ref = reference[ri]
        matches.append({
            "predicted_index": pi,
            "reference_index": ri,
            "overlap_samples": overlap,
            "overlap_seconds": overlap * cadence_s,
            "onset_latency_seconds": (pred[0] - ref[0]) * cadence_s,
            "predicted_duration_seconds": (pred[1] - pred[0]) * cadence_s,
            "reference_duration_seconds": (ref[1] - ref[0]) * cadence_s,
            "duration_error_seconds": ((pred[1] - pred[0]) - (ref[1] - ref[0])) * cadence_s,
        })

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None

    latencies = [m["onset_latency_seconds"] for m in matches]
    duration_errors = [abs(m["duration_error_seconds"]) for m in matches]
    return {
        "reference_events": len(reference),
        "predicted_events": len(predicted),
        "matched_events": tp,
        "missed_events": fn,
        "false_positive_events": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_onset_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "median_onset_latency_seconds": float(np.median(latencies)) if latencies else None,
        "mean_absolute_onset_latency_seconds": float(np.mean(np.abs(latencies))) if latencies else None,
        "mean_absolute_duration_error_seconds": float(np.mean(duration_errors)) if duration_errors else None,
        "matches": matches,
    }


def threshold_metrics(
    residual: np.ndarray,
    reference_active: np.ndarray,
    reference_storm: np.ndarray,
    known: np.ndarray,
    cadence_s: float,
    active_threshold: float,
    storm_threshold: float,
    event_min_duration_s: float = 5 * 60,
    event_max_gap_s: float = 5 * 60,
) -> Dict[str, Any]:
    magnitude = np.abs(np.asarray(residual, dtype=float))
    valid = known & np.isfinite(magnitude)
    pred_active = valid & (magnitude > active_threshold)
    pred_storm = valid & (magnitude > storm_threshold)

    active_sample = binary_metrics(pred_active[valid], reference_active[valid])
    storm_sample = binary_metrics(pred_storm[valid], reference_storm[valid])

    pred_active_events = bool_events(pred_active & valid, cadence_s, event_max_gap_s, event_min_duration_s)
    ref_active_events = bool_events(reference_active & valid, cadence_s, event_max_gap_s, event_min_duration_s)
    pred_storm_events = bool_events(pred_storm & valid, cadence_s, event_max_gap_s, event_min_duration_s)
    ref_storm_events = bool_events(reference_storm & valid, cadence_s, event_max_gap_s, event_min_duration_s)

    return {
        "active": {
            "threshold_nt": active_threshold,
            "sample_level": active_sample,
            "event_level": match_events(pred_active_events, ref_active_events, cadence_s),
        },
        "storm": {
            "threshold_nt": storm_threshold,
            "sample_level": storm_sample,
            "event_level": match_events(pred_storm_events, ref_storm_events, cadence_s),
        },
    }


def threshold_sweep(
    residual: np.ndarray,
    reference_active: np.ndarray,
    reference_storm: np.ndarray,
    known: np.ndarray,
    cadence_s: float,
) -> Dict[str, Any]:
    """Search thresholds without changing production thresholds."""
    active_candidates = [15, 20, 25, 30, 35, 40, 45, 50, 60]
    storm_candidates = [35, 50, 60, 70, 80, 100, 120, 150]

    active_rows = []
    for threshold in active_candidates:
        m = threshold_metrics(
            residual, reference_active, reference_storm, known,
            cadence_s, threshold, max(70.0, threshold * 2),
        )["active"]
        sample = m["sample_level"]
        active_rows.append({"threshold_nt": threshold, **{k: sample[k] for k in ("precision", "recall", "f1", "false_alarm_rate", "miss_rate")}})

    storm_rows = []
    for threshold in storm_candidates:
        m = threshold_metrics(
            residual, reference_active, reference_storm, known,
            cadence_s, max(35.0, threshold / 2), threshold,
        )["storm"]
        sample = m["sample_level"]
        storm_rows.append({"threshold_nt": threshold, **{k: sample[k] for k in ("precision", "recall", "f1", "false_alarm_rate", "miss_rate")}})

    def best(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        valid_rows = [r for r in rows if r["f1"] is not None]
        return max(valid_rows, key=lambda r: r["f1"]) if valid_rows else None

    return {
        "active": {"candidates": active_rows, "best_sample_level_f1": best(active_rows)},
        "storm": {"candidates": storm_rows, "best_sample_level_f1": best(storm_rows)},
    }


def daily_rms(residual: np.ndarray, index: pd.DatetimeIndex) -> Dict[str, float]:
    series = pd.Series(residual, index=index).dropna()
    if series.empty:
        return {}
    grouped = series.groupby(series.index.floor("D"))
    values = grouped.apply(lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy())))))
    return {str(day.date()): round(float(value), 6) for day, value in values.items()}


def benchmark(
    observatory: str,
    start_date: str,
    days: int,
    output_dir: Path,
    sweep_thresholds: bool = False,
    event_min_minutes: float = 5.0,
    event_gap_minutes: float = 5.0,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_start = time.perf_counter()
    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
        samples_per_day="Minute",
    )
    fetch_seconds = time.perf_counter() - fetch_start

    parse_start = time.perf_counter()
    df = parse_iaga2002_to_dataframe(raw)
    parse_seconds = time.perf_counter() - parse_start
    if df.empty:
        raise RuntimeError("INTERMAGNET returned no usable samples.")

    series = pd.to_numeric(df["f_nt"], errors="coerce")
    valid = series.notna()
    if int(valid.sum()) < 24:
        raise RuntimeError(f"Only {int(valid.sum())} valid F samples available; need at least 24.")

    index = series.index
    expected = max(1, int(days * 24 * 60))
    valid_count = int(valid.sum())
    completeness = valid_count / expected
    cadence_seconds = float(index.to_series().diff().dropna().dt.total_seconds().median())

    analysis_start = time.perf_counter()
    _baseline, residual = compute_qdc_baseline(series.to_numpy(dtype=float), cadence_seconds)
    analysis_seconds = time.perf_counter() - analysis_start

    finite_resid = finite_values(residual)
    abs_resid = np.abs(finite_resid)
    rms = float(np.sqrt(np.mean(finite_resid ** 2)))
    mae = float(np.mean(abs_resid))
    median_abs = float(np.median(abs_resid))
    p95_abs = float(np.percentile(abs_resid, 95))
    bias = float(np.mean(finite_resid))
    residual_std = float(np.std(finite_resid))

    quiet_mask = np.isfinite(residual) & (np.abs(residual) <= QUIET_LIMIT_NT)
    quiet_resid = residual[quiet_mask]
    quiet_rms = float(np.sqrt(np.mean(quiet_resid ** 2))) if quiet_resid.size else None
    quiet_mae = float(np.mean(np.abs(quiet_resid))) if quiet_resid.size else None

    global_fetch_start = time.perf_counter()
    kp_aligned, dst_aligned, ref_status = prepare_global_reference(index[0], index[-1], len(index), int(round(cadence_seconds)))
    global_fetch_seconds = time.perf_counter() - global_fetch_start
    refs = activity_reference(kp_aligned, dst_aligned)
    known = refs["reference_known"] & np.isfinite(residual)

    production_metrics = None
    if int(known.sum()) > 0:
        production_metrics = threshold_metrics(
            residual,
            refs["active"],
            refs["storm"],
            known,
            cadence_seconds,
            ACTIVE_LIMIT_NT,
            MINOR_STORM_LIMIT_NT,
            event_min_minutes * 60,
            event_gap_minutes * 60,
        )

    sweep = threshold_sweep(
        residual,
        refs["active"],
        refs["storm"],
        known,
        cadence_seconds,
    ) if sweep_thresholds and int(known.sum()) > 0 else None

    flags = flag_activity(residual)
    anomaly = np.isfinite(residual) & (np.abs(np.diff(residual, prepend=residual[0])) > ANOMALY_DELTA_NT)
    predicted_active = np.isfinite(residual) & (np.abs(residual) > ACTIVE_LIMIT_NT)
    predicted_storm = np.isfinite(residual) & (np.abs(residual) > MINOR_STORM_LIMIT_NT)

    daily = daily_rms(residual, index)
    daily_values = list(daily.values())

    throughput = valid_count / analysis_seconds if analysis_seconds > 0 else None
    realtime_factor = throughput * cadence_seconds if throughput is not None else None
    total_runtime = fetch_seconds + parse_seconds + analysis_seconds + global_fetch_seconds

    report: Dict[str, Any] = {
        "benchmark": {
            "observatory": observatory,
            "start_date": start_date,
            "days": days,
            "field": "f_nt",
            "samples": len(series),
            "valid_samples": valid_count,
            "expected_samples": expected,
            "completeness": safe_float(completeness),
            "cadence_seconds": safe_float(cadence_seconds),
        },
        "baseline_quality_nt": {
            "bias": safe_float(bias),
            "mae": safe_float(mae),
            "rmse": safe_float(rms),
            "rms": safe_float(rms),
            "residual_std": safe_float(residual_std),
            "median_absolute_error": safe_float(median_abs),
            "p95_absolute_error": safe_float(p95_abs),
            "quiet_rms": safe_float(quiet_rms),
            "quiet_mae": safe_float(quiet_mae),
        },
        "activity": {
            "quiet_fraction": safe_float(float(np.mean(flags == "quiet"))),
            "unsettled_fraction": safe_float(float(np.mean(flags == "unsettled"))),
            "active_or_higher_fraction": safe_float(float(np.mean(predicted_active))),
            "storm_or_higher_fraction": safe_float(float(np.mean(predicted_storm))),
            "anomaly_fraction": safe_float(float(np.mean(anomaly))),
            "counts": {
                "quiet": int(np.sum(flags == "quiet")),
                "unsettled": int(np.sum(flags == "unsettled")),
                "active": int(np.sum(flags == "active")),
                "minor_storm": int(np.sum(flags == "minor_storm")),
                "major_storm": int(np.sum(flags == "major_storm")),
                "severe_storm": int(np.sum(flags == "severe_storm")),
                "anomaly": int(np.sum(flags == "anomaly")),
            },
        },
        "reference": {
            **ref_status,
            "samples_with_any_reference": int(np.sum(known)),
            "reference_coverage_fraction": safe_float(float(np.mean(known))) if len(known) else 0.0,
            "kp_known_samples": int(np.sum(refs["kp_known"] & np.isfinite(residual))),
            "dst_known_samples": int(np.sum(refs["dst_known"] & np.isfinite(residual))),
            "kp_coverage_fraction": safe_float(float(np.mean(refs["kp_known"] & np.isfinite(residual)))) if len(known) else 0.0,
            "dst_coverage_fraction": safe_float(float(np.mean(refs["dst_known"] & np.isfinite(residual)))) if len(known) else 0.0,
        },
        "validation": {
            "production_thresholds": {
                "active_nt": ACTIVE_LIMIT_NT,
                "storm_nt": MINOR_STORM_LIMIT_NT,
            },
            "event_settings": {
                "minimum_event_minutes": event_min_minutes,
                "gap_merge_minutes": event_gap_minutes,
            },
            "sample_level": production_metrics,
            "threshold_sweep": sweep,
        },
        "performance": {
            "fetch_seconds": safe_float(fetch_seconds),
            "parse_seconds": safe_float(parse_seconds),
            "analysis_seconds": safe_float(analysis_seconds),
            "global_index_seconds": safe_float(global_fetch_seconds),
            "total_seconds": safe_float(total_runtime),
            "analysis_samples_per_second": safe_float(throughput),
            "realtime_factor": safe_float(realtime_factor),
        },
        "daily_rms_nt": daily,
        "daily_rms_summary_nt": {
            "mean": safe_float(float(np.mean(daily_values))) if daily_values else None,
            "std": safe_float(float(np.std(daily_values))) if daily_values else None,
            "min": safe_float(float(np.min(daily_values))) if daily_values else None,
            "max": safe_float(float(np.max(daily_values))) if daily_values else None,
        },
    }

    json_path = output_dir / f"magnetometer_performance_{observatory}_{start_date}_{days}d.json"
    json_path.write_text(json.dumps(report, indent=2))

    rows = [
        ("completeness", completeness),
        ("bias_nt", bias),
        ("mae_nt", mae),
        ("rmse_nt", rms),
        ("median_abs_error_nt", median_abs),
        ("p95_abs_error_nt", p95_abs),
        ("quiet_rms_nt", quiet_rms),
        ("quiet_mae_nt", quiet_mae),
        ("active_fraction", float(np.mean(predicted_active))),
        ("storm_fraction", float(np.mean(predicted_storm))),
        ("anomaly_fraction", float(np.mean(anomaly))),
        ("reference_coverage", float(np.mean(known)) if len(known) else 0.0),
        ("kp_coverage", float(np.mean(refs["kp_known"] & np.isfinite(residual))) if len(known) else 0.0),
        ("dst_coverage", float(np.mean(refs["dst_known"] & np.isfinite(residual))) if len(known) else 0.0),
        ("analysis_seconds", analysis_seconds),
        ("analysis_samples_per_second", throughput),
        ("realtime_factor", realtime_factor),
    ]
    if production_metrics:
        a = production_metrics["active"]["sample_level"]
        s = production_metrics["storm"]["sample_level"]
        ae = production_metrics["active"]["event_level"]
        se = production_metrics["storm"]["event_level"]
        rows.extend([
            ("active_precision", a["precision"]),
            ("active_recall", a["recall"]),
            ("active_f1", a["f1"]),
            ("active_false_alarm_rate", a["false_alarm_rate"]),
            ("storm_precision", s["precision"]),
            ("storm_recall", s["recall"]),
            ("storm_f1", s["f1"]),
            ("storm_false_alarm_rate", s["false_alarm_rate"]),
            ("active_event_precision", ae["precision"]),
            ("active_event_recall", ae["recall"]),
            ("active_event_f1", ae["f1"]),
            ("active_event_mean_abs_onset_latency_s", ae["mean_absolute_onset_latency_seconds"]),
            ("storm_event_precision", se["precision"]),
            ("storm_event_recall", se["recall"]),
            ("storm_event_f1", se["f1"]),
            ("storm_event_mean_abs_onset_latency_s", se["mean_absolute_onset_latency_seconds"]),
        ])
    csv_path = output_dir / f"magnetometer_performance_{observatory}_{start_date}_{days}d.csv"
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(csv_path, index=False)

    print("\n" + "=" * 72)
    print("MAGNETOMETER PERFORMANCE REPORT")
    print("=" * 72)
    print(f"Observatory:              {observatory}")
    print(f"Period:                   {start_date} ({days} days)")
    print(f"Samples:                  {len(series):,}")
    print(f"Valid samples:            {valid_count:,} ({completeness * 100:.2f}%)")
    print(f"Cadence:                  {cadence_seconds:.1f} s")
    print("-" * 72)
    print(f"Residual MAE:             {mae:.3f} nT")
    print(f"Residual RMSE/RMS:        {rms:.3f} nT")
    print(f"Residual bias:            {bias:.3f} nT")
    print(f"Residual median |e|:      {median_abs:.3f} nT")
    print(f"Residual P95 |e|:         {p95_abs:.3f} nT")
    print(f"Quiet-period RMS:         {quiet_rms:.3f} nT" if quiet_rms is not None else "Quiet-period RMS:         N/A")
    print(f"Quiet-period MAE:         {quiet_mae:.3f} nT" if quiet_mae is not None else "Quiet-period MAE:         N/A")
    print("-" * 72)
    print(f"Reference coverage:       {float(np.mean(known)) * 100:.2f}%")
    print(f"Kp coverage:              {float(np.mean(refs['kp_known'] & np.isfinite(residual))) * 100:.2f}%")
    print(f"Dst coverage:             {float(np.mean(refs['dst_known'] & np.isfinite(residual))) * 100:.2f}%")
    print(f"Detected active+ fraction:{float(np.mean(predicted_active)) * 100:.3f}%")
    print(f"Detected storm+ fraction: {float(np.mean(predicted_storm)) * 100:.3f}%")
    print(f"Anomaly fraction:         {float(np.mean(anomaly)) * 100:.3f}%")

    if production_metrics:
        active = production_metrics["active"]
        storm = production_metrics["storm"]
        ae = active["event_level"]
        se = storm["event_level"]
        print("-" * 72)
        print("SAMPLE-LEVEL VALIDATION")
        for name, group in (("Active", active), ("Storm", storm)):
            m = group["sample_level"]
            print(f"{name} precision:         {m['precision']:.3f}" if m['precision'] is not None else f"{name} precision:         N/A")
            print(f"{name} recall:            {m['recall']:.3f}" if m['recall'] is not None else f"{name} recall:            N/A")
            print(f"{name} F1:                {m['f1']:.3f}" if m['f1'] is not None else f"{name} F1:                N/A")
            print(f"{name} false alarm rate:  {m['false_alarm_rate']:.3f}" if m['false_alarm_rate'] is not None else f"{name} false alarm rate:  N/A")
        print("-" * 72)
        print("EVENT-LEVEL VALIDATION")
        for name, group in (("Active", ae), ("Storm", se)):
            print(f"{name} reference events:   {group['reference_events']}")
            print(f"{name} predicted events:   {group['predicted_events']}")
            print(f"{name} matched events:     {group['matched_events']}")
            print(f"{name} event precision:    {group['precision']:.3f}" if group['precision'] is not None else f"{name} event precision:    N/A")
            print(f"{name} event recall:       {group['recall']:.3f}" if group['recall'] is not None else f"{name} event recall:       N/A")
            print(f"{name} event F1:           {group['f1']:.3f}" if group['f1'] is not None else f"{name} event F1:           N/A")
            print(f"{name} mean |onset|:       {group['mean_absolute_onset_latency_seconds'] / 60:.2f} min" if group['mean_absolute_onset_latency_seconds'] is not None else f"{name} mean |onset|:       N/A")

    if sweep:
        print("-" * 72)
        print("THRESHOLD SWEEP (sample-level F1; production thresholds unchanged)")
        best_active = sweep["active"]["best_sample_level_f1"]
        best_storm = sweep["storm"]["best_sample_level_f1"]
        if best_active:
            print(f"Best active threshold:  {best_active['threshold_nt']:.0f} nT (F1={best_active['f1']:.3f})")
        else:
            print("Best active threshold:  N/A")
        if best_storm:
            print(f"Best storm threshold:   {best_storm['threshold_nt']:.0f} nT (F1={best_storm['f1']:.3f})")
        else:
            print("Best storm threshold:   N/A")

    print("-" * 72)
    print(f"Analysis time:            {analysis_seconds:.3f} s")
    print(f"Analysis throughput:      {throughput:,.0f} samples/s" if throughput is not None else "Analysis throughput:      N/A")
    print(f"Realtime factor:          {realtime_factor:,.0f}×" if realtime_factor is not None else "Realtime factor:          N/A")
    print(f"Total benchmark time:     {total_runtime:.3f} s")
    print(f"JSON report:              {json_path}")
    print(f"CSV summary:              {csv_path}")
    print("=" * 72)

    return report


def self_test() -> None:
    """Deterministic tests for event matching, threshold metrics, and QDC output."""
    cadence = 60.0
    n = 360
    t = np.arange(n) * cadence
    baseline = 100.0 + 2.0 * np.sin(2 * np.pi * t / 86400.0)
    signal = baseline.copy()
    signal[100:130] += 90.0
    signal[220:250] += 180.0
    signal[300] += 140.0

    qdc, residual = compute_qdc_baseline(signal, cadence)
    assert qdc.shape == signal.shape
    assert residual.shape == signal.shape
    assert np.isfinite(qdc).all()

    reference_active = np.zeros(n, dtype=bool)
    reference_active[100:130] = True
    reference_active[220:250] = True
    reference_storm = np.zeros(n, dtype=bool)
    reference_storm[220:250] = True
    known = np.ones(n, dtype=bool)

    metrics = threshold_metrics(
        residual,
        reference_active,
        reference_storm,
        known,
        cadence,
        35.0,
        70.0,
        event_min_duration_s=5 * 60,
        event_max_gap_s=5 * 60,
    )
    assert metrics["active"]["event_level"]["reference_events"] >= 1
    assert metrics["storm"]["event_level"]["reference_events"] >= 1

    matched = match_events([(10, 20), (40, 55)], [(12, 19), (42, 50)], cadence)
    assert matched["matched_events"] == 2
    assert matched["recall"] == 1.0

    print("MAGNETOMETER PERFORMANCE SELF-TEST: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the magnetometer QDC/activity pipeline.")
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "magnetometer" / "data"),
        help="Directory for JSON/CSV benchmark reports.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Evaluate alternate thresholds without changing production thresholds.",
    )
    parser.add_argument(
        "--event-minutes",
        type=float,
        default=5.0,
        help="Minimum duration for an event-level detection (default 5 min).",
    )
    parser.add_argument(
        "--event-gap-minutes",
        type=float,
        default=5.0,
        help="Merge event fragments separated by at most this gap (default 5 min).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic benchmark self-tests without network access.",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    benchmark(
        observatory=args.observatory,
        start_date=args.start_date,
        days=args.days,
        output_dir=Path(args.output_dir).resolve(),
        sweep_thresholds=args.sweep_thresholds,
        event_min_minutes=args.event_minutes,
        event_gap_minutes=args.event_gap_minutes,
    )


if __name__ == "__main__":
    main()
