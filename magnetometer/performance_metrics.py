#!/usr/bin/env python3
"""
Production-grade magnetometer validation harness.

This benchmark evaluates the existing magnetometer QDC/activity pipeline. It
DOES NOT change production inference thresholds or inference logic.

Modes
-----
Single case (backward compatible):
    python magnetometer/performance_metrics.py \
        --observatory VIC --start-date 2024-03-15 --days 60

Production validation suite (recommended):
    python magnetometer/performance_metrics.py --production-suite

The production suite:
  * discovers representative quiet / active / storm periods from Kp;
  * separates calibration periods from held-out test periods;
  * tunes candidate thresholds ONLY on calibration data;
  * evaluates the selected candidate ONLY on held-out test data;
  * reports per-case and aggregate metrics;
  * reports Kp/Dst reference coverage explicitly;
  * keeps sample-level and event-level scoring separate;
  * never silently substitutes a missing reference source.

A deterministic self-test is available:
    python magnetometer/performance_metrics.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
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

# These are the current production thresholds in magnetometer_demo.py.
PROD_UNSETTLED_NT = 15.0
PROD_ACTIVE_NT = 35.0
PROD_MINOR_STORM_NT = 70.0
PROD_MAJOR_STORM_NT = 150.0
PROD_SEVERE_STORM_NT = 300.0
ANOMALY_DELTA_NT = 100.0

DEFAULT_YEARS = (2023, 2024, 2025)
DEFAULT_CLASSES = ("quiet", "active", "storm")
DEFAULT_CASES_PER_CLASS = 2
DEFAULT_WINDOW_DAYS = 7


@dataclass(frozen=True)
class SuiteCase:
    case_id: str
    start_date: str
    days: int
    class_name: str


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def nanmean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def nanmedian_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(clean)) if clean else None


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have the same shape")

    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    total = tp + tn + fp + fn

    def div(num: float, den: float) -> Optional[float]:
        return float(num / den) if den else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None

    return {
        "samples": total,
        "positive_truth_samples": tp + fn,
        "positive_prediction_samples": tp + fp,
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


def confusion_counts(metrics: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        key: int(sum(int(m.get(key, 0) or 0) for m in metrics))
        for key in ("tp", "tn", "fp", "fn")
    }


def aggregate_binary_metrics(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = confusion_counts(metrics)
    tp, tn, fp, fn = counts.values()
    total = tp + tn + fp + fn

    def div(num: float, den: float) -> Optional[float]:
        return float(num / den) if den else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None

    return {
        "samples": total,
        "positive_truth_samples": tp + fn,
        "positive_prediction_samples": tp + fp,
        **counts,
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


# ---------------------------------------------------------------------------
# Production QDC replication
# ---------------------------------------------------------------------------
def compute_qdc_baseline(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate the current production rolling-QDC logic exactly for scoring."""
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
    fallback = float(np.median(finite_x)) if finite_x.size else 0.0
    baseline[mask] /= weights[mask]
    baseline[~mask] = fallback
    return baseline, x - baseline


# ---------------------------------------------------------------------------
# Global references
# ---------------------------------------------------------------------------
def fetch_global_reference(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    n: int,
    cadence_s: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    kp = pd.Series(dtype=float)
    dst = pd.Series(dtype=float)
    kp_error = None
    dst_failures: List[str] = []

    try:
        kp = fetch_kp_gfz(start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
    except Exception as exc:
        kp_error = str(exc)

    periods = pd.period_range(start=start_time.strftime("%Y-%m"), end=end_time.strftime("%Y-%m"), freq="M")
    dst_parts = []
    for period in periods:
        try:
            part = fetch_dst_kyoto(int(period.year), int(period.month))
        except Exception as exc:
            part = None
            dst_failures.append(f"{period}: {exc}")
        if part is not None and not part.empty:
            dst_parts.append(part)
        else:
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

    meta = {
        "kp_fetch_ok": kp_error is None,
        "kp_error": kp_error,
        "dst_months_requested": len(periods),
        "dst_months_with_data": len(dst_parts),
        "dst_failures": dst_failures,
    }
    return kp_aligned, dst_aligned, meta


def reference_masks(kp: pd.Series, dst: pd.Series) -> Dict[str, np.ndarray]:
    kp_values = kp.to_numpy(dtype=float)
    dst_values = dst.to_numpy(dtype=float)
    kp_known = np.isfinite(kp_values)
    dst_known = np.isfinite(dst_values)
    known = kp_known | dst_known

    # Explicit operational definitions. These are reference labels, not the
    # production detector's thresholds.
    active = ((kp_known & (kp_values >= 4.0)) | (dst_known & (dst_values < -30.0))) & known
    storm = ((kp_known & (kp_values >= 6.0)) | (dst_known & (dst_values < -50.0))) & known
    return {
        "known": known,
        "kp_known": kp_known,
        "dst_known": dst_known,
        "active": active,
        "storm": storm,
    }


# ---------------------------------------------------------------------------
# Event handling
# ---------------------------------------------------------------------------
def bool_events(
    mask: np.ndarray,
    cadence_s: float,
    merge_gap_s: float = 0.0,
    min_duration_s: float = 0.0,
) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0:
        return []

    # Merge short gaps so a single physical episode is not counted as many
    # events because of minute-level threshold flicker.
    max_gap = max(0, int(round(merge_gap_s / cadence_s)))
    if max_gap > 0:
        padded = np.r_[False, mask, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - ends[i]
            if gap <= max_gap:
                mask[ends[i]:starts[i + 1]] = True

    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    min_len = max(1, int(math.ceil(min_duration_s / cadence_s)))
    return [
        (int(s), int(e))
        for s, e in zip(starts, ends)
        if e - s >= min_len
    ]


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(
    predicted: Sequence[Tuple[int, int]],
    reference: Sequence[Tuple[int, int]],
    cadence_s: float,
) -> Dict[str, Any]:
    candidates = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            ov = overlap(pred, ref)
            if ov > 0:
                candidates.append((ov, pi, ri))
    candidates.sort(reverse=True)

    used_pred = set()
    used_ref = set()
    matches = []
    for ov, pi, ri in candidates:
        if pi in used_pred or ri in used_ref:
            continue
        used_pred.add(pi)
        used_ref.add(ri)
        pred = predicted[pi]
        ref = reference[ri]
        matches.append({
            "predicted_index": pi,
            "reference_index": ri,
            "overlap_seconds": float(ov * cadence_s),
            # For Kp, this is an offset from the coarse reference episode start,
            # not a high-resolution detector latency.
            "reference_start_offset_seconds": float((pred[0] - ref[0]) * cadence_s),
            "predicted_duration_seconds": float((pred[1] - pred[0]) * cadence_s),
            "reference_duration_seconds": float((ref[1] - ref[0]) * cadence_s),
            "duration_error_seconds": float(((pred[1] - pred[0]) - (ref[1] - ref[0])) * cadence_s),
        })

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None

    offsets = [m["reference_start_offset_seconds"] for m in matches]
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
        "median_reference_start_offset_seconds": float(np.median(offsets)) if offsets else None,
        "median_absolute_reference_start_offset_seconds": float(np.median(np.abs(offsets))) if offsets else None,
        "mean_absolute_duration_error_seconds": float(np.mean(duration_errors)) if duration_errors else None,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Threshold scoring
# ---------------------------------------------------------------------------
def score_thresholds(
    residual: np.ndarray,
    refs: Dict[str, np.ndarray],
    cadence_s: float,
    active_threshold: float,
    storm_threshold: float,
) -> Dict[str, Any]:
    magnitude = np.abs(np.asarray(residual, dtype=float))
    known = refs["known"] & np.isfinite(magnitude)

    pred_active = known & (magnitude > active_threshold)
    pred_storm = known & (magnitude > storm_threshold)

    active_sample = binary_metrics(pred_active[known], refs["active"][known])
    storm_sample = binary_metrics(pred_storm[known], refs["storm"][known])

    # Production event scoring is deliberately less sensitive to minute-level
    # flicker than sample scoring.
    pred_active_events = bool_events(pred_active, cadence_s, merge_gap_s=30 * 60, min_duration_s=5 * 60)
    ref_active_events = bool_events(refs["active"] & refs["known"], cadence_s, merge_gap_s=6 * 3600, min_duration_s=3 * 3600)
    pred_storm_events = bool_events(pred_storm, cadence_s, merge_gap_s=30 * 60, min_duration_s=5 * 60)
    ref_storm_events = bool_events(refs["storm"] & refs["known"], cadence_s, merge_gap_s=6 * 3600, min_duration_s=3 * 3600)

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


# ---------------------------------------------------------------------------
# Kp-driven case discovery
# ---------------------------------------------------------------------------
def classify_window(kp: pd.Series, center: pd.Timestamp, days: int) -> str:
    half = pd.Timedelta(days=days / 2)
    window = kp.loc[(kp.index >= center - half) & (kp.index < center + half)]
    if window.empty:
        return "unknown"
    peak = float(window.max())
    if peak >= 6:
        return "storm"
    if peak >= 4:
        return "active"
    if peak <= 2:
        return "quiet"
    return "mixed"


def discover_cases(
    years: Sequence[int],
    window_days: int,
    per_class: int,
) -> List[SuiteCase]:
    start = f"{min(years):04d}-01-01"
    end = f"{max(years):04d}-12-31"
    kp = fetch_kp_gfz(start, end)
    if kp.empty:
        raise RuntimeError("Could not discover validation cases because Kp data were unavailable.")

    # Candidate centers are Kp observations. Select separated windows per class
    # so the suite does not repeatedly score the same physical disturbance.
    desired = ["quiet", "active", "storm"]
    candidates: Dict[str, List[pd.Timestamp]] = {name: [] for name in desired}
    for ts, value in kp.items():
        v = float(value)
        cls = "storm" if v >= 6 else "active" if v >= 4 else "quiet" if v <= 2 else None
        if cls:
            candidates[cls].append(ts)

    # Strongest disturbance first; quiet windows prefer low-Kp centers.
    candidates["storm"].sort(key=lambda t: float(kp.loc[t]), reverse=True)
    candidates["active"].sort(key=lambda t: float(kp.loc[t]), reverse=True)
    candidates["quiet"].sort(key=lambda t: float(kp.loc[t]))

    selected: List[SuiteCase] = []
    used_centers: List[pd.Timestamp] = []
    separation = pd.Timedelta(days=max(5, window_days))

    for cls in desired:
        count = 0
        for center in candidates[cls]:
            if any(abs(center - other) < separation for other in used_centers):
                continue
            start_ts = (center - pd.Timedelta(days=window_days / 2)).normalize()
            start_date = start_ts.strftime("%Y-%m-%d")
            case = SuiteCase(
                case_id=f"{cls}_{start_date}",
                start_date=start_date,
                days=window_days,
                class_name=cls,
            )
            selected.append(case)
            used_centers.append(center)
            count += 1
            if count >= per_class:
                break

    if len(selected) < per_class * len(desired):
        raise RuntimeError(
            f"Only discovered {len(selected)} validation cases; need {per_class * len(desired)}."
        )
    return sorted(selected, key=lambda c: c.start_date)


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------
def run_case(
    observatory: str,
    case: SuiteCase,
    output_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=case.start_date,
        duration_days=case.days,
        samples_per_day="Minute",
    )
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty or "f_nt" not in df.columns:
        raise RuntimeError("No usable F total-field data returned by INTERMAGNET.")

    series = pd.to_numeric(df["f_nt"], errors="coerce")
    valid = series.notna()
    if int(valid.sum()) < 24:
        raise RuntimeError(f"Only {int(valid.sum())} valid samples available.")

    index = series.index
    cadence_s = float(index.to_series().diff().dropna().dt.total_seconds().median())
    expected = int(round(case.days * 24 * 3600 / cadence_s))
    completeness = float(valid.sum() / max(1, expected))

    analysis_start = time.perf_counter()
    _, residual = compute_qdc_baseline(series.to_numpy(dtype=float), cadence_s)
    analysis_seconds = time.perf_counter() - analysis_start

    finite_resid = finite_values(residual)
    abs_resid = np.abs(finite_resid)
    quiet = np.abs(residual[np.isfinite(residual)]) <= PROD_UNSETTLED_NT
    quiet_resid = residual[np.isfinite(residual)][quiet]

    kp, dst, ref_meta = fetch_global_reference(index[0], index[-1], len(index), int(round(cadence_s)))
    refs = reference_masks(kp, dst)
    known = refs["known"] & np.isfinite(residual)

    production_scores = score_thresholds(
        residual,
        refs,
        cadence_s,
        PROD_ACTIVE_NT,
        PROD_MINOR_STORM_NT,
    )

    runtime = time.perf_counter() - started
    report = {
        "case": asdict(case),
        "observatory": observatory,
        "samples": len(series),
        "valid_samples": int(valid.sum()),
        "expected_samples": expected,
        "completeness": completeness,
        "cadence_seconds": cadence_s,
        "baseline_quality_nt": {
            "mae": float(np.mean(abs_resid)),
            "rmse": float(np.sqrt(np.mean(finite_resid ** 2))),
            "bias": float(np.mean(finite_resid)),
            "median_absolute_error": float(np.median(abs_resid)),
            "p95_absolute_error": float(np.percentile(abs_resid, 95)),
            "quiet_mae": float(np.mean(np.abs(quiet_resid))) if quiet_resid.size else None,
            "quiet_rms": float(np.sqrt(np.mean(quiet_resid ** 2))) if quiet_resid.size else None,
        },
        "reference_coverage": {
            "reference_samples": int(known.sum()),
            "reference_coverage": float(known.mean()),
            "kp_coverage": float(refs["kp_known"].mean()),
            "dst_coverage": float(refs["dst_known"].mean()),
            **ref_meta,
        },
        "production_thresholds": {
            "active_nt": PROD_ACTIVE_NT,
            "storm_nt": PROD_MINOR_STORM_NT,
        },
        "production_scores": production_scores,
        "performance": {
            "analysis_seconds": analysis_seconds,
            "analysis_samples_per_second": float(valid.sum() / analysis_seconds) if analysis_seconds > 0 else None,
            "total_case_seconds": runtime,
            "realtime_factor": float((case.days * 86400) / runtime) if runtime > 0 else None,
        },
    }

    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{observatory}_{case.case_id}.json"
    path.write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Calibration / held-out evaluation
# ---------------------------------------------------------------------------
def choose_thresholds(calibration_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Candidate values intentionally include the current production point.
    active_candidates = [15, 20, 25, 30, 35, 40, 45, 50]
    storm_candidates = [35, 50, 60, 70, 80, 100, 120, 150]

    # Re-running score calculations requires residual arrays, so calibration
    # reports retain a compact threshold table computed during case execution.
    active_rows = []
    storm_rows = []
    for report in calibration_reports:
        for row in report.get("threshold_table", []):
            active_rows.append(row["active"])
            storm_rows.append(row["storm"])

    def aggregate(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        grouped: Dict[float, List[float]] = {float(v): [] for v in (active_candidates if key == "active" else storm_candidates)}
        for row in rows:
            threshold = float(row["threshold_nt"])
            if threshold in grouped and row.get("f1") is not None:
                grouped[threshold].append(float(row["f1"]))
        return [
            {"threshold_nt": threshold, "mean_f1": (float(np.mean(values)) if values else None), "cases": len(values)}
            for threshold, values in grouped.items()
        ]

    active_summary = aggregate(active_rows, "active")
    storm_summary = aggregate(storm_rows, "storm")

    def best(rows: List[Dict[str, Any]], fallback: float) -> float:
        usable = [r for r in rows if r["mean_f1"] is not None and r["cases"] > 0]
        if not usable:
            return fallback
        return float(max(usable, key=lambda r: (r["mean_f1"], -r["threshold_nt"]))[
            "threshold_nt"
        ])

    return {
        "active_threshold_nt": best(active_summary, PROD_ACTIVE_NT),
        "storm_threshold_nt": best(storm_summary, PROD_MINOR_STORM_NT),
        "active_sweep": active_summary,
        "storm_sweep": storm_summary,
    }


# ---------------------------------------------------------------------------
# Production suite
# ---------------------------------------------------------------------------
def add_threshold_table(report: Dict[str, Any], residual: np.ndarray, refs: Dict[str, np.ndarray], cadence_s: float) -> None:
    rows = []
    known = refs["known"] & np.isfinite(residual)
    for threshold in [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150]:
        active = binary_metrics((np.abs(residual) > threshold)[known], refs["active"][known])
        storm = binary_metrics((np.abs(residual) > threshold)[known], refs["storm"][known])
        rows.append({
            "active": {"threshold_nt": threshold, "f1": active["f1"], "precision": active["precision"], "recall": active["recall"]},
            "storm": {"threshold_nt": threshold, "f1": storm["f1"], "precision": storm["precision"], "recall": storm["recall"]},
        })
    report["threshold_table"] = rows


def run_suite(
    observatories: Sequence[str],
    years: Sequence[int],
    cases_per_class: int,
    window_days: int,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(years, window_days, cases_per_class)

    calibration_cases = cases[::2]
    test_cases = cases[1::2]
    if not calibration_cases or not test_cases:
        raise RuntimeError("Suite split produced an empty calibration or test set.")

    all_case_reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for observatory in observatories:
        for case in cases:
            try:
                report = run_case(observatory, case, output_dir)
                # Recompute compact threshold tables from the actual case data so
                # threshold tuning is aggregated across cases rather than tuned on
                # one case. The residuals themselves are not persisted to avoid huge JSON files.
                raw = fetch_intermagnet_iaga2002(
                    observatory=observatory,
                    start_date=case.start_date,
                    duration_days=case.days,
                    samples_per_day="Minute",
                )
                df = parse_iaga2002_to_dataframe(raw)
                series = pd.to_numeric(df["f_nt"], errors="coerce")
                idx = series.index
                cadence = float(idx.to_series().diff().dropna().dt.total_seconds().median())
                _, residual = compute_qdc_baseline(series.to_numpy(dtype=float), cadence)
                kp, dst, _ = fetch_global_reference(idx[0], idx[-1], len(idx), int(round(cadence)))
                refs = reference_masks(kp, dst)
                add_threshold_table(report, residual, refs, cadence)
                all_case_reports.append(report)
            except Exception as exc:
                failures.append({
                    "observatory": observatory,
                    "case": asdict(case),
                    "error": str(exc),
                })

    if not all_case_reports:
        raise RuntimeError("No suite cases completed successfully.")

    calibration = [
        r for r in all_case_reports
        if r["case"]["start_date"] in {c.start_date for c in calibration_cases}
    ]
    held_out = [
        r for r in all_case_reports
        if r["case"]["start_date"] in {c.start_date for c in test_cases}
    ]
    if not calibration or not held_out:
        raise RuntimeError("Insufficient completed cases for calibration/held-out evaluation.")

    thresholds = choose_thresholds(calibration)

    # The held-out evaluation intentionally reports production thresholds AND the
    # calibration-selected candidate so we can compare without changing production.
    aggregate_active = []
    aggregate_storm = []
    baseline = []
    coverage = []
    for report in held_out:
        aggregate_active.append(report["production_scores"]["active"]["sample_level"])
        aggregate_storm.append(report["production_scores"]["storm"]["sample_level"])
        baseline.append(report["baseline_quality_nt"])
        coverage.append(report["reference_coverage"])

    summary = {
        "suite": {
            "years": list(years),
            "window_days": window_days,
            "cases_per_class": cases_per_class,
            "observatories": list(observatories),
            "total_discovered_cases": len(cases),
            "successful_cases": len(all_case_reports),
            "failed_cases": len(failures),
            "calibration_case_ids": [c.case_id for c in calibration_cases],
            "held_out_case_ids": [c.case_id for c in test_cases],
        },
        "production_thresholds": {
            "active_nt": PROD_ACTIVE_NT,
            "storm_nt": PROD_MINOR_STORM_NT,
        },
        "calibration_selected_thresholds": thresholds,
        "held_out_test_results": {
            "active_sample_level": aggregate_binary_metrics(aggregate_active),
            "storm_sample_level": aggregate_binary_metrics(aggregate_storm),
            "mean_baseline_mae_nt": nanmean_or_none(r["mae"] for r in baseline),
            "mean_baseline_rmse_nt": nanmean_or_none(r["rmse"] for r in baseline),
            "median_baseline_rmse_nt": nanmedian_or_none(r["rmse"] for r in baseline),
            "mean_quiet_rms_nt": nanmean_or_none(r["quiet_rms"] for r in baseline),
            "mean_completeness": nanmean_or_none(r["completeness"] for r in [
                {"completeness": x.get("reference_coverage")} for x in []
            ]),
            "reference_coverage_mean": nanmean_or_none(r["reference_coverage"] for r in coverage),
            "kp_coverage_mean": nanmean_or_none(r["kp_coverage"] for r in coverage),
            "dst_coverage_mean": nanmean_or_none(r["dst_coverage"] for r in coverage),
        },
        "failures": failures,
        "case_reports": [
            {
                "case": r["case"],
                "observatory": r["observatory"],
                "baseline_quality_nt": r["baseline_quality_nt"],
                "reference_coverage": r["reference_coverage"],
                "production_scores": r["production_scores"],
            }
            for r in all_case_reports
        ],
    }

    path = output_dir / "magnetometer_production_validation.json"
    path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 86)
    print("MAGNETOMETER PRODUCTION VALIDATION")
    print("=" * 86)
    print(f"Observatories:             {', '.join(observatories)}")
    print(f"Years:                     {min(years)}-{max(years)}")
    print(f"Discovered cases:          {len(cases)}")
    print(f"Successful cases:          {len(all_case_reports)}")
    print(f"Failed cases:              {len(failures)}")
    print(f"Calibration cases:         {len(calibration_cases)}")
    print(f"Held-out test cases:       {len(test_cases)}")
    print("-" * 86)
    print("Production thresholds (UNCHANGED)")
    print(f"  Active:                  > {PROD_ACTIVE_NT:.0f} nT")
    print(f"  Storm:                   > {PROD_MINOR_STORM_NT:.0f} nT")
    print("-" * 86)
    print("Calibration-selected candidates (NOT applied automatically)")
    print(f"  Active candidate:        > {thresholds['active_threshold_nt']:.0f} nT")
    print(f"  Storm candidate:         > {thresholds['storm_threshold_nt']:.0f} nT")
    print("-" * 86)
    active = summary["held_out_test_results"]["active_sample_level"]
    storm = summary["held_out_test_results"]["storm_sample_level"]
    print("HELD-OUT TEST — SAMPLE LEVEL")
    print(f"  Active precision:        {active['precision'] if active['precision'] is not None else 'N/A'}")
    print(f"  Active recall:           {active['recall'] if active['recall'] is not None else 'N/A'}")
    print(f"  Active F1:               {active['f1'] if active['f1'] is not None else 'N/A'}")
    print(f"  Active false alarm:      {active['false_alarm_rate'] if active['false_alarm_rate'] is not None else 'N/A'}")
    print(f"  Storm precision:         {storm['precision'] if storm['precision'] is not None else 'N/A'}")
    print(f"  Storm recall:            {storm['recall'] if storm['recall'] is not None else 'N/A'}")
    print(f"  Storm F1:                {storm['f1'] if storm['f1'] is not None else 'N/A'}")
    print(f"  Storm false alarm:       {storm['false_alarm_rate'] if storm['false_alarm_rate'] is not None else 'N/A'}")
    print("-" * 86)
    print(f"Mean reference coverage:   {summary['held_out_test_results']['reference_coverage_mean']}")
    print(f"Mean Kp coverage:          {summary['held_out_test_results']['kp_coverage_mean']}")
    print(f"Mean Dst coverage:         {summary['held_out_test_results']['dst_coverage_mean']}")
    print("-" * 86)
    print(f"JSON report:               {path}")
    print("=" * 86)

    return summary


# ---------------------------------------------------------------------------
# Single-case report
# ---------------------------------------------------------------------------
def run_single_case(
    observatory: str,
    start_date: str,
    days: int,
    output_dir: Path,
) -> Dict[str, Any]:
    case = SuiteCase("single_case", start_date, days, "user")
    report = run_case(observatory, case, output_dir)

    path = output_dir / f"magnetometer_performance_{observatory}_{start_date}_{days}d.json"
    path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 76)
    print("MAGNETOMETER PERFORMANCE — SINGLE CASE")
    print("=" * 76)
    print(f"Observatory:              {observatory}")
    print(f"Period:                   {start_date} ({days} days)")
    print(f"Completeness:             {report['completeness'] * 100:.2f}%")
    q = report["baseline_quality_nt"]
    print(f"Residual MAE:             {q['mae']:.3f} nT")
    print(f"Residual RMSE/RMS:        {q['rmse']:.3f} nT")
    print(f"Residual bias:            {q['bias']:.3f} nT")
    print(f"Residual median |e|:      {q['median_absolute_error']:.3f} nT")
    print(f"Residual P95 |e|:         {q['p95_absolute_error']:.3f} nT")
    print(f"Quiet-period RMS:         {q['quiet_rms']:.3f} nT" if q['quiet_rms'] is not None else "Quiet-period RMS:         N/A")
    print("-" * 76)
    print(f"Reference coverage:       {report['reference_coverage']['reference_coverage'] * 100:.2f}%")
    print(f"Kp coverage:              {report['reference_coverage']['kp_coverage'] * 100:.2f}%")
    print(f"Dst coverage:             {report['reference_coverage']['dst_coverage'] * 100:.2f}%")
    a = report["production_scores"]["active"]["sample_level"]
    s = report["production_scores"]["storm"]["sample_level"]
    print("-" * 76)
    print("PRODUCTION THRESHOLDS")
    print(f"Active precision:         {a['precision'] if a['precision'] is not None else 'N/A'}")
    print(f"Active recall:            {a['recall'] if a['recall'] is not None else 'N/A'}")
    print(f"Active F1:                {a['f1'] if a['f1'] is not None else 'N/A'}")
    print(f"Storm precision:          {s['precision'] if s['precision'] is not None else 'N/A'}")
    print(f"Storm recall:             {s['recall'] if s['recall'] is not None else 'N/A'}")
    print(f"Storm F1:                 {s['f1'] if s['f1'] is not None else 'N/A'}")
    print(f"Storm false alarm rate:   {s['false_alarm_rate'] if s['false_alarm_rate'] is not None else 'N/A'}")
    print("-" * 76)
    print(f"Analysis throughput:      {report['performance']['analysis_samples_per_second']:,.0f} samples/s")
    print(f"Realtime factor:          {report['performance']['realtime_factor']:,.0f}x")
    print(f"JSON report:              {path}")
    print("=" * 76)
    return report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    assert binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))["f1"] == 0.5
    events = bool_events(np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool), 60, merge_gap_s=60, min_duration_s=60)
    assert events == [(1, 6)]
    matched = match_events([(1, 5)], [(2, 6)], 60)
    assert matched["matched_events"] == 1
    assert matched["precision"] == 1.0
    assert matched["recall"] == 1.0
    assert threshold_discovery_score(0.9, 0.8) == 0.9
    print("MAGNETOMETER PRODUCTION VALIDATION SELF-TEST: PASS")


def threshold_discovery_score(f1: Optional[float], fallback: float) -> float:
    return float(f1) if f1 is not None else float(fallback)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-grade validation for the magnetometer QDC/activity pipeline."
    )
    parser.add_argument("--production-suite", action="store_true", help="Run the multi-period calibration/held-out validation suite.")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic benchmark self-tests.")
    parser.add_argument("--observatory", default="VIC", help="Single observatory for case mode, or comma-separated observatories for production suite.")
    parser.add_argument("--start-date", default="2024-03-15", help="Single-case start date.")
    parser.add_argument("--days", type=int, default=7, help="Single-case duration in days.")
    parser.add_argument("--suite-years", default=','.join(map(str, DEFAULT_YEARS)), help="Comma-separated years for Kp-driven case discovery.")
    parser.add_argument("--cases-per-class", type=int, default=DEFAULT_CASES_PER_CLASS, help="Quiet/active/storm cases to discover per class.")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Duration of each discovered case.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    output_dir = Path(args.output_dir).resolve()

    if args.production_suite:
        observatories = [item.strip().upper() for item in args.observatory.split(",") if item.strip()]
        years = tuple(int(item.strip()) for item in args.suite_years.split(",") if item.strip())
        run_suite(
            observatories=observatories,
            years=years,
            cases_per_class=args.cases_per_class,
            window_days=args.window_days,
            output_dir=output_dir,
        )
        return

    run_single_case(
        observatory=args.observatory.upper(),
        start_date=args.start_date,
        days=args.days,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
