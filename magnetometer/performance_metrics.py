#!/usr/bin/env python3
"""
Production-grade magnetometer validation harness.

This benchmark evaluates the production magnetometer policy. It shares the
thresholds with magnetometer_demo.py and applies the same persistent detector
logic for scoring, so the release gate measures the actual production policy.
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

from magnetometer_demo import (
    ANOMALY_DELTA_NT,
    PROD_ACTIVE_NT,
    PROD_MAJOR_STORM_NT,
    PROD_MINOR_STORM_NT,
    PROD_SEVERE_STORM_NT,
    PROD_UNSETTLED_NT,
    build_design_matrix,
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    parse_iaga2002_to_dataframe,
    robust_harmonic_baseline,
)

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


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
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
        "positive_truth_samples": tp + fn,
        "positive_prediction_samples": tp + fp,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": div(tp + tn, total),
        "balanced_accuracy": (float((recall + specificity) / 2) if recall is not None and specificity is not None else None),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1,
        "false_alarm_rate": div(fp, fp + tn),
        "miss_rate": div(fn, fn + tp),
    }


def confusion_counts(metrics: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {key: int(sum(int(m.get(key, 0) or 0) for m in metrics)) for key in ("tp", "tn", "fp", "fn")}


def aggregate_binary_metrics(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = confusion_counts(metrics)
    tp, tn, fp, fn = counts.values()
    total = tp + tn + fp + fn

    def div(a: float, b: float) -> Optional[float]:
        return float(a / b) if b else None

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
        "balanced_accuracy": (float((recall + specificity) / 2) if recall is not None and specificity is not None else None),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1,
        "false_alarm_rate": div(fp, fp + tn),
        "miss_rate": div(fn, fn + tp),
    }


def compute_qdc_baseline(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
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
        seg_base, coeffs = robust_harmonic_baseline(segment, cadence_s, t_hours=t_seg, t_ref_min=t_min, t_ref_max=t_max)
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


def fetch_global_reference(start_time: pd.Timestamp, end_time: pd.Timestamp, n: int, cadence_s: float) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
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
    target_index = pd.date_range(start=start_time, periods=n, freq=pd.Timedelta(seconds=cadence_s), tz="UTC")
    tolerance = pd.Timedelta("3h")
    kp_aligned = kp.reindex(target_index, method="ffill", tolerance=tolerance) if not kp.empty else pd.Series(np.nan, index=target_index)
    dst_aligned = dst.reindex(target_index, method="ffill", tolerance=tolerance) if not dst.empty else pd.Series(np.nan, index=target_index)
    return kp_aligned, dst_aligned, {
        "kp_fetch_ok": kp_error is None,
        "kp_error": kp_error,
        "dst_months_requested": len(periods),
        "dst_months_with_data": len(dst_parts),
        "dst_failures": dst_failures,
    }


def reference_masks(kp: pd.Series, dst: pd.Series) -> Dict[str, np.ndarray]:
    kp_values = kp.to_numpy(dtype=float)
    dst_values = dst.to_numpy(dtype=float)
    kp_known = np.isfinite(kp_values)
    dst_known = np.isfinite(dst_values)
    known = kp_known | dst_known
    active = ((kp_known & (kp_values >= 4.0)) | (dst_known & (dst_values < -30.0))) & known
    storm = ((kp_known & (kp_values >= 6.0)) | (dst_known & (dst_values < -50.0))) & known
    return {"known": known, "kp_known": kp_known, "dst_known": dst_known, "active": active, "storm": storm}


def persistent_mask(mask: np.ndarray, min_samples: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or min_samples <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    start = None
    for i, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_samples:
                out[start:i] = True
            start = None
    return out


def production_detection_masks(residual: np.ndarray, cadence_s: float, active_threshold: float, storm_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(np.asarray(residual, dtype=float))
    smooth_samples = max(1, int(round(15 * 60 / max(cadence_s, 1.0))))
    smooth_samples = min(smooth_samples, 31)
    if smooth_samples % 2 == 0:
        smooth_samples += 1
    smooth = pd.Series(magnitude).rolling(smooth_samples, center=True, min_periods=1).median().to_numpy()
    active_min = max(1, int(round(10 * 60 / max(cadence_s, 1.0))))
    storm_min = max(1, int(round(15 * 60 / max(cadence_s, 1.0))))
    active = persistent_mask(smooth > active_threshold, active_min)
    storm = persistent_mask(smooth > storm_threshold, storm_min)
    return active, storm


def bool_events(mask: np.ndarray, cadence_s: float, merge_gap_s: float = 0.0, min_duration_s: float = 0.0) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0:
        return []
    max_gap = max(0, int(round(merge_gap_s / cadence_s)))
    if max_gap > 0:
        padded = np.r_[False, mask, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i in range(len(starts) - 1):
            if starts[i + 1] - ends[i] <= max_gap:
                mask[ends[i]:starts[i + 1]] = True
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    min_len = max(1, int(math.ceil(min_duration_s / cadence_s)))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(predicted: Sequence[Tuple[int, int]], reference: Sequence[Tuple[int, int]], cadence_s: float) -> Dict[str, Any]:
    candidates = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            ov = overlap(pred, ref)
            if ov > 0:
                candidates.append((ov, pi, ri))
    candidates.sort(reverse=True)
    used_pred, used_ref, matches = set(), set(), []
    for ov, pi, ri in candidates:
        if pi in used_pred or ri in used_ref:
            continue
        used_pred.add(pi); used_ref.add(ri)
        pred = predicted[pi]; ref = reference[ri]
        matches.append({"predicted_index": pi, "reference_index": ri, "overlap_seconds": float(ov * cadence_s)})
    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": len(reference), "predicted_events": len(predicted), "matched_events": tp, "missed_events": fn, "false_positive_events": fp, "precision": precision, "recall": recall, "f1": f1, "matches": matches}


def score_thresholds(residual: np.ndarray, refs: Dict[str, np.ndarray], cadence_s: float, active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    magnitude = np.abs(np.asarray(residual, dtype=float))
    known = refs["known"] & np.isfinite(magnitude)
    pred_active, pred_storm = production_detection_masks(magnitude, cadence_s, active_threshold, storm_threshold)
    active_sample = binary_metrics(pred_active[known], refs["active"][known])
    storm_sample = binary_metrics(pred_storm[known], refs["storm"][known])
    pred_active_events = bool_events(pred_active, cadence_s, merge_gap_s=30 * 60, min_duration_s=5 * 60)
    ref_active_events = bool_events(refs["active"] & refs["known"], cadence_s, merge_gap_s=6 * 3600, min_duration_s=3 * 3600)
    pred_storm_events = bool_events(pred_storm, cadence_s, merge_gap_s=30 * 60, min_duration_s=5 * 60)
    ref_storm_events = bool_events(refs["storm"] & refs["known"], cadence_s, merge_gap_s=6 * 3600, min_duration_s=3 * 3600)
    return {
        "active": {"threshold_nt": active_threshold, "sample_level": active_sample, "event_level": match_events(pred_active_events, ref_active_events, cadence_s)},
        "storm": {"threshold_nt": storm_threshold, "sample_level": storm_sample, "event_level": match_events(pred_storm_events, ref_storm_events, cadence_s)},
    }


def __getattr__(name: str):
    if name == "ANOMALY_DELTA_NT":
        return ANOMALY_DELTA_NT
    raise AttributeError(name)
