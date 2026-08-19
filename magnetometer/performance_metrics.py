#!/usr/bin/env python3
"""Production magnetometer metrics and validation primitives.

The scoring path uses the same deterministic production detector as the live
demo. Calibration may supply active/storm thresholds; the final test set is
never used to choose them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from detector_core import detect_activity_masks  # noqa: E402
from magnetometer_demo import (  # noqa: E402
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


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def finite_values(values: np.ndarray) -> np.ndarray:
    return finite(values)


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=bool); truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have matching shapes")
    tp = int(np.sum(pred & truth)); tn = int(np.sum(~pred & ~truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth)); total = tp + tn + fp + fn
    def div(a: float, b: float) -> Optional[float]: return float(a / b) if b else None
    precision = div(tp, tp + fp); recall = div(tp, tp + fn); specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {"samples": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "positive_truth_samples": tp + fn, "positive_prediction_samples": tp + fp, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "false_alarm_rate": div(fp, fp + tn), "miss_rate": div(fn, fn + tp), "accuracy": div(tp + tn, total), "balanced_accuracy": ((recall + specificity) / 2 if recall is not None and specificity is not None else None)}


def aggregate_binary(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]: return aggregate_binary_metrics(metrics)


def aggregate_binary_metrics(metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(m.get("tp", 0) or 0) for m in metrics); tn = sum(int(m.get("tn", 0) or 0) for m in metrics); fp = sum(int(m.get("fp", 0) or 0) for m in metrics); fn = sum(int(m.get("fn", 0) or 0) for m in metrics)
    return binary_metrics_from_counts(tp, tn, fp, fn)


def binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    total = tp + tn + fp + fn
    def div(a: float, b: float) -> Optional[float]: return float(a / b) if b else None
    precision = div(tp, tp + fp); recall = div(tp, tp + fn); specificity = div(tn, tn + fp); f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {"samples": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "positive_truth_samples": tp + fn, "positive_prediction_samples": tp + fp, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "false_alarm_rate": div(fp, fp + tn), "miss_rate": div(fn, fn + tp), "accuracy": div(tp + tn, total), "balanced_accuracy": ((recall + specificity) / 2 if recall is not None and specificity is not None else None)}


def compute_qdc_baseline(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float); n = len(x); baseline = np.zeros(n, dtype=float); weights = np.zeros(n, dtype=float); window_samples = max(1, int(24 * 3600 / cadence_s)); step_samples = max(1, window_samples // 2); t_global = np.arange(n, dtype=float) * cadence_s / 3600.0; t_min = float(t_global.min()) if n else 0.0; t_max = float(t_global.max()) if n else 0.0; last_good_coeffs = None
    for start in range(0, max(1, n - step_samples), step_samples):
        end = min(start + window_samples, n)
        if end - start < step_samples // 2: break
        segment = x[start:end]; t_seg = t_global[start:end]
        if np.isfinite(segment).sum() < (end - start) * 0.5: continue
        seg_base, coeffs = robust_harmonic_baseline(segment, cadence_s, t_hours=t_seg, t_ref_min=t_min, t_ref_max=t_max); seg_res = segment - seg_base; storm_frac = float(np.mean(np.abs(seg_res) > 50.0))
        if storm_frac > 0.05 and last_good_coeffs is not None: seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good_coeffs
        elif storm_frac <= 0.05: last_good_coeffs = coeffs
        w = np.hanning(end - start); baseline[start:end] += seg_base * w; weights[start:end] += w
    mask = weights > 0; fallback = float(np.median(finite(x))) if finite(x).size else 0.0; baseline[mask] /= weights[mask]; baseline[~mask] = fallback
    return baseline, x - baseline


def reference_masks(kp: pd.Series, dst: pd.Series) -> Dict[str, np.ndarray]:
    """Build independent environmental reference labels.

    NOAA's geomagnetic storm scale defines G1/minor storm conditions at Kp=5.
    Kp 3-4 is retained as the locally 'active' class; Kp>=5 is storm.
    Dst provides an independent disturbance reference when available.
    """
    kp_values = kp.to_numpy(dtype=float); dst_values = dst.to_numpy(dtype=float)
    kp_known = np.isfinite(kp_values); dst_known = np.isfinite(dst_values); known = kp_known | dst_known
    active = ((kp_known & (kp_values >= 4.0)) | (dst_known & (dst_values < -30.0))) & known
    storm = ((kp_known & (kp_values >= 5.0)) | (dst_known & (dst_values < -50.0))) & known
    return {"known": known, "kp_known": kp_known, "dst_known": dst_known, "active": active, "storm": storm}


def production_detection_masks(residual: np.ndarray, cadence_s: float, active_threshold: float, storm_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return certification masks from the shared deterministic detector."""
    active, storm, _major, _severe, _diagnostics = detect_activity_masks(residual, cadence_s=cadence_s, active_threshold=active_threshold, storm_threshold=storm_threshold, unsettled_threshold=PROD_UNSETTLED_NT, major_threshold=PROD_MAJOR_STORM_NT, severe_threshold=PROD_SEVERE_STORM_NT)
    return active, storm


def bool_events(mask: np.ndarray, cadence_s: float, merge_gap_s: float = 0.0, min_duration_s: float = 0.0) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).copy()
    if not mask.size: return []
    gap = max(0, int(round(merge_gap_s / cadence_s)))
    if gap:
        padded = np.r_[False, mask, False]; starts = np.flatnonzero(~padded[:-1] & padded[1:]); ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i in range(len(starts) - 1):
            if starts[i + 1] - ends[i] <= gap: mask[ends[i]:starts[i + 1]] = True
    padded = np.r_[False, mask, False]; starts = np.flatnonzero(~padded[:-1] & padded[1:]); ends = np.flatnonzero(padded[:-1] & ~padded[1:]); min_len = max(1, int(math.ceil(min_duration_s / cadence_s)))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int: return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(predicted: Sequence[Tuple[int, int]], reference: Sequence[Tuple[int, int]], cadence_s: float) -> Dict[str, Any]:
    pairs = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            ov = overlap(pred, ref)
            if ov > 0: pairs.append((ov, pi, ri))
    pairs.sort(reverse=True); used_p, used_r, matches = set(), set(), []
    for ov, pi, ri in pairs:
        if pi in used_p or ri in used_r: continue
        used_p.add(pi); used_r.add(ri); matches.append({"predicted_index": pi, "reference_index": ri, "overlap_seconds": float(ov * cadence_s)})
    tp = len(matches); fp = len(predicted) - tp; fn = len(reference) - tp; precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None; f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": len(reference), "predicted_events": len(predicted), "matched_events": tp, "missed_events": fn, "false_positive_events": fp, "precision": precision, "recall": recall, "f1": f1, "matches": matches}


def score_thresholds(residual: np.ndarray, refs: Dict[str, np.ndarray], cadence_s: float, active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    known = refs["known"] & np.isfinite(residual); pred_active, pred_storm = production_detection_masks(residual, cadence_s, active_threshold, storm_threshold); active_sample = binary_metrics(pred_active[known], refs["active"][known]); storm_sample = binary_metrics(pred_storm[known], refs["storm"][known]); active_events = match_events(bool_events(pred_active, cadence_s, 1800, 300), bool_events(refs["active"] & refs["known"], cadence_s, 21600, 10800), cadence_s); storm_events = match_events(bool_events(pred_storm, cadence_s, 1800, 300), bool_events(refs["storm"] & refs["known"], cadence_s, 21600, 10800), cadence_s)
    return {"active": {"threshold_nt": active_threshold, "sample_level": active_sample, "event_level": active_events}, "storm": {"threshold_nt": storm_threshold, "sample_level": storm_sample, "event_level": storm_events}}
