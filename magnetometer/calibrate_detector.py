#!/usr/bin/env python3
"""Chronological production calibration for the causal-disturbance detector."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import production_grade_validation as pg
from .detector_core import DETECTOR_VERSION, DetectorProfile, detect_activity_masks

MIN_PRECISION = 0.85
MIN_RECALL = 0.80
MIN_F1 = 0.82
MAX_STORM_FAR = 0.01
MIN_EVENT_PRECISION = 0.85
MIN_EVENT_RECALL = 0.90
MIN_EVENT_F1 = 0.87
DEFAULT_WORKERS = 6
CACHE_NAMESPACE = "case_cache_causal_v8"

ACTIVE_THRESHOLDS = (15.0, 17.5, 20.0, 25.0, 30.0, 35.0, 40.0)
STORM_THRESHOLDS = (35.0, 40.0, 50.0, 60.0, 75.0, 100.0)
ACTIVE_ON = (2.0, 3.0, 5.0, 10.0)
ACTIVE_OFF = (20.0, 30.0, 45.0, 60.0)
STORM_ON = (5.0, 10.0, 15.0, 20.0)
STORM_OFF = (90.0, 120.0, 180.0, 240.0)


class PreparedCase:
    __slots__ = ("observatory", "case", "residual", "cadence_s", "known", "active_ref", "storm_ref")

    def __init__(self, observatory: str, case: pg.Case, data: dict[str, Any]) -> None:
        self.observatory = observatory
        self.case = case
        self.residual = np.asarray(data["residual"], dtype=float)
        self.cadence_s = float(data["cadence_s"])
        refs = data["refs"]
        self.known = np.asarray(refs["known"], dtype=bool)
        self.active_ref = np.asarray(refs["active"], dtype=bool)
        self.storm_ref = np.asarray(refs["storm"], dtype=bool)


def _binary(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=bool); truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth must have matching shapes")
    tp = int(np.sum(pred & truth)); tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _aggregate_binary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _binary_from_counts(sum(int(r["tp"]) for r in rows), sum(int(r["tn"]) for r in rows), sum(int(r["fp"]) for r in rows), sum(int(r["fn"]) for r in rows))


def _binary_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _event_metrics(predicted: np.ndarray, reference: np.ndarray, cadence_s: float) -> dict[str, Any]:
    pred_events = pg.pm.bool_events(predicted, cadence_s, merge_gap_s=1800, min_duration_s=300)
    ref_events = pg.pm.bool_events(reference, cadence_s, merge_gap_s=21600, min_duration_s=10800)
    return pg.pm.match_events(pred_events, ref_events, cadence_s)


def _merge_events(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reference = sum(int(r["reference_events"]) for r in rows); predicted = sum(int(r["predicted_events"]) for r in rows); matched = sum(int(r["matched_events"]) for r in rows)
    precision = matched / predicted if predicted else None; recall = matched / reference if reference else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": reference, "predicted_events": predicted, "matched_events": matched, "missed_events": max(0, reference - matched), "false_positive_events": max(0, predicted - matched), "precision": precision, "recall": recall, "f1": f1}


def _score_case(case: PreparedCase, profile: DetectorProfile):
    active, storm, _major, _severe, _diag = detect_activity_masks(case.residual, cadence_s=case.cadence_s, profile=profile, include_anomaly=False)
    known = case.known & np.isfinite(case.residual)
    return (_binary(active[known], case.active_ref[known]), _binary(storm[known], case.storm_ref[known]), _event_metrics(active & known, case.active_ref & known, case.cadence_s), _event_metrics(storm & known, case.storm_ref & known, case.cadence_s))


def _evaluate(cases: Sequence[PreparedCase], profile: DetectorProfile) -> dict[str, Any]:
    rows = [_score_case(case, profile) for case in cases]
    return {"active": _aggregate_binary([r[0] for r in rows]), "storm": _aggregate_binary([r[1] for r in rows]), "active_event": _merge_events([r[2] for r in rows]), "storm_event": _merge_events([r[3] for r in rows])}


def _violation(score: dict[str, Any]) -> float:
    a, s, e = score["active"], score["storm"], score["storm_event"]
    return float(sum((max(0.0, MIN_PRECISION - float(a["precision"] or 0.0)), max(0.0, MIN_RECALL - float(a["recall"] or 0.0)), max(0.0, MIN_F1 - float(a["f1"] or 0.0)), max(0.0, MIN_PRECISION - float(s["precision"] or 0.0)), max(0.0, MIN_RECALL - float(s["recall"] or 0.0)), max(0.0, MIN_F1 - float(s["f1"] or 0.0)), max(0.0, float(s["far"] or 1.0) - MAX_STORM_FAR), max(0.0, MIN_EVENT_PRECISION - float(e["precision"] or 0.0)), max(0.0, MIN_EVENT_RECALL - float(e["recall"] or 0.0)), max(0.0, MIN_EVENT_F1 - float(e["f1"] or 0.0))))


def _worst_case_floor(cases: Sequence[PreparedCase], profile: DetectorProfile) -> float:
    worst = 1.0
    for case in cases:
        a, s, _ae, se = _score_case(case, profile)
        worst = min(worst, float(a["precision"] or 0.0), float(a["recall"] or 0.0), float(s["precision"] or 0.0), float(s["recall"] or 0.0), float(se["precision"] or 0.0), float(se["recall"] or 0.0))
    return worst


def _candidate_key(score: dict[str, Any], cases: Sequence[PreparedCase], profile: DetectorProfile) -> tuple:
    quality = float(np.mean([float(score["active"]["f1"] or 0.0), float(score["storm"]["f1"] or 0.0), float(score["active_event"]["f1"] or 0.0), float(score["storm_event"]["f1"] or 0.0)]))
    return (round(_violation(score), 12), round(max(0.0, 0.75 - _worst_case_floor(cases, profile)), 12), -round(quality, 12), float(profile.active_on_minutes), float(profile.storm_on_minutes))


def _search(cases: Sequence[PreparedCase], base: DetectorProfile) -> DetectorProfile:
    candidates = []
    for active_nt in ACTIVE_THRESHOLDS:
        for storm_nt in STORM_THRESHOLDS:
            if storm_nt <= active_nt:
                continue
            candidate = replace(base, active_nt=active_nt, storm_nt=storm_nt)
            candidates.append((_candidate_key(_evaluate(cases, candidate), cases, candidate), candidate))
    profile = min(candidates, key=lambda item: item[0])[1] if candidates else base
    for name, values in (("active_on_minutes", ACTIVE_ON), ("active_off_minutes", ACTIVE_OFF), ("storm_on_minutes", STORM_ON), ("storm_off_minutes", STORM_OFF)):
        best = profile; best_key = _candidate_key(_evaluate(cases, best), cases, best)
        for value in values:
            candidate = replace(profile, **{name: value})
            try: candidate.validate()
            except ValueError: continue
            key = _candidate_key(_evaluate(cases, candidate), cases, candidate)
            if key < best_key: best, best_key = candidate, key
        profile = best
    return profile
