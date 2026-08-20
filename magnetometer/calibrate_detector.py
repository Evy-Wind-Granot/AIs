#!/usr/bin/env python3
"""Chronological production calibration for the causal-disturbance detector."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg
from detector_core import DETECTOR_VERSION, DetectorProfile, detect_activity_masks

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
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
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
    reference = sum(int(r["reference_events"]) for r in rows)
    predicted = sum(int(r["predicted_events"]) for r in rows)
    matched = sum(int(r["matched_events"]) for r in rows)
    precision = matched / predicted if predicted else None
    recall = matched / reference if reference else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": reference, "predicted_events": predicted, "matched_events": matched, "missed_events": max(0, reference - matched), "false_positive_events": max(0, predicted - matched), "precision": precision, "recall": recall, "f1": f1}


def _score_case(case: PreparedCase, profile: DetectorProfile) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    active, storm, _major, _severe, _diag = detect_activity_masks(case.residual, cadence_s=case.cadence_s, profile=profile, include_anomaly=False)
    known = case.known & np.isfinite(case.residual)
    active_sample = _binary(active[known], case.active_ref[known])
    storm_sample = _binary(storm[known], case.storm_ref[known])
    active_event = _event_metrics(active & known, case.active_ref & known, case.cadence_s)
    storm_event = _event_metrics(storm & known, case.storm_ref & known, case.cadence_s)
    return active_sample, storm_sample, active_event, storm_event


def _evaluate(cases: Sequence[PreparedCase], profile: DetectorProfile) -> dict[str, Any]:
    active_rows: list[dict[str, Any]] = []
    storm_rows: list[dict[str, Any]] = []
    active_events: list[dict[str, Any]] = []
    storm_events: list[dict[str, Any]] = []
    for case in cases:
        active, storm, active_event, storm_event = _score_case(case, profile)
        active_rows.append(active); storm_rows.append(storm); active_events.append(active_event); storm_events.append(storm_event)
    return {"active": _aggregate_binary(active_rows), "storm": _aggregate_binary(storm_rows), "active_event": _merge_events(active_events), "storm_event": _merge_events(storm_events)}


def _violation(score: dict[str, Any]) -> float:
    a, s, e = score["active"], score["storm"], score["storm_event"]
    return float(sum((
        max(0.0, MIN_PRECISION - float(a["precision"] or 0.0)),
        max(0.0, MIN_RECALL - float(a["recall"] or 0.0)),
        max(0.0, MIN_F1 - float(a["f1"] or 0.0)),
        max(0.0, MIN_PRECISION - float(s["precision"] or 0.0)),
        max(0.0, MIN_RECALL - float(s["recall"] or 0.0)),
        max(0.0, MIN_F1 - float(s["f1"] or 0.0)),
        max(0.0, float(s["far"] or 1.0) - MAX_STORM_FAR),
        max(0.0, MIN_EVENT_PRECISION - float(e["precision"] or 0.0)),
        max(0.0, MIN_EVENT_RECALL - float(e["recall"] or 0.0)),
        max(0.0, MIN_EVENT_F1 - float(e["f1"] or 0.0)),
    )))


def _worst_case_floor(cases: Sequence[PreparedCase], profile: DetectorProfile) -> float:
    worst = 1.0
    for case in cases:
        a, s, _ae, se = _score_case(case, profile)
        worst = min(worst, float(a["precision"] or 0.0), float(a["recall"] or 0.0), float(s["precision"] or 0.0), float(s["recall"] or 0.0), float(se["precision"] or 0.0), float(se["recall"] or 0.0))
    return worst


def _candidate_key(score: dict[str, Any], cases: Sequence[PreparedCase], profile: DetectorProfile) -> tuple:
    violation = _violation(score)
    worst_penalty = max(0.0, 0.75 - _worst_case_floor(cases, profile))
    quality = float(np.mean([float(score["active"]["f1"] or 0.0), float(score["storm"]["f1"] or 0.0), float(score["active_event"]["f1"] or 0.0), float(score["storm_event"]["f1"] or 0.0)]))
    return (round(violation, 12), round(worst_penalty, 12), -round(quality, 12), float(profile.active_on_minutes), float(profile.storm_on_minutes))


def _search(cases: Sequence[PreparedCase], base: DetectorProfile) -> DetectorProfile:
    candidates: list[tuple[tuple, DetectorProfile]] = []
    for active_nt in ACTIVE_THRESHOLDS:
        for storm_nt in STORM_THRESHOLDS:
            if storm_nt <= active_nt:
                continue
            candidate = replace(base, active_nt=active_nt, storm_nt=storm_nt)
            candidates.append((_candidate_key(_evaluate(cases, candidate), cases, candidate), candidate))
    profile = min(candidates, key=lambda item: item[0])[1] if candidates else base
    for name, values in (("active_on_minutes", ACTIVE_ON), ("active_off_minutes", ACTIVE_OFF), ("storm_on_minutes", STORM_ON), ("storm_off_minutes", STORM_OFF)):
        best = profile
        best_key = _candidate_key(_evaluate(cases, best), cases, best)
        for value in values:
            candidate = replace(profile, **{name: value})
            try:
                candidate.validate()
            except ValueError:
                continue
            key = _candidate_key(_evaluate(cases, candidate), cases, candidate)
            if key < best_key:
                best, best_key = candidate, key
        profile = best
    return profile


def _passes(score: dict[str, Any]) -> bool:
    for name in ("active", "storm"):
        row = score[name]
        if (row["precision"] or 0.0) < MIN_PRECISION or (row["recall"] or 0.0) < MIN_RECALL or (row["f1"] or 0.0) < MIN_F1:
            return False
    if (score["storm"]["far"] or 1.0) > MAX_STORM_FAR:
        return False
    event = score["storm_event"]
    return (event["precision"] or 0.0) >= MIN_EVENT_PRECISION and (event["recall"] or 0.0) >= MIN_EVENT_RECALL and (event["f1"] or 0.0) >= MIN_EVENT_F1


def _load_one(observatory: str, case: pg.Case) -> tuple[str, pg.Case, dict[str, Any]]:
    return observatory, case, pg.load_case(observatory, case)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict chronological calibration for the production magnetometer detector.")
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--cases-per-class-per-year", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    args = parser.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    if len(years) < 3:
        raise SystemExit("At least three chronological years are required.")
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Production calibration requires at least 10 target cases per class per year.")

    pg.DEFAULT_CACHE_DIR = HERE / "data" / CACHE_NAMESPACE
    pg.DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year * 2, args.window_days)
    cases = [c for c in cases if c.split in ("calibration", "validation")]

    master_kp = pg._fetch_kp_cached(f"{min(years):04d}-01-01", f"{max(years):04d}-12-31")
    pg._fetch_kp_cached = lambda _start, _end: master_kp
    months: set[tuple[int, int]] = set()
    for case in cases:
        start = pd.Timestamp(case.start_date, tz="UTC")
        end = start + pd.Timedelta(days=case.days - 1)
        months.update((p.year, p.month) for p in pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"))
    for year, month in sorted(months):
        pg._fetch_dst_cached(int(year), int(month))

    tasks = [(obs, case) for obs in observatories for case in cases]
    successes: list[tuple[str, pg.Case, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    workers = max(1, min(int(args.workers), 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one, obs, case): (obs, case) for obs, case in tasks}
        for future in as_completed(futures):
            obs, case = futures[future]
            try:
                _, _, data = future.result()
                successes.append((obs, case, data))
                print(f"OK {obs} {case.case_id}", flush=True)
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
                print(f"SKIP {obs} {case.case_id}: {exc}", flush=True)

    from collections import Counter
    counts = Counter((obs, c.split, int(c.year), c.class_name) for obs, c, _ in successes)
    shortages: list[dict[str, Any]] = []
    for obs in observatories:
        for split in ("calibration", "validation"):
            for year in splits[split]:
                for cls in ("quiet", "active", "storm"):
                    usable = counts[(obs, split, int(year), cls)]
                    if usable < 4:
                        shortages.append({"type": "station_year_class", "observatory": obs, "split": split, "year": year, "class": cls, "usable": usable, "required": 4})
    for split in ("calibration", "validation"):
        for year in splits[split]:
            for cls in ("quiet", "active", "storm"):
                usable = sum(counts[(obs, split, int(year), cls)] for obs in observatories)
                if usable < 8:
                    shortages.append({"type": "pooled_year_class", "split": split, "year": year, "class": cls, "usable": usable, "required": 8})

    if shortages:
        output = {"status": "blocked", "detector_version": DETECTOR_VERSION, "reason": "insufficient independent event coverage after data-quality filtering", "shortages": shortages, "failed_source_cases": failures}
        Path(args.profile_path).resolve().with_suffix(".blocked.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2))
        raise SystemExit(2)

    used = Counter()
    selected: list[tuple[str, pg.Case, dict[str, Any]]] = []
    for obs, case, data in sorted(successes, key=lambda x: (x[0], x[1].split, x[1].year, x[1].class_name, x[1].center_date)):
        key = (obs, case.split, int(case.year), case.class_name)
        if used[key] < args.cases_per_class_per_year:
            selected.append((obs, case, data))
            used[key] += 1

    calibration = [PreparedCase(obs, case, data) for obs, case, data in selected if case.split == "calibration"]
    validation = [PreparedCase(obs, case, data) for obs, case, data in selected if case.split == "validation"]
    print(f"Prepared {len(calibration)} calibration and {len(validation)} validation cases.", flush=True)

    profile = _search(calibration, DetectorProfile())
    profile.validate()
    calibration_score = _evaluate(calibration, profile)
    validation_score = _evaluate(validation, profile)
    calibration_passed = _passes(calibration_score)
    validation_passed = _passes(validation_score)
    certified = calibration_passed and validation_passed

    output = {
        "status": "certified" if certified else "candidate",
        "detector_version": DETECTOR_VERSION,
        "profile": asdict(profile),
        "sampling_policy": {"target_cases_per_station_year": args.cases_per_class_per_year, "minimum_station_cases": 4, "minimum_pooled_cases_per_year": 8, "under_supplied_years_retain_all_independent_usable_cases": True},
        "selection": {"calibration_years": splits["calibration"], "validation_years": splits["validation"], "final_test_years": splits["test"], "final_test_used": False, "candidate_pool_per_class_per_year": args.cases_per_class_per_year * 2},
        "calibration": calibration_score,
        "calibration_passed": calibration_passed,
        "validation": validation_score,
        "validation_passed": validation_passed,
        "passed_validation": validation_passed,
        "failed_source_cases": failures,
        "certification_policy": {"search": "bounded absolute-threshold search followed by bounded persistence refinement", "production_floors": {"sample_precision": MIN_PRECISION, "sample_recall": MIN_RECALL, "sample_f1": MIN_F1, "storm_false_alarm_rate": MAX_STORM_FAR, "storm_event_precision": MIN_EVENT_PRECISION, "storm_event_recall": MIN_EVENT_RECALL, "storm_event_f1": MIN_EVENT_F1}, "calibration_must_pass": True, "validation_must_pass": True, "final_test_used": False},
    }

    path = Path(args.profile_path).resolve()
    if certified:
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"CERTIFIED detector profile written to {path}", flush=True)
        raise SystemExit(0)

    candidate = path.with_suffix(".candidate.json")
    candidate.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Certification blocked; certified profile was NOT replaced. Candidate: {candidate}", flush=True)
    print(json.dumps(output, indent=2))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
