#!/usr/bin/env python3
"""Strict exact-profile release gate for the causal-disturbance detector.

The gate evaluates the certified profile on validation first. The final-test
split is not fetched until validation, coverage, and confidence requirements
pass. No profile parameters are tuned here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import calibrate_detector as cd
from . import production_grade_validation as pg
from .detector_core import DETECTOR_VERSION, DetectorProfile

MIN_STATION_CASES = 4
MIN_POOLED_CASES_PER_YEAR = 8
MIN_CASES_PER_CLASS_PER_SPLIT = 12
MIN_REFERENCE_COVERAGE = 0.99
MIN_COMPLETENESS = 0.995
MIN_ACTIVE_PRECISION = 0.85
MIN_ACTIVE_RECALL = 0.80
MIN_ACTIVE_F1 = 0.82
MIN_STORM_PRECISION = 0.85
MIN_STORM_RECALL = 0.80
MIN_STORM_F1 = 0.82
MAX_STORM_FAR = 0.01
MIN_STORM_EVENT_PRECISION = 0.85
MIN_STORM_EVENT_RECALL = 0.90
MIN_STORM_EVENT_F1 = 0.87
MIN_STORM_F1_CI = 0.75
MIN_STORM_RECALL_CI = 0.70
MIN_STORM_EVENT_F1_CI = 0.75
BOOTSTRAPS = 2000
SEED = 20260820


def _prepare(rows: Sequence[dict[str, Any]]) -> list[cd.PreparedCase]:
    return [cd.PreparedCase(row["observatory"], pg.Case(**row["case"]), row) for row in rows]


def _aggregate(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "active": cd._aggregate_binary([x["active"] for x in scores]),
        "storm": cd._aggregate_binary([x["storm"] for x in scores]),
        "active_event": cd._merge_events([x["active_event"] for x in scores]),
        "storm_event": cd._merge_events([x["storm_event"] for x in scores]),
    }


def _score_rows(rows: Sequence[dict[str, Any]], profile: DetectorProfile) -> list[dict[str, Any]]:
    prepared = _prepare(rows)
    out = []
    for row, case in zip(rows, prepared):
        active, storm, active_event, storm_event = cd._score_case(case, profile)
        out.append({"active": active, "storm": storm, "active_event": active_event, "storm_event": storm_event, "case": row["case"], "observatory": row["observatory"]})
    return out


def _bootstrap(scores: Sequence[dict[str, Any]], metric: str, kind: str, seed: int) -> dict[str, float | None]:
    if not scores:
        return {"lower": None, "median": None, "upper": None}
    rng = np.random.default_rng(seed)
    n = len(scores)
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        sample = [scores[int(i)] for i in rng.integers(0, n, size=n)]
        if metric == "event_f1":
            value = cd._merge_events([x[f"{kind}_event"] for x in sample])["f1"]
        else:
            value = cd._aggregate_binary([x[kind] for x in sample]).get(metric)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {"lower": None, "median": None, "upper": None}
    arr = np.asarray(values, dtype=float)
    return {"lower": float(np.percentile(arr, 2.5)), "median": float(np.percentile(arr, 50)), "upper": float(np.percentile(arr, 97.5))}


def _passes(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agg = _aggregate(scores)
    checks = {
        "active_precision": (agg["active"]["precision"] or 0.0) >= MIN_ACTIVE_PRECISION,
        "active_recall": (agg["active"]["recall"] or 0.0) >= MIN_ACTIVE_RECALL,
        "active_f1": (agg["active"]["f1"] or 0.0) >= MIN_ACTIVE_F1,
        "storm_precision": (agg["storm"]["precision"] or 0.0) >= MIN_STORM_PRECISION,
        "storm_recall": (agg["storm"]["recall"] or 0.0) >= MIN_STORM_RECALL,
        "storm_f1": (agg["storm"]["f1"] or 0.0) >= MIN_STORM_F1,
        "storm_false_alarm_rate": (agg["storm"]["far"] or 1.0) <= MAX_STORM_FAR,
        "storm_event_precision": (agg["storm_event"]["precision"] or 0.0) >= MIN_STORM_EVENT_PRECISION,
        "storm_event_recall": (agg["storm_event"]["recall"] or 0.0) >= MIN_STORM_EVENT_RECALL,
        "storm_event_f1": (agg["storm_event"]["f1"] or 0.0) >= MIN_STORM_EVENT_F1,
    }
    storm_f1_ci = _bootstrap(scores, "f1", "storm", SEED)
    storm_recall_ci = _bootstrap(scores, "recall", "storm", SEED + 1)
    storm_event_ci = _bootstrap(scores, "event_f1", "storm", SEED + 2)
    checks["storm_f1_ci_lower"] = (storm_f1_ci["lower"] or 0.0) >= MIN_STORM_F1_CI
    checks["storm_recall_ci_lower"] = (storm_recall_ci["lower"] or 0.0) >= MIN_STORM_RECALL_CI
    checks["storm_event_f1_ci_lower"] = (storm_event_ci["lower"] or 0.0) >= MIN_STORM_EVENT_F1_CI
    return {"passed": all(checks.values()), "score": agg, "checks": checks, "storm_f1_ci": storm_f1_ci, "storm_recall_ci": storm_recall_ci, "storm_event_f1_ci": storm_event_ci}


def _coverage(rows: Sequence[dict[str, Any]], observatories: Sequence[str], splits: dict[str, list[int]]) -> dict[str, Any]:
    counts = Counter((r["observatory"], r["case"]["split"], int(r["case"]["year"]), r["case"]["class_name"]) for r in rows)
    shortages: list[dict[str, Any]] = []
    for obs in observatories:
        for split in ("calibration", "validation", "test"):
            for year in splits.get(split, []):
                for cls in ("quiet", "active", "storm"):
                    usable = counts[(obs, split, int(year), cls)]
                    if usable < MIN_STATION_CASES:
                        shortages.append({"type": "station_minimum", "observatory": obs, "split": split, "year": year, "class": cls, "usable": usable, "required": MIN_STATION_CASES})
    for split in ("calibration", "validation", "test"):
        for year in splits.get(split, []):
            for cls in ("quiet", "active", "storm"):
                usable = sum(counts[(obs, split, int(year), cls)] for obs in observatories)
                if usable < MIN_POOLED_CASES_PER_YEAR:
                    shortages.append({"type": "pooled_year_minimum", "split": split, "year": year, "class": cls, "usable": usable, "required": MIN_POOLED_CASES_PER_YEAR})
    return {"passed": not shortages, "shortages": shortages}


def _load(observatories: Sequence[str], cases: Sequence[pg.Case]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for obs in observatories:
        for case in cases:
            try:
                rows.append(pg.load_case(obs, case))
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-profile production release gate for the causal-disturbance detector.")
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--cases-per-class-per-year", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--profile-path", default=str(Path(__file__).resolve().with_name("detector_profile.json")))
    args = parser.parse_args()

    if args.cases_per_class_per_year < 10:
        raise SystemExit("Release gate requires --cases-per-class-per-year >= 10.")
    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    if len(years) < 3:
        raise SystemExit("Release gate requires at least three chronological years.")

    profile_path = Path(args.profile_path).resolve()
    if not profile_path.exists():
        raise SystemExit(f"FAIL: certified profile missing: {profile_path}")
    payload = json.loads(profile_path.read_text())
    if payload.get("status") != "certified":
        raise SystemExit("FAIL: detector profile is not certified; final-test holdout remains untouched.")
    if payload.get("detector_version") != DETECTOR_VERSION:
        raise SystemExit(f"FAIL: profile targets {payload.get('detector_version')!r}, expected {DETECTOR_VERSION!r}")
    profile = DetectorProfile.from_dict(payload.get("profile", payload))
    profile.validate()

    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    calibration_cases = [c for c in cases if c.split == "calibration"]
    validation_cases = [c for c in cases if c.split == "validation"]
    test_cases = [c for c in cases if c.split == "test"]

    print("=" * 90)
    print("CAUSAL MAGNETOMETER EXACT-PROFILE PRODUCTION RELEASE GATE")
    print("=" * 90)
    print(f"Detector version:  {DETECTOR_VERSION}")
    print(f"Calibration years: {splits['calibration']}")
    print(f"Validation years:  {splits['validation']}")
    print(f"Final-test years:  {splits['test']}")
    print("Final-test access: BLOCKED until validation passes")

    calibration_rows, calibration_failures = _load(observatories, calibration_cases)
    validation_rows, validation_failures = _load(observatories, validation_cases)
    pre_rows = calibration_rows + validation_rows
    coverage = _coverage(pre_rows, observatories, splits)

    validation_scores = _score_rows(validation_rows, profile)
    validation_gate = _passes(validation_scores) if validation_scores else {"passed": False, "score": None, "checks": {}}
    validation_gate["coverage_policy"] = _coverage(validation_rows, observatories, splits)["passed"]
    validation_gate["data_failures"] = len(validation_failures)
    validation_gate["passed"] = bool(validation_gate["passed"] and validation_gate["coverage_policy"] and not validation_failures)

    calibration_scores = _score_rows(calibration_rows, profile)
    calibration_gate = _passes(calibration_scores) if calibration_scores else {"passed": False, "score": None, "checks": {}}
    calibration_gate["data_failures"] = len(calibration_failures)
    calibration_gate["passed"] = bool(calibration_gate["passed"] and not calibration_failures)

    if not calibration_gate["passed"] or not validation_gate["passed"]:
        report = {
            "schema_version": "6.0-causal-disturbance-v2.1",
            "release_status": "FAIL",
            "detector_version": DETECTOR_VERSION,
            "profile": {"path": str(profile_path), "values": asdict(profile)},
            "splits": splits,
            "coverage": coverage,
            "calibration": calibration_gate,
            "validation": validation_gate,
            "final_test": {"accessed": False, "reason": "calibration or validation failed; holdout not fetched"},
            "failures": calibration_failures + validation_failures,
        }
        out = Path(__file__).resolve().with_name("data") / "magnetometer_exact_profile_release_gate.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print("RELEASE GATE: FAIL — final-test holdout untouched")
        raise SystemExit(2)

    test_rows, test_failures = _load(observatories, test_cases)
    test_coverage = _coverage(test_rows, observatories, splits)
    test_scores = _score_rows(test_rows, profile)
    final_gate = _passes(test_scores) if test_scores else {"passed": False, "score": None, "checks": {}}
    final_gate["coverage_policy"] = test_coverage["passed"]
    final_gate["data_failures"] = len(test_failures)
    final_gate["passed"] = bool(final_gate["passed"] and final_gate["coverage_policy"] and not test_failures)

    report = {
        "schema_version": "6.0-causal-disturbance-v2.1",
        "release_status": "PASS" if final_gate["passed"] else "FAIL",
        "detector_version": DETECTOR_VERSION,
        "profile": {"path": str(profile_path), "values": asdict(profile)},
        "splits": splits,
        "coverage": {"pretest": coverage, "final_test": test_coverage},
        "calibration": calibration_gate,
        "validation": validation_gate,
        "final_test": {"accessed": True, **final_gate},
        "failures": calibration_failures + validation_failures + test_failures,
    }
    out = Path(__file__).resolve().with_name("data") / "magnetometer_exact_profile_release_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RELEASE GATE: {'PASS' if final_gate['passed'] else 'FAIL'}")
    print(f"Report: {out}")
    raise SystemExit(0 if final_gate["passed"] else 2)


if __name__ == "__main__":
    main()
