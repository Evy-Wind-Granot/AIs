#!/usr/bin/env python3
"""Strict causal detector production release gate v5.

The gate filters individual source/data-quality failures instead of allowing a
single unusable historical window to invalidate an otherwise adequately
covered validation split. Failed cases remain explicitly recorded and cannot
be silently scored. The final-test split remains inaccessible until validation
passes all production floors and confidence bounds.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from . import calibrate_detector as cd
from . import production_grade_validation as pg
from .detector_core import DetectorProfile

MIN_STATION_CASES = 4
MIN_POOLED_CASES_PER_YEAR = 8
MIN_CASES_PER_CLASS_PER_SPLIT = 12
MIN_REFERENCE_COVERAGE = 0.99
MIN_COMPLETENESS = 0.995
MIN_ACTIVE_PRECISION = 0.85
MIN_ACTIVE_RECALL = 0.80
MIN_ACTIVE_F1 = 0.82
MIN_STORM_PRECISION = 0.85
MIN_STORM_RECALL = 0.90
MIN_STORM_F1 = 0.87
MAX_STORM_FAR = 0.01
MIN_STORM_F1_CI = 0.75
MIN_STORM_RECALL_CI = 0.80
MIN_STORM_EVENT_F1_CI = 0.75
BOOTSTRAPS = 2000
SEED = 20260819


def _coverage_ok(rows):
    return bool(rows) and all(
        float(r.get("completeness", 0)) >= MIN_COMPLETENESS
        and float(r.get("reference_coverage", 0)) >= MIN_REFERENCE_COVERAGE
        for r in rows
    )


def _shortages(rows, observatories, splits, split_name):
    counts = Counter(
        (r["observatory"], int(r["case"]["year"]), r["case"]["class_name"])
        for r in rows if r["case"]["split"] == split_name
    )
    out = []
    for obs in observatories:
        for year in splits.get(split_name, []):
            for cls in ("quiet", "active", "storm"):
                n = counts[(obs, int(year), cls)]
                if n < MIN_STATION_CASES:
                    out.append({"type": "station_minimum", "observatory": obs, "split": split_name, "year": year, "class": cls, "usable": n, "required": MIN_STATION_CASES})
    for year in splits.get(split_name, []):
        for cls in ("quiet", "active", "storm"):
            n = sum(counts[(obs, int(year), cls)] for obs in observatories)
            if n < MIN_POOLED_CASES_PER_YEAR:
                out.append({"type": "pooled_year_minimum", "split": split_name, "year": year, "class": cls, "usable": n, "required": MIN_POOLED_CASES_PER_YEAR})
    for cls in ("quiet", "active", "storm"):
        n = sum(counts[(obs, int(year), cls)] for obs in observatories for year in splits.get(split_name, []))
        if n < MIN_CASES_PER_CLASS_PER_SPLIT:
            out.append({"type": "pooled_split_minimum", "split": split_name, "class": cls, "usable": n, "required": MIN_CASES_PER_CLASS_PER_SPLIT})
    return out


def _per_case_scores(rows, profile):
    output = []
    for row, case in zip(rows, [cd.PreparedCase(r) for r in rows]):
        active, storm, active_event, storm_event = cd._score_case(case, profile)
        output.append({"active": active, "storm": storm, "active_event": active_event, "storm_event": storm_event, "case": row["case"], "observatory": row["observatory"]})
    return output


def _aggregate(scores):
    return {
        "active": cd._counts([x["active"] for x in scores]),
        "storm": cd._counts([x["storm"] for x in scores]),
        "active_event": cd._merge_event_metrics([x["active_event"] for x in scores]),
        "storm_event": cd._merge_event_metrics([x["storm_event"] for x in scores]),
    }


def _bootstrap(scores, metric_name, class_name, seed):
    if not scores:
        return {"lower": None, "median": None, "upper": None}
    rng = np.random.default_rng(seed)
    n = len(scores)
    values = []
    for _ in range(BOOTSTRAPS):
        sample = [scores[int(i)] for i in rng.integers(0, n, size=n)]
        if metric_name == "event_f1":
            value = cd._merge_event_metrics([x[f"{class_name}_event"] for x in sample]).get("f1")
        else:
            value = cd._counts([x[class_name] for x in sample]).get(metric_name)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {"lower": None, "median": None, "upper": None}
    arr = np.asarray(values)
    return {"lower": float(np.percentile(arr, 2.5)), "median": float(np.percentile(arr, 50)), "upper": float(np.percentile(arr, 97.5))}


def _passes(scores):
    agg = _aggregate(scores)
    checks = {
        "active_precision": (agg["active"]["precision"] or 0) >= MIN_ACTIVE_PRECISION,
        "active_recall": (agg["active"]["recall"] or 0) >= MIN_ACTIVE_RECALL,
        "active_f1": (agg["active"]["f1"] or 0) >= MIN_ACTIVE_F1,
        "storm_precision": (agg["storm"]["precision"] or 0) >= MIN_STORM_PRECISION,
        "storm_recall": (agg["storm"]["recall"] or 0) >= MIN_STORM_RECALL,
        "storm_f1": (agg["storm"]["f1"] or 0) >= MIN_STORM_F1,
        "storm_far": (agg["storm"]["far"] or 1) <= MAX_STORM_FAR,
        "storm_event_precision": (agg["storm_event"]["precision"] or 0) >= MIN_STORM_PRECISION,
        "storm_event_recall": (agg["storm_event"]["recall"] or 0) >= MIN_STORM_RECALL,
        "storm_event_f1": (agg["storm_event"]["f1"] or 0) >= MIN_STORM_F1,
    }
    f1_ci = _bootstrap(scores, "f1", "storm", SEED)
    recall_ci = _bootstrap(scores, "recall", "storm", SEED + 1)
    event_ci = _bootstrap(scores, "event_f1", "storm", SEED + 2)
    checks["storm_f1_ci_lower"] = (f1_ci["lower"] or 0) >= MIN_STORM_F1_CI
    checks["storm_recall_ci_lower"] = (recall_ci["lower"] or 0) >= MIN_STORM_RECALL_CI
    checks["storm_event_f1_ci_lower"] = (event_ci["lower"] or 0) >= MIN_STORM_EVENT_F1_CI
    return {"passed": all(checks.values()), "score": agg, "checks": checks, "storm_f1_ci": f1_ci, "storm_recall_ci": recall_ci, "storm_event_f1_ci": event_ci}


def _load(observatories, cases):
    loaded, failures = [], []
    for obs in observatories:
        for case in cases:
            try:
                loaded.append(pg.load_case(obs, case))
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
    return loaded, failures


def _quality_filter(rows):
    accepted, rejected = [], []
    for row in rows:
        completeness = float(row.get("completeness", 0))
        reference_coverage = float(row.get("reference_coverage", 0))
        if completeness >= MIN_COMPLETENESS and reference_coverage >= MIN_REFERENCE_COVERAGE:
            accepted.append(row)
        else:
            rejected.append({"observatory": row["observatory"], "case": row["case"], "error": f"quality below release minimum: completeness={completeness:.3f}, reference_coverage={reference_coverage:.3f}"})
    return accepted, rejected


def main():
    ap = argparse.ArgumentParser(description="Strict causal detector production release gate v5.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--profile-path", default=str(Path(__file__).resolve().with_name("detector_profile.json")))
    args = ap.parse_args()
    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted({int(x) for x in args.years.split(",") if x.strip()})
    if len(years) < 3:
        raise SystemExit("Release gate requires at least three chronological years.")
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Release gate requires --cases-per-class-per-year >= 10.")

    profile_path = Path(args.profile_path).resolve()
    if not profile_path.exists():
        raise SystemExit(f"FAIL: certified profile missing: {profile_path}")
    payload = json.loads(profile_path.read_text())
    if payload.get("status") != "certified":
        raise SystemExit("FAIL: certified detector profile is not present; holdout remains untouched.")
    profile = DetectorProfile.from_dict(payload.get("profile", payload))
    profile.validate()

    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    pretest_cases = [c for c in cases if c.split != "test"]
    test_cases = [c for c in cases if c.split == "test"]
    print("=" * 100)
    print("MAGNETOMETER EXACT-PROFILE PRODUCTION RELEASE GATE v5")
    print("=" * 100)
    print(f"Profile: {profile_path}")
    print(f"Calibration years: {splits['calibration']}")
    print(f"Validation years: {splits['validation']}")
    print(f"Final-test years: {splits['test']}")
    print("Final-test access: BLOCKED until validation passes")

    pre_rows, pre_failures = _load(observatories, pretest_cases)
    validation_raw = [r for r in pre_rows if r["case"]["split"] == "validation"]
    validation_rows, validation_rejected = _quality_filter(validation_raw)
    shortages = _shortages(validation_rows, observatories, splits, "validation")
    validation_gate = _passes(_per_case_scores(validation_rows, profile)) if not shortages else {"passed": False, "score": None, "checks": {"coverage_policy": False}, "shortages": shortages}
    validation_gate["coverage_policy"] = _coverage_ok(validation_rows)
    validation_gate["passed"] = bool(validation_gate["passed"] and validation_gate["coverage_policy"])

    if not validation_gate["passed"]:
        report = {
            "schema_version": "6.0-strict-quality-sequenced",
            "release_status": "FAIL",
            "profile": {"path": str(profile_path), "values": asdict(profile)},
            "splits": splits,
            "validation": validation_gate,
            "final_test": {"accessed": False, "reason": "validation did not pass; holdout not fetched"},
            "data_quality": {"rejected_cases": pre_failures + validation_rejected, "accepted_validation_cases": len(validation_rows)},
            "sampling_policy": {"target_cases_per_station_year": args.cases_per_class_per_year, "minimum_station_cases": MIN_STATION_CASES, "minimum_pooled_cases_per_year": MIN_POOLED_CASES_PER_YEAR, "minimum_cases_per_class_per_split": MIN_CASES_PER_CLASS_PER_SPLIT, "minimum_reference_coverage": MIN_REFERENCE_COVERAGE, "minimum_completeness": MIN_COMPLETENESS},
        }
        out = Path(__file__).resolve().with_name("data") / "magnetometer_exact_profile_release_gate.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print("Validation gate: FAIL — final-test holdout untouched")
        raise SystemExit(2)

    test_raw, test_failures = _load(observatories, test_cases)
    test_rows, test_rejected = _quality_filter(test_raw)
    shortages = _shortages(test_rows, observatories, splits, "test")
    final_gate = _passes(_per_case_scores(test_rows, profile)) if not shortages else {"passed": False, "score": None, "checks": {"coverage_policy": False}, "shortages": shortages}
    final_gate["coverage_policy"] = _coverage_ok(test_rows)
    final_gate["passed"] = bool(final_gate["passed"] and final_gate["coverage_policy"])
    report = {
        "schema_version": "6.0-strict-quality-sequenced",
        "release_status": "PASS" if final_gate["passed"] else "FAIL",
        "profile": {"path": str(profile_path), "values": asdict(profile)},
        "splits": splits,
        "validation": validation_gate,
        "final_test": {"accessed": True, **final_gate},
        "data_quality": {"rejected_cases": pre_failures + validation_rejected + test_failures + test_rejected, "accepted_validation_cases": len(validation_rows), "accepted_final_test_cases": len(test_rows)},
    }
    out = Path(__file__).resolve().with_name("data") / "magnetometer_exact_profile_release_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Final release gate: {'PASS' if final_gate['passed'] else 'FAIL'}")
    print(f"Report: {out}")
    raise SystemExit(0 if final_gate["passed"] else 2)


if __name__ == "__main__":
    main()
