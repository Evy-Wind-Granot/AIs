#!/usr/bin/env python3
"""Profile-consistent, held-out production gate for the magnetometer detector.

This gate evaluates exactly the certified DetectorProfile used by inference.
Calibration/validation are never mixed with the final test set, and the final
2025 holdout is not even scored until the certified profile passes validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg  # noqa: E402
from detector_core import DetectorProfile  # noqa: E402

MIN_CASES_PER_CLASS_PER_YEAR = 10
MIN_REFERENCE_COVERAGE = 0.99
MIN_COMPLETENESS = 0.995
MIN_STORM_PRECISION = 0.85
MIN_STORM_RECALL = 0.90
MIN_STORM_F1 = 0.87
MAX_STORM_FAR = 0.01
MIN_ACTIVE_PRECISION = 0.85
MIN_ACTIVE_RECALL = 0.80
MIN_ACTIVE_F1 = 0.82
MIN_EVENT_PRECISION = 0.85
MIN_EVENT_RECALL = 0.90
MIN_EVENT_F1 = 0.87
MIN_STORM_F1_CI = 0.75
MIN_STORM_RECALL_CI = 0.80
MIN_EVENT_F1_CI = 0.75
BOOTSTRAP_ITERATIONS = 5000
CI_SEED = 20260819


def _metric(value: Any, default: float = -1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _group(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row["observatory"] if key == "observatory" else str(row["case"][key])
        result.setdefault(value, []).append(row)
    return result


def _aggregate_event(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reference = sum(int(r.get("reference_events", 0)) for r in rows)
    predicted = sum(int(r.get("predicted_events", 0)) for r in rows)
    matched = sum(int(r.get("matched_events", 0)) for r in rows)
    missed = sum(int(r.get("missed_events", 0)) for r in rows)
    false_positive = sum(int(r.get("false_positive_events", 0)) for r in rows)
    precision = matched / predicted if predicted else None
    recall = matched / reference if reference else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": reference, "predicted_events": predicted, "matched_events": matched, "missed_events": missed, "false_positive_events": false_positive, "precision": precision, "recall": recall, "f1": f1}


def _score(rows: Sequence[Dict[str, Any]], profile: DetectorProfile) -> Dict[str, Any]:
    active = []; storm = []; storm_events = []; active_events = []
    for row in rows:
        result = pg.score_case(row, profile.active_nt, profile.storm_nt)
        active.append(result["active"]["sample_level"])
        storm.append(result["storm"]["sample_level"])
        active_events.append(result["active"]["event_level"])
        storm_events.append(result["storm"]["event_level"])
    aggregate = pg.aggregate_test(rows, profile.active_nt, profile.storm_nt) if rows else {
        "cases": 0,
        "active": pg.aggregate_binary([]), "storm": pg.aggregate_binary([]),
        "coverage": {"min_reference": None, "min_completeness": None},
        "case_metrics": {"active": [], "storm": []},
    }
    aggregate["storm_event"] = _aggregate_event(storm_events)
    aggregate["active_event"] = _aggregate_event(active_events)
    return aggregate


def _floors(score: Dict[str, Any]) -> Dict[str, bool]:
    s = score["storm"]; a = score["active"]; e = score["storm_event"]
    return {
        "storm_precision": _metric(s.get("precision")) >= MIN_STORM_PRECISION,
        "storm_recall": _metric(s.get("recall")) >= MIN_STORM_RECALL,
        "storm_f1": _metric(s.get("f1")) >= MIN_STORM_F1,
        "storm_false_alarm_rate": _metric(s.get("false_alarm_rate"), 1.0) <= MAX_STORM_FAR,
        "active_precision": _metric(a.get("precision")) >= MIN_ACTIVE_PRECISION,
        "active_recall": _metric(a.get("recall")) >= MIN_ACTIVE_RECALL,
        "active_f1": _metric(a.get("f1")) >= MIN_ACTIVE_F1,
        "storm_event_precision": _metric(e.get("precision")) >= MIN_EVENT_PRECISION,
        "storm_event_recall": _metric(e.get("recall")) >= MIN_EVENT_RECALL,
        "storm_event_f1": _metric(e.get("f1")) >= MIN_EVENT_F1,
    }


def _coverage_ok(rows: Sequence[Dict[str, Any]]) -> bool:
    return bool(rows) and all(
        _metric(row.get("reference_coverage"), 0.0) >= MIN_REFERENCE_COVERAGE
        and _metric(row.get("completeness"), 0.0) >= MIN_COMPLETENESS
        for row in rows
    )


def _counts_ok(rows: Sequence[Dict[str, Any]], minimum: int) -> bool:
    for _year, year_rows in _group(rows, "year").items():
        counts = {name: sum(1 for r in year_rows if r["case"]["class_name"] == name) for name in ("quiet", "active", "storm")}
        if any(v < minimum for v in counts.values()):
            return False
    return True


def _ci(case_metrics: Sequence[Dict[str, Any]], metric: str, seed: int) -> Dict[str, float | None]:
    return pg.bootstrap_metric_ci(case_metrics, metric, seed, BOOTSTRAP_ITERATIONS)


def _event_ci(rows: Sequence[Dict[str, Any]], profile: DetectorProfile) -> Dict[str, float | None]:
    event_rows = [pg.score_case(r, profile.active_nt, profile.storm_nt)["storm"]["event_level"] for r in rows]
    if not event_rows:
        return {"lower": None, "median": None, "upper": None}
    rng = np.random.default_rng(CI_SEED + 2)
    values = []
    n = len(event_rows)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [event_rows[int(i)] for i in rng.integers(0, n, size=n)]
        value = _aggregate_event(sample)["f1"]
        if value is not None:
            values.append(float(value))
    if not values:
        return {"lower": None, "median": None, "upper": None}
    arr = np.asarray(values, dtype=float)
    return {"lower": float(np.percentile(arr, 2.5)), "median": float(np.percentile(arr, 50)), "upper": float(np.percentile(arr, 97.5))}


def _validation_gate(score: Dict[str, Any], rows: Sequence[Dict[str, Any]], bootstrap: bool) -> Dict[str, Any]:
    floors = _floors(score)
    storm_f1_ci = _ci(score["case_metrics"]["storm"], "f1", CI_SEED) if bootstrap else {"lower": None, "median": None, "upper": None}
    storm_recall_ci = _ci(score["case_metrics"]["storm"], "recall", CI_SEED + 1) if bootstrap else {"lower": None, "median": None, "upper": None}
    event_f1_ci = _event_ci(rows, CURRENT_PROFILE) if bootstrap else {"lower": None, "median": None, "upper": None}
    checks = {
        **floors,
        "coverage_and_completeness": _coverage_ok(rows),
        "minimum_cases": _counts_ok(rows, MIN_CASES_PER_CLASS_PER_YEAR),
        "storm_f1_ci_lower": _metric(storm_f1_ci.get("lower")) >= MIN_STORM_F1_CI,
        "storm_recall_ci_lower": _metric(storm_recall_ci.get("lower")) >= MIN_STORM_RECALL_CI,
        "storm_event_f1_ci_lower": _metric(event_f1_ci.get("lower")) >= MIN_EVENT_F1_CI,
    }
    return {"passed": all(checks.values()), "checks": checks, "storm_f1_ci": storm_f1_ci, "storm_recall_ci": storm_recall_ci, "storm_event_f1_ci": event_f1_ci}


CURRENT_PROFILE: DetectorProfile


def main() -> None:
    global CURRENT_PROFILE
    parser = argparse.ArgumentParser(description="Profile-consistent held-out production gate for the magnetometer detector.")
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--cases-per-class-per-year", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    if args.cases_per_class_per_year < MIN_CASES_PER_CLASS_PER_YEAR:
        raise SystemExit(f"Strict release requires at least {MIN_CASES_PER_CLASS_PER_YEAR} cases per class per year.")

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    if len(years) < 3:
        raise SystemExit("At least three chronological years are required.")

    profile_path = Path(args.profile_path).resolve()
    if not profile_path.exists():
        raise SystemExit(f"FAIL: certified detector profile is missing: {profile_path}")
    payload = json.loads(profile_path.read_text())
    if payload.get("status") != "certified":
        raise SystemExit("FAIL: detector profile is not certified; final-test data will not be scored.")
    CURRENT_PROFILE = DetectorProfile.from_dict(payload.get("profile", payload))
    CURRENT_PROFILE.validate()
    os.environ["MAGNETOMETER_DETECTOR_PROFILE"] = str(profile_path)

    started = time.perf_counter()
    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    loaded: Dict[str, List[Dict[str, Any]]] = {"calibration": [], "validation": [], "test": []}
    failures: List[Dict[str, Any]] = []
    print("\n" + "=" * 100)
    print("MAGNETOMETER PROFILE-CONSISTENT PRODUCTION GATE")
    print("=" * 100)
    print(f"Profile:            {profile_path}")
    print(f"Observatories:      {', '.join(observatories)}")
    print(f"Calibration years:  {splits['calibration']}")
    print(f"Validation years:   {splits['validation']}")
    print(f"Final-test years:   {splits['test']}")
    print("Final-test rule:    never used until validation passes")

    for obs in observatories:
        for case in cases:
            try:
                loaded[case.split].append(pg.load_case(obs, case))
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
                print(f"[FAIL] {obs} {case.case_id}: {exc}")

    validation = loaded["validation"]
    test = loaded["test"]
    calibration = loaded["calibration"]
    validation_score = _score(validation, CURRENT_PROFILE)
    validation_gate = _validation_gate(validation_score, validation, bootstrap=not args.no_bootstrap)

    final_test_scored = bool(validation_gate["passed"])
    test_score = None
    test_checks = {"validation_gate": validation_gate["passed"]}
    test_ci = {}
    if final_test_scored:
        test_score = _score(test, CURRENT_PROFILE)
        test_floors = _floors(test_score)
        test_ci = {
            "storm_f1_ci": _ci(test_score["case_metrics"]["storm"], "f1", CI_SEED) if not args.no_bootstrap else None,
            "storm_recall_ci": _ci(test_score["case_metrics"]["storm"], "recall", CI_SEED + 1) if not args.no_bootstrap else None,
            "storm_event_f1_ci": _event_ci(test, CURRENT_PROFILE) if not args.no_bootstrap else None,
        }
        test_checks.update(test_floors)
        test_checks["coverage_and_completeness"] = _coverage_ok(test)
        test_checks["mandatory_zero_failures"] = not any(f["case"]["split"] == "test" for f in failures)
        test_checks["minimum_cases"] = _counts_ok(test, MIN_CASES_PER_CLASS_PER_YEAR)
        if not args.no_bootstrap:
            test_checks["storm_f1_ci_lower"] = _metric(test_ci["storm_f1_ci"].get("lower")) >= MIN_STORM_F1_CI
            test_checks["storm_recall_ci_lower"] = _metric(test_ci["storm_recall_ci"].get("lower")) >= MIN_STORM_RECALL_CI
            test_checks["storm_event_f1_ci_lower"] = _metric(test_ci["storm_event_f1_ci"].get("lower")) >= MIN_EVENT_F1_CI
        for obs, rows in _group(test, "observatory").items():
            test_checks[f"observatory_{obs}"] = all(_floors(_score(rows, CURRENT_PROFILE)).values())
        for year, rows in _group(test, "year").items():
            test_checks[f"year_{year}"] = all(_floors(_score(rows, CURRENT_PROFILE)).values())

    release_passed = final_test_scored and bool(test_checks) and all(test_checks.values())
    report = {
        "schema_version": "3.0-profile-consistent",
        "release_status": "PASS" if release_passed else "FAIL",
        "profile": {"path": str(profile_path), "status": payload.get("status"), "values": asdict(CURRENT_PROFILE)},
        "suite": {"observatories": observatories, "years": years, "splits": splits, "window_days": args.window_days, "cases_per_class_per_year": args.cases_per_class_per_year},
        "calibration": {"cases": len(calibration), "selected": "none_at_release_gate", "final_test_used": False},
        "validation": {"score": validation_score, "gate": validation_gate, "final_test_used": False},
        "final_test": {"scored": final_test_scored, "score": test_score, "checks": test_checks, "confidence_intervals": test_ci},
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
        "methodology": {"threshold_source": "certified DetectorProfile artifact", "selection": "none in release gate", "final_test": "unseen audit only", "data_reference": "Kp/Dst are environmental references; missing Dst is reported through reference coverage"},
    }
    out = REPO_ROOT / "magnetometer" / "data" / "magnetometer_production_release_gate_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n" + "-" * 100)
    print(f"Validation: {'PASS' if validation_gate['passed'] else 'FAIL'}")
    if test_score is not None:
        print(f"Final test: {'PASS' if all(test_checks.values()) else 'FAIL'}")
        print(f"Storm F1={test_score['storm']['f1']:.3f} recall={test_score['storm']['recall']:.3f} precision={test_score['storm']['precision']:.3f} FAR={test_score['storm']['false_alarm_rate']:.3f}")
        print(f"Active F1={test_score['active']['f1']:.3f} recall={test_score['active']['recall']:.3f} precision={test_score['active']['precision']:.3f}")
        print(f"Storm events F1={test_score['storm_event']['f1']}")
    else:
        print("Final test: NOT SCORED (validation did not pass)")
    print(f"Release gate: {'PASS' if release_passed else 'FAIL'}")
    print(f"Report: {out}")
    print("=" * 100)
    raise SystemExit(0 if release_passed else 2)


if __name__ == "__main__":
    main()
