#!/usr/bin/env python3
"""Strict production gate with zero holdout access before validation passes."""
from __future__ import annotations

import argparse
import json
import os
import sys
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

MIN_CASES = 10
MIN_REFERENCE = 0.99
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
BOOTSTRAPS = 5000
SEED = 20260819


def metric(v: Any, default: float = -1.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def group(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row["observatory"] if key == "observatory" else str(row["case"][key])
        out.setdefault(value, []).append(row)
    return out


def event_aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ref = sum(int(r.get("reference_events", 0)) for r in rows)
    pred = sum(int(r.get("predicted_events", 0)) for r in rows)
    matched = sum(int(r.get("matched_events", 0)) for r in rows)
    missed = sum(int(r.get("missed_events", 0)) for r in rows)
    fp = sum(int(r.get("false_positive_events", 0)) for r in rows)
    precision = matched / pred if pred else None
    recall = matched / ref if ref else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": ref, "predicted_events": pred, "matched_events": matched, "missed_events": missed, "false_positive_events": fp, "precision": precision, "recall": recall, "f1": f1}


def score(rows: Sequence[Dict[str, Any]], profile: DetectorProfile) -> Dict[str, Any]:
    aggregate = pg.aggregate_test(rows, profile.active_nt, profile.storm_nt)
    storm_events = [pg.score_case(r, profile.active_nt, profile.storm_nt)["storm"]["event_level"] for r in rows]
    active_events = [pg.score_case(r, profile.active_nt, profile.storm_nt)["active"]["event_level"] for r in rows]
    aggregate["storm_event"] = event_aggregate(storm_events)
    aggregate["active_event"] = event_aggregate(active_events)
    return aggregate


def floors(s: Dict[str, Any]) -> Dict[str, bool]:
    storm, active, event = s["storm"], s["active"], s["storm_event"]
    return {
        "storm_precision": metric(storm.get("precision")) >= MIN_STORM_PRECISION,
        "storm_recall": metric(storm.get("recall")) >= MIN_STORM_RECALL,
        "storm_f1": metric(storm.get("f1")) >= MIN_STORM_F1,
        "storm_false_alarm_rate": metric(storm.get("false_alarm_rate"), 1.0) <= MAX_STORM_FAR,
        "active_precision": metric(active.get("precision")) >= MIN_ACTIVE_PRECISION,
        "active_recall": metric(active.get("recall")) >= MIN_ACTIVE_RECALL,
        "active_f1": metric(active.get("f1")) >= MIN_ACTIVE_F1,
        "storm_event_precision": metric(event.get("precision")) >= MIN_EVENT_PRECISION,
        "storm_event_recall": metric(event.get("recall")) >= MIN_EVENT_RECALL,
        "storm_event_f1": metric(event.get("f1")) >= MIN_EVENT_F1,
    }


def coverage_ok(rows: Sequence[Dict[str, Any]]) -> bool:
    return bool(rows) and all(metric(r.get("reference_coverage"), 0) >= MIN_REFERENCE and metric(r.get("completeness"), 0) >= MIN_COMPLETENESS for r in rows)


def count_ok(rows: Sequence[Dict[str, Any]], minimum: int) -> bool:
    for year_rows in group(rows, "year").values():
        counts = {c: sum(1 for r in year_rows if r["case"]["class_name"] == c) for c in ("quiet", "active", "storm")}
        if any(v < minimum for v in counts.values()):
            return False
    return True


def ci(rows: Sequence[Dict[str, Any]], name: str, seed: int) -> Dict[str, float | None]:
    return pg.bootstrap_metric_ci(rows, name, seed, BOOTSTRAPS)


def event_ci(rows: Sequence[Dict[str, Any]], profile: DetectorProfile) -> Dict[str, float | None]:
    event_rows = [pg.score_case(r, profile.active_nt, profile.storm_nt)["storm"]["event_level"] for r in rows]
    if not event_rows:
        return {"lower": None, "median": None, "upper": None}
    rng = np.random.default_rng(SEED + 2)
    vals = []
    n = len(event_rows)
    for _ in range(BOOTSTRAPS):
        sample = [event_rows[int(i)] for i in rng.integers(0, n, size=n)]
        value = event_aggregate(sample)["f1"]
        if value is not None:
            vals.append(float(value))
    if not vals:
        return {"lower": None, "median": None, "upper": None}
    arr = np.asarray(vals)
    return {"lower": float(np.percentile(arr, 2.5)), "median": float(np.percentile(arr, 50)), "upper": float(np.percentile(arr, 97.5))}


def gate(rows: Sequence[Dict[str, Any]], profile: DetectorProfile, *, require_zero_failures: bool = False, failures: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    s = score(rows, profile)
    c = floors(s)
    storm_f1_ci = ci(s["case_metrics"]["storm"], "f1", SEED)
    storm_recall_ci = ci(s["case_metrics"]["storm"], "recall", SEED + 1)
    event_f1_ci = event_ci(rows, profile)
    checks = {
        **c,
        "coverage_and_completeness": coverage_ok(rows),
        "minimum_cases_per_class_per_year": count_ok(rows, MIN_CASES),
        "storm_f1_ci_lower": metric(storm_f1_ci.get("lower")) >= MIN_STORM_F1_CI,
        "storm_recall_ci_lower": metric(storm_recall_ci.get("lower")) >= MIN_STORM_RECALL_CI,
        "storm_event_f1_ci_lower": metric(event_f1_ci.get("lower")) >= MIN_EVENT_F1_CI,
    }
    if require_zero_failures:
        checks["zero_case_failures"] = len(failures) == 0
    return {"passed": all(checks.values()), "score": s, "checks": checks, "storm_f1_ci": storm_f1_ci, "storm_recall_ci": storm_recall_ci, "storm_event_f1_ci": event_f1_ci}


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict profile-consistent detector release gate.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    args = ap.parse_args()
    if args.cases_per_class_per_year < MIN_CASES:
        raise SystemExit(f"Strict release requires at least {MIN_CASES} cases per class per year.")

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    if len(years) < 3:
        raise SystemExit("Strict release requires at least three chronological years.")

    profile_path = Path(args.profile_path).resolve()
    if not profile_path.exists():
        raise SystemExit(f"FAIL: profile missing: {profile_path}")
    payload = json.loads(profile_path.read_text())
    if payload.get("status") != "certified":
        raise SystemExit("FAIL: detector profile is not certified; final-test data was not accessed.")
    profile = DetectorProfile.from_dict(payload.get("profile", payload))
    profile.validate()
    os.environ["MAGNETOMETER_DETECTOR_PROFILE"] = str(profile_path)

    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    pretest = [c for c in cases if c.split != "test"]
    test_cases = [c for c in cases if c.split == "test"]
    loaded: Dict[str, List[Dict[str, Any]]] = {"calibration": [], "validation": [], "test": []}
    failures: List[Dict[str, Any]] = []

    print("\n" + "=" * 100)
    print("MAGNETOMETER STRICT PROFILE RELEASE GATE")
    print("=" * 100)
    print(f"Profile:            {profile_path}")
    print(f"Calibration years:  {splits['calibration']}")
    print(f"Validation years:   {splits['validation']}")
    print(f"Final-test years:   {splits['test']}")
    print("Final-test access:  BLOCKED until validation passes")

    for obs in observatories:
        for case in pretest:
            try:
                loaded[case.split].append(pg.load_case(obs, case))
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})

    validation_gate = gate(loaded["validation"], profile, require_zero_failures=False, failures=failures)
    if validation_gate["passed"]:
        print("Validation gate: PASS — opening final-test holdout")
        for obs in observatories:
            for case in test_cases:
                try:
                    loaded["test"].append(pg.load_case(obs, case))
                except Exception as exc:
                    failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
        final_gate = gate(loaded["test"], profile, require_zero_failures=True, failures=[f for f in failures if f["case"]["split"] == "test"])
    else:
        print("Validation gate: FAIL — final-test holdout remains untouched")
        final_gate = {"passed": False, "score": None, "checks": {"validation_gate": False}}

    report = {
        "schema_version": "4.0-strict-sequenced",
        "release_status": "PASS" if final_gate["passed"] else "FAIL",
        "profile": {"path": str(profile_path), "status": payload.get("status"), "values": asdict(profile)},
        "suite": {"observatories": observatories, "years": years, "splits": splits, "window_days": args.window_days, "cases_per_class_per_year": args.cases_per_class_per_year},
        "validation": validation_gate,
        "final_test": {"accessed": bool(validation_gate["passed"]), **final_gate},
        "failures": failures,
        "methodology": {"threshold_selection": "none at release gate; certified profile artifact is evaluated unchanged", "final_test": "not fetched or scored until validation passes", "references": "Kp/Dst are environmental references; coverage is gated"},
    }
    out = REPO_ROOT / "magnetometer" / "data" / "magnetometer_production_release_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Release gate: {'PASS' if final_gate['passed'] else 'FAIL'}")
    print(f"Report: {out}")
    raise SystemExit(0 if final_gate["passed"] else 2)


if __name__ == "__main__":
    main()
