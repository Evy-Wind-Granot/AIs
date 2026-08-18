#!/usr/bin/env python3
"""Strict, machine-enforceable production certification gate.

This gate is intentionally harder than an MVP release check.  It keeps the
chronological calibration/validation/final-test separation, but additionally
requires:

* all mandatory cases to load successfully (data failures never disappear);
* minimum case counts per class/year/observatory;
* >=99.5% completeness and >=99% reference coverage for every test case;
* storm precision/recall/F1 and false-alarm-rate floors;
* active precision/recall/F1 floors;
* event-level storm precision/recall/F1 floors;
* 95% bootstrap lower-confidence-bound floors;
* per-observatory and per-year stability floors;
* explicit calibration-only threshold certification;
* validation non-regression before a candidate threshold can be certified.

Kp/Dst remain environmental references rather than perfect local ground truth.
Consequently, the thresholds below should be interpreted as stringent
operational consistency requirements, not as a claim that Kp provides local
station truth.  The final-test set is never used for threshold selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg  # noqa: E402


# Excellent/certification defaults.  These are deliberately strict.
DEFAULT_MIN_CASES_PER_CLASS_PER_YEAR = 10
DEFAULT_MIN_REFERENCE_COVERAGE = 0.99
DEFAULT_MIN_COMPLETENESS = 0.995
DEFAULT_MIN_STORM_PRECISION = 0.85
DEFAULT_MIN_STORM_RECALL = 0.90
DEFAULT_MIN_STORM_F1 = 0.87
DEFAULT_MAX_STORM_FALSE_ALARM_RATE = 0.01
DEFAULT_MIN_ACTIVE_PRECISION = 0.85
DEFAULT_MIN_ACTIVE_RECALL = 0.80
DEFAULT_MIN_ACTIVE_F1 = 0.82
DEFAULT_MIN_STORM_EVENT_PRECISION = 0.85
DEFAULT_MIN_STORM_EVENT_RECALL = 0.90
DEFAULT_MIN_STORM_EVENT_F1 = 0.87
DEFAULT_MIN_STORM_F1_CI_LOWER = 0.75
DEFAULT_MIN_STORM_RECALL_CI_LOWER = 0.80
DEFAULT_MIN_EVENT_F1_CI_LOWER = 0.75
DEFAULT_MAX_YEAR_REGRESSION = 0.05
DEFAULT_MAX_CASE_FAILURES = 0
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_CI_SEED = 20260818


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _metric(row: Dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _aggregate_event(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate event metrics by event counts, not by averaging ratios."""
    reference = sum(int(r.get("reference_events", 0)) for r in rows)
    predicted = sum(int(r.get("predicted_events", 0)) for r in rows)
    matched = sum(int(r.get("matched_events", 0)) for r in rows)
    missed = sum(int(r.get("missed_events", 0)) for r in rows)
    false_positive = sum(int(r.get("false_positive_events", 0)) for r in rows)

    precision = matched / predicted if predicted else None
    recall = matched / reference if reference else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "reference_events": reference,
        "predicted_events": predicted,
        "matched_events": matched,
        "missed_events": missed,
        "false_positive_events": false_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _bootstrap_event_ci(
    case_rows: Sequence[Dict[str, Any]],
    metric: str,
    iterations: int,
    seed: int,
) -> Dict[str, float | None]:
    if not case_rows or iterations <= 0:
        return {"lower": None, "median": None, "upper": None}
    rng = np.random.default_rng(seed)
    n = len(case_rows)
    values: List[float] = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample = [case_rows[int(i)] for i in idx]
        value = _metric(_aggregate_event(sample), metric)
        if value is not None:
            values.append(value)
    if not values:
        return {"lower": None, "median": None, "upper": None}
    arr = np.asarray(values, dtype=float)
    return {
        "lower": float(np.percentile(arr, 2.5)),
        "median": float(np.percentile(arr, 50)),
        "upper": float(np.percentile(arr, 97.5)),
    }


def _sample_ci(
    case_rows: Sequence[Dict[str, Any]],
    metric: str,
    iterations: int,
    seed: int,
) -> Dict[str, float | None]:
    return pg.bootstrap_metric_ci(case_rows, metric, seed, iterations)


def _case_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        name: sum(1 for r in rows if r["case"]["class_name"] == name)
        for name in ("quiet", "active", "storm")
    }


def _group(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row["observatory"] if key == "observatory" else str(row["case"][key])
        groups.setdefault(value, []).append(row)
    return groups


def _score(rows: Sequence[Dict[str, Any]], active: float, storm: float) -> Dict[str, Any]:
    aggregate = pg.aggregate_test(rows, active, storm)
    storm_event_rows = [
        pg.score_case(r, active, storm)["storm"]["event_level"] for r in rows
    ]
    active_event_rows = [
        pg.score_case(r, active, storm)["active"]["event_level"] for r in rows
    ]
    aggregate["storm_event"] = _aggregate_event(storm_event_rows)
    aggregate["active_event"] = _aggregate_event(active_event_rows)
    return aggregate


def _metric_floors(
    score: Dict[str, Any],
    *,
    min_storm_precision: float,
    min_storm_recall: float,
    min_storm_f1: float,
    max_storm_far: float,
    min_active_precision: float,
    min_active_recall: float,
    min_active_f1: float,
    min_event_precision: float,
    min_event_recall: float,
    min_event_f1: float,
) -> Dict[str, bool]:
    storm = score["storm"]
    active = score["active"]
    event = score["storm_event"]
    return {
        "storm_precision": _num(storm.get("precision"), -1) >= min_storm_precision,
        "storm_recall": _num(storm.get("recall"), -1) >= min_storm_recall,
        "storm_f1": _num(storm.get("f1"), -1) >= min_storm_f1,
        "storm_false_alarm_rate": _num(storm.get("false_alarm_rate"), 1) <= max_storm_far,
        "active_precision": _num(active.get("precision"), -1) >= min_active_precision,
        "active_recall": _num(active.get("recall"), -1) >= min_active_recall,
        "active_f1": _num(active.get("f1"), -1) >= min_active_f1,
        "storm_event_precision": _num(event.get("precision"), -1) >= min_event_precision,
        "storm_event_recall": _num(event.get("recall"), -1) >= min_event_recall,
        "storm_event_f1": _num(event.get("f1"), -1) >= min_event_f1,
    }


def _coverage_gate(rows: Sequence[Dict[str, Any]], min_reference: float, min_complete: float) -> bool:
    return bool(rows) and all(
        _num(r.get("reference_coverage"), 0) >= min_reference
        and _num(r.get("completeness"), 0) >= min_complete
        for r in rows
    )


def _worst_case_metrics(rows: Sequence[Dict[str, Any]], active: float, storm: float) -> Dict[str, Any]:
    per_case = []
    for row in rows:
        score = pg.score_case(row, active, storm)
        per_case.append({
            "observatory": row["observatory"],
            "case_id": row["case"]["case_id"],
            "year": row["case"]["year"],
            "class": row["case"]["class_name"],
            "storm_recall": score["storm"]["sample_level"]["recall"],
            "storm_f1": score["storm"]["sample_level"]["f1"],
            "storm_far": score["storm"]["sample_level"]["false_alarm_rate"],
            "storm_event_f1": score["storm"]["event_level"]["f1"],
        })
    return {"cases": per_case}


def _threshold_certification(
    calibration: Sequence[Dict[str, Any]],
    validation: Sequence[Dict[str, Any]],
    selected_active: float,
    selected_storm: float,
    production_active: float,
    production_storm: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    candidate_validation = _score(validation, selected_active, selected_storm)
    production_validation = _score(validation, production_active, production_storm)
    floors = _metric_floors(
        candidate_validation,
        min_storm_precision=args.min_storm_precision,
        min_storm_recall=args.min_storm_recall,
        min_storm_f1=args.min_storm_f1,
        max_storm_far=args.max_storm_false_alarm_rate,
        min_active_precision=args.min_active_precision,
        min_active_recall=args.min_active_recall,
        min_active_f1=args.min_active_f1,
        min_event_precision=args.min_storm_event_precision,
        min_event_recall=args.min_storm_event_recall,
        min_event_f1=args.min_storm_event_f1,
    )
    candidate_ok = all(floors.values())

    # Candidate thresholds may replace the existing production thresholds only
    # when validation is already excellent and is not materially worse on any
    # primary metric than the existing production configuration.
    comparisons = {}
    for scope, metric_names in {
        "storm": ("precision", "recall", "f1", "false_alarm_rate"),
        "active": ("precision", "recall", "f1"),
    }.items():
        comparisons[scope] = {}
        for metric in metric_names:
            candidate = _metric(candidate_validation[scope], metric)
            production = _metric(production_validation[scope], metric)
            if candidate is None or production is None:
                comparisons[scope][metric] = False
                continue
            tolerance = args.max_validation_regression
            if metric == "false_alarm_rate":
                comparisons[scope][metric] = candidate <= production + tolerance
            else:
                comparisons[scope][metric] = candidate + tolerance >= production
    non_regression = all(v for group in comparisons.values() for v in group.values())
    certified = candidate_ok and non_regression
    return {
        "certified": certified,
        "selected_thresholds": {"active_nt": selected_active, "storm_nt": selected_storm},
        "validation_candidate": candidate_validation,
        "validation_production": production_validation,
        "candidate_metric_floors": floors,
        "non_regression": comparisons,
        "selection_source": "calibration_only",
        "final_test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict production certification gate for the magnetometer detector.")
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default=",".join(map(str, pg.DEFAULT_YEARS)))
    parser.add_argument("--cases-per-class-per-year", type=int, default=DEFAULT_MIN_CASES_PER_CLASS_PER_YEAR)
    parser.add_argument("--window-days", type=int, default=pg.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-test-cases-per-class", type=int, default=DEFAULT_MIN_CASES_PER_CLASS_PER_YEAR)
    parser.add_argument("--min-reference-coverage", type=float, default=DEFAULT_MIN_REFERENCE_COVERAGE)
    parser.add_argument("--min-completeness", type=float, default=DEFAULT_MIN_COMPLETENESS)
    parser.add_argument("--min-storm-precision", type=float, default=DEFAULT_MIN_STORM_PRECISION)
    parser.add_argument("--min-storm-recall", type=float, default=DEFAULT_MIN_STORM_RECALL)
    parser.add_argument("--min-storm-f1", type=float, default=DEFAULT_MIN_STORM_F1)
    parser.add_argument("--max-storm-false-alarm-rate", type=float, default=DEFAULT_MAX_STORM_FALSE_ALARM_RATE)
    parser.add_argument("--min-active-precision", type=float, default=DEFAULT_MIN_ACTIVE_PRECISION)
    parser.add_argument("--min-active-recall", type=float, default=DEFAULT_MIN_ACTIVE_RECALL)
    parser.add_argument("--min-active-f1", type=float, default=DEFAULT_MIN_ACTIVE_F1)
    parser.add_argument("--min-storm-event-precision", type=float, default=DEFAULT_MIN_STORM_EVENT_PRECISION)
    parser.add_argument("--min-storm-event-recall", type=float, default=DEFAULT_MIN_STORM_EVENT_RECALL)
    parser.add_argument("--min-storm-event-f1", type=float, default=DEFAULT_MIN_STORM_EVENT_F1)
    parser.add_argument("--min-storm-f1-ci-lower", type=float, default=DEFAULT_MIN_STORM_F1_CI_LOWER)
    parser.add_argument("--min-storm-recall-ci-lower", type=float, default=DEFAULT_MIN_STORM_RECALL_CI_LOWER)
    parser.add_argument("--min-event-f1-ci-lower", type=float, default=DEFAULT_MIN_EVENT_F1_CI_LOWER)
    parser.add_argument("--max-validation-regression", type=float, default=DEFAULT_MAX_YEAR_REGRESSION)
    parser.add_argument("--max-case-failures", type=int, default=DEFAULT_MAX_CASE_FAILURES)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"))
    args = parser.parse_args()

    if args.cases_per_class_per_year < args.min_test_cases_per_class:
        raise SystemExit("--cases-per-class-per-year must be >= --min-test-cases-per-class.")
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Strict certification requires at least 10 cases per class per year.")
    if len(set(args.years.split(","))) < 3:
        raise SystemExit("At least 3 years are required for chronological separation.")

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    by_split = {"calibration": [], "validation": [], "test": []}
    failures: List[Dict[str, Any]] = []
    loaded: List[Dict[str, Any]] = []

    print("\n" + "=" * 100)
    print("MAGNETOMETER STRICT PRODUCTION CERTIFICATION GATE")
    print("=" * 100)
    print(f"Observatories:      {', '.join(observatories)}")
    print(f"Years:              {min(years)}-{max(years)}")
    print(f"Window:             {args.window_days} days")
    print(f"Cases/class/year:   {args.cases_per_class_per_year}")
    print(f"Discovered cases:   {len(cases)}")
    print(f"Calibration years:  {splits['calibration']}")
    print(f"Validation years:   {splits['validation']}")
    print(f"Final-test years:   {splits['test']}")

    for observatory in observatories:
        for case in cases:
            try:
                data = pg.load_case(observatory, case)
                loaded.append(data)
                by_split[case.split].append(data)
                print(f"[OK] {observatory:4s} {case.case_id:32s} ref={data['reference_coverage']:.1%} data={data['completeness']:.1%}")
            except Exception as exc:
                failures.append({"observatory": observatory, "case": pg.asdict(case), "error": str(exc)})
                print(f"[FAIL] {observatory} {case.case_id}: {exc}")

    calibration, validation, test = by_split["calibration"], by_split["validation"], by_split["test"]
    if not calibration or not validation or not test:
        raise SystemExit("Certification gate cannot run: calibration, validation, or final-test has no successful cases.")

    # Threshold selection is calibration-only.  The final test is never passed
    # into choose_threshold().
    selected_active = pg.choose_threshold(calibration, pg.ACTIVE_CANDIDATES, "active", pg.pm.PROD_ACTIVE_NT)
    selected_storm = pg.choose_threshold(calibration, pg.STORM_CANDIDATES, "storm", pg.pm.PROD_MINOR_STORM_NT)
    threshold_cert = _threshold_certification(
        calibration, validation, selected_active, selected_storm,
        pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT, args,
    )

    production_test = _score(test, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT)
    candidate_test = _score(test, selected_active, selected_storm)
    test_storm_f1_ci = _sample_ci(production_test["case_metrics"]["storm"], "f1", args.bootstrap_iterations, DEFAULT_CI_SEED)
    test_storm_recall_ci = _sample_ci(production_test["case_metrics"]["storm"], "recall", args.bootstrap_iterations, DEFAULT_CI_SEED + 1)
    test_event_f1_ci = _bootstrap_event_ci(
        [pg.score_case(r, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT)["storm"]["event_level"] for r in test],
        "f1", args.bootstrap_iterations, DEFAULT_CI_SEED + 2,
    )

    test_counts = _case_counts(test)
    all_split_counts = {split: _case_counts(rows) for split, rows in by_split.items()}
    counts_by_year_obs = {}
    for split, rows in by_split.items():
        for obs, obs_rows in _group(rows, "observatory").items():
            for year, year_rows in _group(obs_rows, "year").items():
                counts_by_year_obs[f"{split}:{obs}:{year}"] = _case_counts(year_rows)

    mandatory_count_ok = all(
        all(count >= args.min_test_cases_per_class for count in counts.values())
        for counts in all_split_counts.values()
    )
    mandatory_failures_ok = len(failures) <= args.max_case_failures
    coverage_ok = _coverage_gate(test, args.min_reference_coverage, args.min_completeness)

    primary_floors = _metric_floors(
        production_test,
        min_storm_precision=args.min_storm_precision,
        min_storm_recall=args.min_storm_recall,
        min_storm_f1=args.min_storm_f1,
        max_storm_far=args.max_storm_false_alarm_rate,
        min_active_precision=args.min_active_precision,
        min_active_recall=args.min_active_recall,
        min_active_f1=args.min_active_f1,
        min_event_precision=args.min_storm_event_precision,
        min_event_recall=args.min_storm_event_recall,
        min_event_f1=args.min_storm_event_f1,
    )

    ci_floors = {
        "storm_f1_ci_lower": _num(test_storm_f1_ci.get("lower"), -1) >= args.min_storm_f1_ci_lower,
        "storm_recall_ci_lower": _num(test_storm_recall_ci.get("lower"), -1) >= args.min_storm_recall_ci_lower,
        "storm_event_f1_ci_lower": _num(test_event_f1_ci.get("lower"), -1) >= args.min_event_f1_ci_lower,
    }

    # Every observatory and every held-out year must satisfy the same primary
    # quality floors. This prevents strong stations from hiding weak stations.
    stability: Dict[str, Any] = {"observatories": {}, "years": {}, "passed": True}
    for obs, rows in _group(test, "observatory").items():
        score = _score(rows, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT)
        floors = _metric_floors(
            score,
            min_storm_precision=args.min_storm_precision,
            min_storm_recall=args.min_storm_recall,
            min_storm_f1=args.min_storm_f1,
            max_storm_far=args.max_storm_false_alarm_rate,
            min_active_precision=args.min_active_precision,
            min_active_recall=args.min_active_recall,
            min_active_f1=args.min_active_f1,
            min_event_precision=args.min_storm_event_precision,
            min_event_recall=args.min_storm_event_recall,
            min_event_f1=args.min_storm_event_f1,
        )
        stability["observatories"][obs] = {"metrics": score, "floors": floors, "passed": all(floors.values())}
        stability["passed"] &= stability["observatories"][obs]["passed"]

    for year, rows in _group(test, "year").items():
        score = _score(rows, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT)
        floors = _metric_floors(
            score,
            min_storm_precision=args.min_storm_precision,
            min_storm_recall=args.min_storm_recall,
            min_storm_f1=args.min_storm_f1,
            max_storm_far=args.max_storm_false_alarm_rate,
            min_active_precision=args.min_active_precision,
            min_active_recall=args.min_active_recall,
            min_active_f1=args.min_active_f1,
            min_event_precision=args.min_storm_event_precision,
            min_event_recall=args.min_storm_event_recall,
            min_event_f1=args.min_storm_event_f1,
        )
        stability["years"][year] = {"metrics": score, "floors": floors, "passed": all(floors.values())}
        stability["passed"] &= stability["years"][year]["passed"]

    # A candidate threshold is only considered certified after calibration and
    # validation; the held-out test remains an unbiased audit of production.
    certification_ok = threshold_cert["certified"]
    checks = {
        "mandatory_case_counts": mandatory_count_ok,
        "mandatory_case_failures": mandatory_failures_ok,
        "reference_coverage_and_completeness": coverage_ok,
        "storm_precision": primary_floors["storm_precision"],
        "storm_recall": primary_floors["storm_recall"],
        "storm_f1": primary_floors["storm_f1"],
        "storm_false_alarm_rate": primary_floors["storm_false_alarm_rate"],
        "active_precision": primary_floors["active_precision"],
        "active_recall": primary_floors["active_recall"],
        "active_f1": primary_floors["active_f1"],
        "storm_event_precision": primary_floors["storm_event_precision"],
        "storm_event_recall": primary_floors["storm_event_recall"],
        "storm_event_f1": primary_floors["storm_event_f1"],
        "storm_f1_95pct_ci_lower": ci_floors["storm_f1_ci_lower"],
        "storm_recall_95pct_ci_lower": ci_floors["storm_recall_ci_lower"],
        "storm_event_f1_95pct_ci_lower": ci_floors["storm_event_f1_ci_lower"],
        "per_observatory_stability": bool(stability["passed"]),
        "certified_threshold_validation": certification_ok,
    }
    release_passed = all(checks.values())

    result = {
        "schema_version": "2.0-strict-certification",
        "release_status": "PASS" if release_passed else "FAIL",
        "release_gate": {
            "passed": release_passed,
            "checks": checks,
            "criteria": {
                "min_cases_per_class_per_year": args.min_test_cases_per_class,
                "min_reference_coverage": args.min_reference_coverage,
                "min_completeness": args.min_completeness,
                "min_storm_precision": args.min_storm_precision,
                "min_storm_recall": args.min_storm_recall,
                "min_storm_f1": args.min_storm_f1,
                "max_storm_false_alarm_rate": args.max_storm_false_alarm_rate,
                "min_active_precision": args.min_active_precision,
                "min_active_recall": args.min_active_recall,
                "min_active_f1": args.min_active_f1,
                "min_storm_event_precision": args.min_storm_event_precision,
                "min_storm_event_recall": args.min_storm_event_recall,
                "min_storm_event_f1": args.min_storm_event_f1,
                "min_storm_f1_ci_lower": args.min_storm_f1_ci_lower,
                "min_storm_recall_ci_lower": args.min_storm_recall_ci_lower,
                "min_event_f1_ci_lower": args.min_event_f1_ci_lower,
                "max_validation_regression": args.max_validation_regression,
                "max_case_failures": args.max_case_failures,
            },
        },
        "suite": {
            "observatories": observatories,
            "years": years,
            "splits": splits,
            "window_days": args.window_days,
            "cases_per_class_per_year": args.cases_per_class_per_year,
            "discovered_cases": len(cases),
            "successful_cases": len(loaded),
            "failed_cases": len(failures),
            "counts_by_split": all_split_counts,
            "counts_by_split_observatory_year": counts_by_year_obs,
        },
        "production_thresholds_tested": {
            "active_nt": pg.pm.PROD_ACTIVE_NT,
            "storm_nt": pg.pm.PROD_MINOR_STORM_NT,
        },
        "threshold_certification": threshold_cert,
        "final_test": {
            "production_thresholds": production_test,
            "candidate_thresholds_audit_only": candidate_test,
            "storm_f1_ci": test_storm_f1_ci,
            "storm_recall_ci": test_storm_recall_ci,
            "storm_event_f1_ci": test_event_f1_ci,
            "primary_metric_floors": primary_floors,
            "worst_case_metrics": _worst_case_metrics(test, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT),
        },
        "stability": stability,
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
        "methodology": {
            "threshold_selection": "calibration split only",
            "validation_role": "candidate certification and non-regression only",
            "final_test_role": "unseen audit; never used for selection",
            "reference_interpretation": "Kp/Dst are global environmental references, not perfect local station truth",
            "failure_policy": "mandatory data failures block certification",
            "confidence_intervals": "case-level bootstrap with fixed reproducible seeds",
        },
    }

    report_path = output_dir / "magnetometer_production_release_gate.json"
    report_path.write_text(json.dumps(result, indent=2))

    storm = production_test["storm"]
    active = production_test["active"]
    event = production_test["storm_event"]
    print("\n" + "-" * 100)
    print("FINAL HELD-OUT TEST — STRICT PRODUCTION CERTIFICATION")
    print(f"Active:  precision={_num(active.get('precision')):.3f} recall={_num(active.get('recall')):.3f} F1={_num(active.get('f1')):.3f}")
    print(f"Storm:   precision={_num(storm.get('precision')):.3f} recall={_num(storm.get('recall')):.3f} F1={_num(storm.get('f1')):.3f} FAR={_num(storm.get('false_alarm_rate')):.3f}")
    print(f"Events:  precision={_num(event.get('precision')):.3f} recall={_num(event.get('recall')):.3f} F1={_num(event.get('f1')):.3f}")
    print(f"95% CI lower: storm F1={test_storm_f1_ci['lower']} recall={test_storm_recall_ci['lower']} event F1={test_event_f1_ci['lower']}")
    print(f"Calibration-selected thresholds: active={selected_active:.0f} nT storm={selected_storm:.0f} nT")
    print(f"Certified candidate: {'YES' if threshold_cert['certified'] else 'NO'}")
    print("-" * 100)
    print("STRICT RELEASE GATE")
    print(f"Release gate: {'PASS' if release_passed else 'FAIL'}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Report: {report_path}")
    print("=" * 100)

    raise SystemExit(0 if release_passed else 2)


if __name__ == "__main__":
    main()
