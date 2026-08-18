#!/usr/bin/env python3
"""Production release gate for the magnetometer detector.

The detector is evaluated against two different kinds of evidence:

1. Operational geomagnetic consistency: Kp (and Dst when available) is a global
   environmental reference. It is appropriate for checking whether the station
   detector catches storm conditions and avoids excessive false alarms, but it
   is NOT local station ground truth for precision/F1.
2. Data/processing quality: coverage, completeness, and test-case coverage are
   hard release requirements.

This avoids the scientifically invalid situation where a local magnetometer is
penalized because a global Kp event does not map one-to-one onto the local field.
NOAA explicitly distinguishes station K from planetary Kp and notes that
localized disturbances may differ from the global index.

Usage:
    python magnetometer/production_release_gate.py \
        --observatory VIC,BOU \
        --years 2022,2023,2024,2025 \
        --cases-per-class-per-year 2 \
        --window-days 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg  # noqa: E402


DEFAULT_MIN_STORM_RECALL = 0.50
DEFAULT_MAX_STORM_FALSE_ALARM_RATE = 0.03
DEFAULT_MIN_REFERENCE_COVERAGE = 0.95
DEFAULT_MIN_COMPLETENESS = 0.99
DEFAULT_MIN_TEST_CASES_PER_CLASS = 2


def aggregate_operational_cases(rows):
    """Aggregate the sample-level operational metrics across held-out cases."""
    if not rows:
        return {"precision": None, "recall": None, "f1": None, "false_alarm_rate": None}
    return pg.aggregate_binary(
        [row["storm"]["sample_level"] for row in rows]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Machine-enforceable production gate for local geomagnetic alerting."
    )
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default=",".join(map(str, pg.DEFAULT_YEARS)))
    parser.add_argument("--cases-per-class-per-year", type=int, default=pg.DEFAULT_CASES_PER_CLASS_PER_YEAR)
    parser.add_argument("--window-days", type=int, default=pg.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-test-cases-per-class", type=int, default=DEFAULT_MIN_TEST_CASES_PER_CLASS)
    parser.add_argument("--min-reference-coverage", type=float, default=DEFAULT_MIN_REFERENCE_COVERAGE)
    parser.add_argument("--min-completeness", type=float, default=DEFAULT_MIN_COMPLETENESS)
    parser.add_argument("--min-storm-recall", type=float, default=DEFAULT_MIN_STORM_RECALL)
    parser.add_argument("--max-storm-false-alarm-rate", type=float, default=DEFAULT_MAX_STORM_FALSE_ALARM_RATE)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"))
    args = parser.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))

    if len(years) < 3:
        raise SystemExit("Need at least 3 years for chronological calibration/validation/final-test separation.")
    if args.cases_per_class_per_year < 2:
        raise SystemExit("--cases-per-class-per-year must be at least 2.")
    if not observatories:
        raise SystemExit("At least one observatory is required.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    splits, cases = pg.discover_suite(
        years,
        args.cases_per_class_per_year,
        args.window_days,
    )

    by_split = {"calibration": [], "validation": [], "test": []}
    failures = []
    loaded = []

    print("\n" + "=" * 88)
    print("MAGNETOMETER PRODUCTION RELEASE GATE")
    print("=" * 88)
    print(f"Observatories:      {', '.join(observatories)}")
    print(f"Years:              {min(years)}-{max(years)}")
    print(f"Window:             {args.window_days} days")
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
                print(
                    f"[OK] {observatory:4s} {case.case_id:28s} "
                    f"ref={data['reference_coverage']:.1%} data={data['completeness']:.1%}"
                )
            except Exception as exc:
                failures.append({
                    "observatory": observatory,
                    "case": pg.asdict(case),
                    "error": str(exc),
                })
                print(f"[FAIL] {observatory} {case.case_id}: {exc}")

    calibration = by_split["calibration"]
    validation = by_split["validation"]
    test = by_split["test"]
    if not calibration or not validation or not test:
        raise SystemExit(
            "Release gate cannot run: a calibration, validation, or final-test split has no successful cases."
        )

    # Threshold selection is based ONLY on calibration data.
    selected_active = pg.choose_threshold(
        calibration,
        pg.ACTIVE_CANDIDATES,
        "active",
        pg.pm.PROD_ACTIVE_NT,
    )
    selected_storm = pg.choose_threshold(
        calibration,
        pg.STORM_CANDIDATES,
        "storm",
        pg.pm.PROD_MINOR_STORM_NT,
    )

    validation_production = pg.aggregate_test(
        validation,
        pg.pm.PROD_ACTIVE_NT,
        pg.pm.PROD_MINOR_STORM_NT,
    )
    validation_candidate = pg.aggregate_test(
        validation,
        selected_active,
        selected_storm,
    )
    test_production = pg.aggregate_test(
        test,
        pg.pm.PROD_ACTIVE_NT,
        pg.pm.PROD_MINOR_STORM_NT,
    )
    test_candidate = pg.aggregate_test(
        test,
        selected_active,
        selected_storm,
    )

    # The candidate is informational only. Production thresholds are not
    # changed automatically by this gate.
    test_storm_ci = pg.bootstrap_metric_ci(
        test_production["case_metrics"]["storm"],
        "f1",
        202,
        args.bootstrap_iterations,
    )

    test_counts = {
        class_name: sum(
            1
            for row in test
            if row["case"]["class_name"] == class_name
        )
        for class_name in ("quiet", "active", "storm")
    }

    min_reference = min(
        (row["reference_coverage"] for row in test),
        default=0.0,
    )
    min_completeness = min(
        (row["completeness"] for row in test),
        default=0.0,
    )

    storm = test_production["storm"]
    storm_recall = storm["recall"] or 0.0
    storm_far = storm["false_alarm_rate"] if storm["false_alarm_rate"] is not None else 1.0

    # IMPORTANT: Kp is a global reference, not local-station ground truth.
    # Therefore local precision/F1 are reported, but are NOT release blockers.
    checks = {
        "minimum_test_cases_per_class": all(
            count >= args.min_test_cases_per_class
            for count in test_counts.values()
        ),
        "reference_coverage": min_reference >= args.min_reference_coverage,
        "data_completeness": min_completeness >= args.min_completeness,
        "storm_recall": storm_recall >= args.min_storm_recall,
        "storm_false_alarm_rate": storm_far <= args.max_storm_false_alarm_rate,
    }
    release_passed = all(checks.values())

    result = {
        "release_status": "PASS" if release_passed else "FAIL",
        "release_gate": {
            "passed": release_passed,
            "checks": checks,
            "criteria": {
                "min_test_cases_per_class": args.min_test_cases_per_class,
                "min_reference_coverage": args.min_reference_coverage,
                "min_completeness": args.min_completeness,
                "min_storm_recall": args.min_storm_recall,
                "max_storm_false_alarm_rate": args.max_storm_false_alarm_rate,
                "precision_and_f1_are_non_blocking_due_to_global_reference": True,
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
            "test_case_counts_by_class": test_counts,
        },
        "production_thresholds": {
            "active_nt": pg.pm.PROD_ACTIVE_NT,
            "storm_nt": pg.pm.PROD_MINOR_STORM_NT,
        },
        "calibration_selected_thresholds": {
            "active_nt": selected_active,
            "storm_nt": selected_storm,
            "applied_to_production": False,
        },
        "validation_years": {
            "production_thresholds": validation_production,
            "calibration_selected_candidate": validation_candidate,
        },
        "final_test_years": {
            "production_thresholds": test_production,
            "calibration_selected_candidate": test_candidate,
            "storm_f1_case_bootstrap_95pct": test_storm_ci,
            "operational_gate_metrics": {
                "storm_recall": storm_recall,
                "storm_false_alarm_rate": storm_far,
            },
        },
        "reference_sources": {
            "primary": "GFZ planetary Kp",
            "secondary": "Kyoto Dst when available",
            "dst_available_fraction": float(
                np.mean([row["dst_coverage"] for row in test])
            ) if test else 0.0,
            "interpretation": (
                "Kp is a global planetary index. NOAA notes that station K and planetary Kp are different "
                "quantities and that localized disturbances can differ from the global index. Local precision/F1 "
                "against Kp are therefore diagnostic only, not release-gate ground truth."
            ),
        },
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
    }

    report_path = output_dir / "magnetometer_production_release_gate.json"
    report_path.write_text(json.dumps(result, indent=2))

    print("\n" + "-" * 88)
    print("FINAL HELD-OUT TEST — PRODUCTION THRESHOLDS")
    print(f"Active precision:         {test_production['active']['precision']:.3f}")
    print(f"Active recall:            {test_production['active']['recall']:.3f}")
    print(f"Active F1:                {test_production['active']['f1']:.3f}")
    print(f"Storm precision:          {test_production['storm']['precision']:.3f}")
    print(f"Storm recall:             {test_production['storm']['recall']:.3f}")
    print(f"Storm F1:                 {test_production['storm']['f1']:.3f}")
    print(f"Storm false alarm rate:   {test_production['storm']['false_alarm_rate']:.3f}")
    print(f"Storm F1 95% CI:          {test_storm_ci['lower']} .. {test_storm_ci['upper']}")
    print("-" * 88)
    print(f"Calibration-selected active threshold: {selected_active:.0f} nT")
    print(f"Calibration-selected storm threshold:  {selected_storm:.0f} nT")
    print("NOTE: calibration candidates are reported but are NOT applied automatically.")
    print("-" * 88)
    print("RELEASE CRITERIA")
    print(f"  Storm recall >=           {args.min_storm_recall:.3f}")
    print(f"  Storm false alarm <=      {args.max_storm_false_alarm_rate:.3f}")
    print("  Kp-based precision/F1 are diagnostic, not blocking criteria.")
    print("-" * 88)
    print(f"Release gate:              {'PASS' if release_passed else 'FAIL'}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Report: {report_path}")
    print("=" * 88)

    raise SystemExit(0 if release_passed else 2)


if __name__ == "__main__":
    main()
