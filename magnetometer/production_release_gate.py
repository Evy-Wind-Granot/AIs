#!/usr/bin/env python3
"""Stable production release gate for magnetometer validation.

This entry point reuses the production-grade validation primitives already in
production_grade_validation.py, but keeps the release/reporting layer small
and explicit. In particular, it avoids the earlier case_metrics nesting bug.

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Machine-enforceable magnetometer production release gate.")
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default=",".join(map(str, pg.DEFAULT_YEARS)))
    parser.add_argument("--cases-per-class-per-year", type=int, default=pg.DEFAULT_CASES_PER_CLASS_PER_YEAR)
    parser.add_argument("--window-days", type=int, default=pg.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-test-cases-per-class", type=int, default=2)
    parser.add_argument("--min-reference-coverage", type=float, default=pg.DEFAULT_MIN_REFERENCE_COVERAGE)
    parser.add_argument("--min-completeness", type=float, default=pg.DEFAULT_MIN_COMPLETENESS)
    parser.add_argument("--min-storm-precision", type=float, default=pg.DEFAULT_MIN_STORM_PRECISION)
    parser.add_argument("--min-storm-recall", type=float, default=pg.DEFAULT_MIN_STORM_RECALL)
    parser.add_argument("--min-storm-f1", type=float, default=pg.DEFAULT_MIN_STORM_F1)
    parser.add_argument("--max-storm-false-alarm-rate", type=float, default=pg.DEFAULT_MIN_STORM_FALSE_ALARM_RATE)
    parser.add_argument("--bootstrap-iterations", type=int, default=pg.DEFAULT_BOOTSTRAPS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"))
    args = parser.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    if len(years) < 3:
        raise SystemExit("Need at least 3 years for calibration/validation/final-test separation.")
    if args.cases_per_class_per_year < 2:
        raise SystemExit("--cases-per-class-per-year must be at least 2.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)

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
                failures.append({"observatory": observatory, "case": pg.asdict(case), "error": str(exc)})
                print(f"[FAIL] {observatory} {case.case_id}: {exc}")

    calibration = by_split["calibration"]
    validation = by_split["validation"]
    test = by_split["test"]
    if not calibration or not validation or not test:
        raise SystemExit("Release gate cannot run: a calibration, validation, or final-test split has no successful cases.")

    selected_active = pg.choose_threshold(calibration, pg.ACTIVE_CANDIDATES, "active", pg.pm.PROD_ACTIVE_NT)
    selected_storm = pg.choose_threshold(calibration, pg.STORM_CANDIDATES, "storm", pg.pm.PROD_MINOR_STORM_NT)

    validation_production = pg.aggregate_test(validation, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT)
    validation_candidate = pg.aggregate_test(validation, selected_active, selected_storm)
    test_production = pg.aggregate_test(test, pg.pm.PROD_ACTIVE_NT, pg.pm.PROD_MINOR_STORM_NT)
    test_candidate = pg.aggregate_test(test, selected_active, selected_storm)

    test_active_ci = pg.bootstrap_metric_ci(
        test_production["case_metrics"]["active"], "f1", 101, args.bootstrap_iterations
    )
    test_storm_ci = pg.bootstrap_metric_ci(
        test_production["case_metrics"]["storm"], "f1", 202, args.bootstrap_iterations
    )

    gate = pg.release_gate(
        test_production,
        args.min_test_cases_per_class,
        args.min_reference_coverage,
        args.min_completeness,
        args.min_storm_precision,
        args.min_storm_recall,
        args.min_storm_f1,
        args.max_storm_false_alarm_rate,
    )

    test_counts = {
        class_name: sum(1 for row in test if row["case"]["class_name"] == class_name)
        for class_name in ("quiet", "active", "storm")
    }
    gate["checks"]["test_cases_per_class"] = all(
        count >= args.min_test_cases_per_class for count in test_counts.values()
    )
    gate["passed"] = all(gate["checks"].values())

    result = {
        "release_status": "PASS" if gate["passed"] else "FAIL",
        "release_gate": gate,
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
        "selected_on_calibration_only": {
            "active_nt": selected_active,
            "storm_nt": selected_storm,
        },
        "validation_years": {
            "production_thresholds": validation_production,
            "calibration_selected_candidate": validation_candidate,
        },
        "final_test_years": {
            "production_thresholds": test_production,
            "calibration_selected_candidate": test_candidate,
            "confidence_intervals_95pct": {
                "active_f1": test_active_ci,
                "storm_f1": test_storm_ci,
            },
        },
        "reference_sources": {
            "primary": "GFZ Kp",
            "secondary": "Kyoto Dst when available",
            "dst_available_fraction": float(np.mean([r["dst_coverage"] for r in test])) if test else 0.0,
            "note": "Kp is a coarse global reference, not local-station ground truth."
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
    print("NOTE: thresholds are NOT changed automatically.")
    print("-" * 88)
    print(f"Release gate:              {'PASS' if gate['passed'] else 'FAIL'}")
    for name, passed in gate["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Report: {report_path}")
    print("=" * 88)

    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
