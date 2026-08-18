#!/usr/bin/env python3
"""Production-grade, held-out validation gate for the magnetometer pipeline.

This script evaluates the EXISTING production detector without changing its
inference logic or thresholds. It is deliberately conservative:

* Case discovery is based on daily maximum Kp and is separated by year.
* Chronological splits prevent future information leaking into calibration.
* Candidate thresholds are selected on calibration years only.
* Validation years are used only as a model-selection checkpoint.
* Final test years are never used for threshold selection.
* Test results are aggregated across cases and observatories.
* Case-level bootstrap confidence intervals are reported.
* Data/reference coverage is a release gate, not silently ignored.
* A failed case is recorded and excluded, never silently counted as success.

Example:
    python magnetometer/production_grade_validation.py \
        --observatory VIC,BOU \
        --years 2022,2023,2024,2025 \
        --cases-per-class-per-year 2

The script writes:
    magnetometer/data/magnetometer_production_grade_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import performance_metrics as pm  # noqa: E402
from magnetometer_demo import (  # noqa: E402
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    parse_iaga2002_to_dataframe,
)


@dataclass(frozen=True)
class Case:
    case_id: str
    center_date: str
    start_date: str
    days: int
    class_name: str
    year: int
    split: str


ACTIVE_CANDIDATES = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
STORM_CANDIDATES = (35.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0)

DEFAULT_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_CASES_PER_CLASS_PER_YEAR = 2
DEFAULT_WINDOW_DAYS = 7
DEFAULT_MIN_REFERENCE_COVERAGE = 0.95
DEFAULT_MIN_COMPLETENESS = 0.99
DEFAULT_MIN_STORM_PRECISION = 0.70
DEFAULT_MIN_STORM_RECALL = 0.60
DEFAULT_MIN_STORM_F1 = 0.60
DEFAULT_MIN_STORM_FALSE_ALARM_RATE = 0.05
DEFAULT_BOOTSTRAPS = 2000


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def aggregate_binary(metric_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return pm.aggregate_binary_metrics(metric_rows)


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    total = tp + tn + fp + fn

    def div(a: float, b: float) -> float | None:
        return float(a / b) if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {
        "samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_alarm_rate": div(fp, fp + tn),
        "miss_rate": div(fn, fn + tp),
    }


def bootstrap_metric_ci(
    case_metrics: Sequence[Dict[str, Any]],
    metric: str,
    seed: int,
    iterations: int,
) -> Dict[str, float | None]:
    """Bootstrap a case-level aggregate metric, preserving case independence."""
    if not case_metrics or iterations <= 0:
        return {"lower": None, "median": None, "upper": None}

    rng = np.random.default_rng(seed)
    n = len(case_metrics)
    values: List[float] = []

    for _ in range(iterations):
        sample = rng.integers(0, n, size=n)
        tp = tn = fp = fn = 0
        for idx in sample:
            row = case_metrics[int(idx)]
            tp += int(row.get("tp", 0))
            tn += int(row.get("tn", 0))
            fp += int(row.get("fp", 0))
            fn += int(row.get("fn", 0))
        result = metrics_from_counts(tp, tn, fp, fn).get(metric)
        if result is not None and math.isfinite(float(result)):
            values.append(float(result))

    if not values:
        return {"lower": None, "median": None, "upper": None}

    arr = np.asarray(values)
    return {
        "lower": float(np.percentile(arr, 2.5)),
        "median": float(np.percentile(arr, 50)),
        "upper": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------
def split_years(years: Sequence[int]) -> Dict[str, List[int]]:
    """Chronological 50/25/25 split: calibration / validation / final test."""
    years = sorted(set(int(y) for y in years))
    if len(years) < 3:
        raise ValueError("At least 3 distinct years are required for calibration/validation/final-test splits.")

    n = len(years)
    cal_n = max(1, n // 2)
    val_n = max(1, (n - cal_n) // 2)
    if cal_n + val_n >= n:
        val_n = 1
        cal_n = n - 2

    return {
        "calibration": years[:cal_n],
        "validation": years[cal_n : cal_n + val_n],
        "test": years[cal_n + val_n :],
    }


def discover_cases_for_year(
    kp: pd.Series,
    year: int,
    class_name: str,
    per_year: int,
    window_days: int,
    split: str,
) -> List[Case]:
    year_kp = kp[(kp.index.year == year)]
    if year_kp.empty:
        return []

    daily = year_kp.resample("1D").max().dropna()
    if class_name == "quiet":
        candidates = daily[daily <= 2.0].sort_values()
    elif class_name == "active":
        candidates = daily[(daily > 2.0) & (daily < 6.0)].sort_values(ascending=False)
    else:
        candidates = daily[daily >= 6.0].sort_values(ascending=False)

    selected: List[Case] = []
    min_separation = pd.Timedelta(days=max(window_days + 3, 14))

    for center_date, _ in candidates.items():
        if any(abs(center_date - pd.Timestamp(c.center_date, tz="UTC")) < min_separation for c in selected):
            continue

        start = (center_date - pd.Timedelta(days=window_days // 2)).normalize()
        case = Case(
            case_id=f"{split}_{class_name}_{center_date.strftime('%Y%m%d')}",
            center_date=center_date.strftime("%Y-%m-%d"),
            start_date=start.strftime("%Y-%m-%d"),
            days=window_days,
            class_name=class_name,
            year=year,
            split=split,
        )
        selected.append(case)
        if len(selected) >= per_year:
            break

    return selected


def discover_suite(years: Sequence[int], cases_per_class_per_year: int, window_days: int) -> Tuple[Dict[str, List[int]], List[Case]]:
    splits = split_years(years)
    kp = fetch_kp_gfz(f"{min(years):04d}-01-01", f"{max(years):04d}-12-31")
    if kp.empty:
        raise RuntimeError("Kp discovery returned no data.")

    cases: List[Case] = []
    for split, split_years_list in splits.items():
        for year in split_years_list:
            for class_name in ("quiet", "active", "storm"):
                cases.extend(
                    discover_cases_for_year(
                        kp,
                        year,
                        class_name,
                        cases_per_class_per_year,
                        window_days,
                        split,
                    )
                )

    return splits, sorted(cases, key=lambda c: (c.split, c.year, c.class_name, c.center_date))


# ---------------------------------------------------------------------------
# Case loading and scoring
# ---------------------------------------------------------------------------
def load_case(observatory: str, case: Case) -> Dict[str, Any]:
    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=case.start_date,
        duration_days=case.days,
        samples_per_day="Minute",
    )
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty or "f_nt" not in df.columns:
        raise RuntimeError("No usable total-field data returned.")

    series = pd.to_numeric(df["f_nt"], errors="coerce")
    valid_count = int(series.notna().sum())
    expected = max(1, int(case.days * 24 * 60))
    completeness = valid_count / expected
    if valid_count < max(24, int(expected * 0.50)):
        raise RuntimeError(f"Too few valid samples: {valid_count}/{expected} ({completeness:.1%}).")

    index = series.index
    cadence = index.to_series().diff().dropna().dt.total_seconds()
    cadence_s = float(cadence.median()) if not cadence.empty else 60.0
    if not math.isfinite(cadence_s) or cadence_s <= 0:
        raise RuntimeError("Invalid cadence in magnetometer data.")

    _, residual = pm.compute_qdc_baseline(series.to_numpy(dtype=float), cadence_s)

    kp = pd.Series(dtype=float)
    dst = pd.Series(dtype=float)
    kp_error = None
    dst_months = 0
    dst_ok = 0

    try:
        kp = fetch_kp_gfz(index[0].strftime("%Y-%m-%d"), index[-1].strftime("%Y-%m-%d"))
    except Exception as exc:
        kp_error = str(exc)

    periods = pd.period_range(index[0].strftime("%Y-%m"), index[-1].strftime("%Y-%m"), freq="M")
    dst_parts = []
    for period in periods:
        dst_months += 1
        try:
            part = fetch_dst_kyoto(int(period.year), int(period.month))
        except Exception:
            part = None
        if part is not None and not part.empty:
            dst_ok += 1
            dst_parts.append(part)
    if dst_parts:
        dst = pd.concat(dst_parts).sort_index()

    target = pd.date_range(
        index[0], periods=len(index), freq=pd.Timedelta(seconds=cadence_s), tz="UTC"
    )
    tolerance = pd.Timedelta("3h")
    kp_aligned = kp.reindex(target, method="ffill", tolerance=tolerance) if not kp.empty else pd.Series(np.nan, index=target)
    dst_aligned = dst.reindex(target, method="ffill", tolerance=tolerance) if not dst.empty else pd.Series(np.nan, index=target)
    refs = pm.reference_masks(kp_aligned, dst_aligned)

    return {
        "observatory": observatory,
        "case": asdict(case),
        "series": series,
        "residual": residual,
        "cadence_s": cadence_s,
        "completeness": completeness,
        "refs": refs,
        "kp_coverage": float(refs["kp_known"].mean()),
        "dst_coverage": float(refs["dst_known"].mean()),
        "reference_coverage": float(refs["known"].mean()),
        "kp_error": kp_error,
        "dst_months_requested": dst_months,
        "dst_months_available": dst_ok,
    }


def score_case(data: Dict[str, Any], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    return pm.score_thresholds(
        data["residual"],
        data["refs"],
        data["cadence_s"],
        active_threshold,
        storm_threshold,
    )


def choose_threshold(cases: Sequence[Dict[str, Any]], candidates: Sequence[float], kind: str, fixed_other: float) -> float:
    best: Tuple[float, float] | None = None
    for threshold in candidates:
        rows = []
        for data in cases:
            score = score_case(
                data,
                threshold if kind == "active" else fixed_other,
                threshold if kind == "storm" else fixed_other,
            )
            rows.append(score[kind]["sample_level"])
        aggregate = aggregate_binary(rows)
        if aggregate["f1"] is None:
            continue
        candidate = (float(aggregate["f1"]), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[1] if best is not None else fixed_other


def aggregate_test(cases: Sequence[Dict[str, Any]], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    active_rows = []
    storm_rows = []
    baseline = []
    coverage = []
    completeness = []

    for data in cases:
        score = score_case(data, active_threshold, storm_threshold)
        active_rows.append(score["active"]["sample_level"])
        storm_rows.append(score["storm"]["sample_level"])
        r = finite(data["residual"])
        baseline.append({
            "mae": float(np.mean(np.abs(r))),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
            "p95": float(np.percentile(np.abs(r), 95)),
        })
        coverage.append(data["reference_coverage"])
        completeness.append(data["completeness"])

    return {
        "cases": len(cases),
        "active": aggregate_binary(active_rows),
        "storm": aggregate_binary(storm_rows),
        "baseline": {
            "mean_mae_nt": float(np.mean([x["mae"] for x in baseline])) if baseline else None,
            "mean_rmse_nt": float(np.mean([x["rmse"] for x in baseline])) if baseline else None,
            "mean_p95_abs_nt": float(np.mean([x["p95"] for x in baseline])) if baseline else None,
        },
        "coverage": {
            "mean_reference": float(np.mean(coverage)) if coverage else None,
            "min_reference": float(np.min(coverage)) if coverage else None,
            "mean_completeness": float(np.mean(completeness)) if completeness else None,
            "min_completeness": float(np.min(completeness)) if completeness else None,
        },
        "case_metrics": {
            "active": active_rows,
            "storm": storm_rows,
        },
    }


# ---------------------------------------------------------------------------
# Release gates
# ---------------------------------------------------------------------------
def release_gate(
    test_result: Dict[str, Any],
    min_cases_per_class: int,
    min_reference_coverage: float,
    min_completeness: float,
    min_storm_precision: float,
    min_storm_recall: float,
    min_storm_f1: float,
    max_storm_far: float,
) -> Dict[str, Any]:
    checks = {
        "minimum_test_cases": test_result["cases"] >= min_cases_per_class * 3,
        "reference_coverage": (test_result["coverage"]["min_reference"] or 0.0) >= min_reference_coverage,
        "data_completeness": (test_result["coverage"]["min_completeness"] or 0.0) >= min_completeness,
        "storm_precision": (test_result["storm"]["precision"] or 0.0) >= min_storm_precision,
        "storm_recall": (test_result["storm"]["recall"] or 0.0) >= min_storm_recall,
        "storm_f1": (test_result["storm"]["f1"] or 0.0) >= min_storm_f1,
        "storm_false_alarm_rate": (test_result["storm"]["false_alarm_rate"] or 1.0) <= max_storm_far,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_test_cases_per_class": min_cases_per_class,
            "min_reference_coverage": min_reference_coverage,
            "min_completeness": min_completeness,
            "min_storm_precision": min_storm_precision,
            "min_storm_recall": min_storm_recall,
            "min_storm_f1": min_storm_f1,
            "max_storm_false_alarm_rate": max_storm_far,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Production-grade held-out validation gate for the magnetometer pipeline.")
    parser.add_argument("--observatory", default="VIC,BOU")
    parser.add_argument("--years", default=",".join(map(str, DEFAULT_YEARS)))
    parser.add_argument("--cases-per-class-per-year", type=int, default=DEFAULT_CASES_PER_CLASS_PER_YEAR)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-test-cases-per-class", type=int, default=2)
    parser.add_argument("--min-reference-coverage", type=float, default=DEFAULT_MIN_REFERENCE_COVERAGE)
    parser.add_argument("--min-completeness", type=float, default=DEFAULT_MIN_COMPLETENESS)
    parser.add_argument("--min-storm-precision", type=float, default=DEFAULT_MIN_STORM_PRECISION)
    parser.add_argument("--min-storm-recall", type=float, default=DEFAULT_MIN_STORM_RECALL)
    parser.add_argument("--min-storm-f1", type=float, default=DEFAULT_MIN_STORM_F1)
    parser.add_argument("--max-storm-false-alarm-rate", type=float, default=DEFAULT_MIN_STORM_FALSE_ALARM_RATE)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"))
    parser.add_argument("--keep-case-json", action="store_true")
    args = parser.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    splits, cases = discover_suite(years, args.cases_per_class_per_year, args.window_days)

    by_split = {"calibration": [], "validation": [], "test": []}
    loaded: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print("\n" + "=" * 88)
    print("MAGNETOMETER PRODUCTION-GRADE VALIDATION")
    print("=" * 88)
    print(f"Observatories: {', '.join(observatories)}")
    print(f"Years: {min(years)}-{max(years)}")
    print(f"Window: {args.window_days} days")
    print(f"Discovered cases: {len(cases)}")
    print(f"Calibration years: {splits['calibration']}")
    print(f"Validation years:  {splits['validation']}")
    print(f"Final-test years:  {splits['test']}")

    for observatory in observatories:
        for case in cases:
            try:
                data = load_case(observatory, case)
                loaded.append(data)
                by_split[case.split].append(data)
                print(f"[OK]   {observatory} {case.case_id} ref={data['reference_coverage']:.1%} data={data['completeness']:.1%}")
                if args.keep_case_json:
                    path = output_dir / f"validation_{observatory}_{case.case_id}.json"
                    compact = {
                        "observatory": observatory,
                        "case": asdict(case),
                        "completeness": data["completeness"],
                        "reference_coverage": data["reference_coverage"],
                        "kp_coverage": data["kp_coverage"],
                        "dst_coverage": data["dst_coverage"],
                        "cadence_seconds": data["cadence_s"],
                    }
                    path.write_text(json.dumps(compact, indent=2))
            except Exception as exc:
                failures.append({"observatory": observatory, "case": asdict(case), "error": str(exc)})
                print(f"[FAIL] {observatory} {case.case_id}: {exc}")

    calibration = by_split["calibration"]
    validation = by_split["validation"]
    test = by_split["test"]
    if not calibration or not validation or not test:
        raise RuntimeError("Production gate requires successful calibration, validation, and final-test cases.")

    # Threshold selection happens ONLY on calibration years.
    selected_active = choose_threshold(calibration, ACTIVE_CANDIDATES, "active", pm.PROD_ACTIVE_NT)
    selected_storm = choose_threshold(calibration, STORM_CANDIDATES, "storm", pm.PROD_MINOR_STORM_NT)

    validation_production = aggregate_test(validation, pm.PROD_ACTIVE_NT, pm.PROD_MINOR_STORM_NT)
    validation_candidate = aggregate_test(validation, selected_active, selected_storm)
    test_production = aggregate_test(test, pm.PROD_ACTIVE_NT, pm.PROD_MINOR_STORM_NT)
    test_candidate = aggregate_test(test, selected_active, selected_storm)

    # Confidence intervals are based on whole cases, not millions of correlated samples.
    test_active_ci = bootstrap_metric_ci(
        test_production["active"]["case_metrics"], "f1", 101, args.bootstrap_iterations
    )
    test_storm_ci = bootstrap_metric_ci(
        test_production["storm"]["case_metrics"], "f1", 202, args.bootstrap_iterations
    )

    gate = release_gate(
        test_production,
        args.min_test_cases_per_class,
        args.min_reference_coverage,
        args.min_completeness,
        args.min_storm_precision,
        args.min_storm_recall,
        args.min_storm_f1,
        args.max_storm_false_alarm_rate,
    )

    # Require at least min_test_cases_per_class in EACH class across the final test.
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
            "active_nt": pm.PROD_ACTIVE_NT,
            "storm_nt": pm.PROD_MINOR_STORM_NT,
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
            "note": "Kp is a coarse global reference, not local station ground truth. A passing gate is therefore operational validation, not proof of perfect local event labeling.",
        },
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
    }

    path = output_dir / "magnetometer_production_grade_validation.json"
    path.write_text(json.dumps(result, indent=2))

    print("\n" + "-" * 88)
    print("FINAL HELD-OUT TEST — PRODUCTION THRESHOLDS")
    print(f"Active precision:         {test_production['active']['precision']:.3f}")
    print(f"Active recall:            {test_production['active']['recall']:.3f}")
    print(f"Active F1:                {test_production['active']['f1']:.3f}")
    print(f"Storm precision:          {test_production['storm']['precision']:.3f}")
    print(f"Storm recall:             {test_production['storm']['recall']:.3f}")
    print(f"Storm F1:                 {test_production['storm']['f1']:.3f}")
    print(f"Storm false alarm rate:   {test_production['storm']['false_alarm_rate']:.3f}")
    print(f"Storm F1 95% CI:          {test_storm_ci['lower']!s} .. {test_storm_ci['upper']!s}")
    print("-" * 88)
    print(f"Calibration-selected active threshold: {selected_active:.0f} nT")
    print(f"Calibration-selected storm threshold:  {selected_storm:.0f} nT")
    print("NOTE: thresholds are NOT changed automatically.")
    print("-" * 88)
    print(f"Release gate:              {'PASS' if gate['passed'] else 'FAIL'}")
    for name, passed in gate["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Report: {path}")
    print("=" * 88)

    # A production gate should be machine-enforceable in CI.
    raise SystemExit(0 if gate["passed"] else 2)


if __name__ == "__main__":
    main()
