#!/usr/bin/env python3
"""Production validation runner for the magnetometer pipeline.

This runner keeps the production detector unchanged and evaluates it across a
Kp-stratified, held-out validation suite. It uses the existing
performance_metrics.py implementation for QDC and scoring primitives.

Recommended:
    python magnetometer/production_validation.py --observatory VIC

Multiple observatories:
    python magnetometer/production_validation.py --observatory VIC,BOU

More cases:
    python magnetometer/production_validation.py --observatory VIC \
        --years 2023,2024,2025 --cases-per-class 3 --window-days 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    start_date: str
    days: int
    class_name: str


ACTIVE_CANDIDATES = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
STORM_CANDIDATES = [35.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0]


def discover_cases(years: Sequence[int], window_days: int, per_class: int) -> List[Case]:
    kp = fetch_kp_gfz(f"{min(years):04d}-01-01", f"{max(years):04d}-12-31")
    if kp.empty:
        raise RuntimeError("Kp discovery returned no data.")

    pools = {"quiet": [], "active": [], "storm": []}
    for ts, value in kp.items():
        value = float(value)
        if value <= 2:
            pools["quiet"].append(ts)
        elif value < 6:
            pools["active"].append(ts)
        else:
            pools["storm"].append(ts)

    pools["quiet"].sort(key=lambda ts: float(kp.loc[ts]))
    pools["active"].sort(key=lambda ts: float(kp.loc[ts]), reverse=True)
    pools["storm"].sort(key=lambda ts: float(kp.loc[ts]), reverse=True)

    separation = pd.Timedelta(days=max(window_days + 2, 7))
    selected: List[Case] = []
    used: List[pd.Timestamp] = []

    for class_name in ("quiet", "active", "storm"):
        count = 0
        for center in pools[class_name]:
            if any(abs(center - previous) < separation for previous in used):
                continue
            start = (center - pd.Timedelta(days=window_days / 2)).normalize()
            selected.append(Case(
                case_id=f"{class_name}_{start.strftime('%Y%m%d')}",
                start_date=start.strftime("%Y-%m-%d"),
                days=window_days,
                class_name=class_name,
            ))
            used.append(center)
            count += 1
            if count >= per_class:
                break

    if len(selected) < per_class * 3:
        raise RuntimeError(f"Could only discover {len(selected)} of {per_class * 3} requested cases.")
    return sorted(selected, key=lambda c: c.start_date)


def split_cases(cases: Sequence[Case]) -> Tuple[List[Case], List[Case]]:
    calibration: List[Case] = []
    held_out: List[Case] = []
    by_class: Dict[str, List[Case]] = {"quiet": [], "active": [], "storm": []}
    for case in cases:
        by_class.setdefault(case.class_name, []).append(case)
    for class_name, class_cases in by_class.items():
        for i, case in enumerate(class_cases):
            (calibration if i % 2 == 0 else held_out).append(case)
    return sorted(calibration, key=lambda c: c.start_date), sorted(held_out, key=lambda c: c.start_date)


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
    if int(series.notna().sum()) < 24:
        raise RuntimeError("Too few valid samples.")

    index = series.index
    cadence_s = float(index.to_series().diff().dropna().dt.total_seconds().median())
    _, residual = pm.compute_qdc_baseline(series.to_numpy(dtype=float), cadence_s)

    kp = pd.Series(dtype=float)
    dst = pd.Series(dtype=float)
    try:
        kp = fetch_kp_gfz(index[0].strftime("%Y-%m-%d"), index[-1].strftime("%Y-%m-%d"))
    except Exception:
        pass

    periods = pd.period_range(index[0].strftime("%Y-%m"), index[-1].strftime("%Y-%m"), freq="M")
    dst_parts = []
    for p in periods:
        part = fetch_dst_kyoto(int(p.year), int(p.month))
        if part is not None and not part.empty:
            dst_parts.append(part)
    if dst_parts:
        dst = pd.concat(dst_parts).sort_index()

    target = pd.date_range(index[0], periods=len(index), freq=pd.Timedelta(seconds=cadence_s), tz="UTC")
    tolerance = pd.Timedelta("3h")
    kp_aligned = kp.reindex(target, method="ffill", tolerance=tolerance) if not kp.empty else pd.Series(np.nan, index=target)
    dst_aligned = dst.reindex(target, method="ffill", tolerance=tolerance) if not dst.empty else pd.Series(np.nan, index=target)
    refs = pm.reference_masks(kp_aligned, dst_aligned)

    return {
        "case": asdict(case),
        "index": index,
        "cadence_s": cadence_s,
        "series": series,
        "residual": residual,
        "refs": refs,
        "kp_coverage": float(refs["kp_known"].mean()),
        "dst_coverage": float(refs["dst_known"].mean()),
        "reference_coverage": float(refs["known"].mean()),
    }


def sample_score(data: Dict[str, Any], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    return pm.score_thresholds(
        data["residual"],
        data["refs"],
        data["cadence_s"],
        active_threshold,
        storm_threshold,
    )


def choose_threshold(calibration: Sequence[Dict[str, Any]], candidates: Sequence[float], kind: str, default: float) -> float:
    scores = []
    for threshold in candidates:
        metric_dicts = []
        for data in calibration:
            other = pm.PROD_MINOR_STORM_NT if kind == "active" else pm.PROD_ACTIVE_NT
            scoreset = sample_score(data, threshold, other)
            metric_dicts.append(scoreset[kind]["sample_level"])
        aggregate = pm.aggregate_binary_metrics(metric_dicts)
        if aggregate["f1"] is not None:
            scores.append((float(aggregate["f1"]), float(threshold)))
    if not scores:
        return default
    scores.sort(reverse=True)
    return scores[0][1]


def aggregate_cases(reports: Sequence[Dict[str, Any]], kind: str, threshold: float) -> Dict[str, Any]:
    metric_dicts = []
    for report in reports:
        data = report["_data"]
        other = pm.PROD_MINOR_STORM_NT if kind == "active" else pm.PROD_ACTIVE_NT
        metric_dicts.append(sample_score(data, threshold, other)[kind]["sample_level"])
    return pm.aggregate_binary_metrics(metric_dicts)


def compact_report(data: Dict[str, Any], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    finite = pm.finite_values(data["residual"])
    abs_resid = np.abs(finite)
    valid = int(data["series"].notna().sum())
    expected = int(round(data["case"]["days"] * 86400 / data["cadence_s"]))
    score = sample_score(data, active_threshold, storm_threshold)
    return {
        "case": data["case"],
        "samples": len(data["series"]),
        "valid_samples": valid,
        "completeness": valid / max(1, expected),
        "cadence_seconds": data["cadence_s"],
        "reference_coverage": data["reference_coverage"],
        "kp_coverage": data["kp_coverage"],
        "dst_coverage": data["dst_coverage"],
        "baseline_quality_nt": {
            "mae": float(np.mean(abs_resid)),
            "rmse": float(np.sqrt(np.mean(finite ** 2))),
            "bias": float(np.mean(finite)),
            "median_absolute_error": float(np.median(abs_resid)),
            "p95_absolute_error": float(np.percentile(abs_resid, 95)),
        },
        "scores": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out production validation for the magnetometer pipeline.")
    parser.add_argument("--observatory", default="VIC", help="One or more comma-separated INTERMAGNET observatories.")
    parser.add_argument("--years", default="2023,2024,2025")
    parser.add_argument("--cases-per-class", type=int, default=2)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"))
    args = parser.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_started = time.perf_counter()
    cases = discover_cases(years, args.window_days, args.cases_per_class)
    calibration_cases, held_out_cases = split_cases(cases)

    all_reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for observatory in observatories:
        for case in cases:
            try:
                data = load_case(observatory, case)
                data["observatory"] = observatory
                all_reports.append(data)
                case_json = compact_report(data, pm.PROD_ACTIVE_NT, pm.PROD_MINOR_STORM_NT)
                (output_dir / f"validation_{observatory}_{case.case_id}.json").write_text(json.dumps(case_json, indent=2))
                print(f"[OK] {observatory} {case.case_id}")
            except Exception as exc:
                failures.append({"observatory": observatory, "case": asdict(case), "error": str(exc)})
                print(f"[FAIL] {observatory} {case.case_id}: {exc}")

    if not all_reports:
        raise RuntimeError("No validation cases completed successfully.")

    calibration_data = [
        r for r in all_reports
        if r["case"]["case_id"] in {c.case_id for c in calibration_cases}
    ]
    held_out_data = [
        r for r in all_reports
        if r["case"]["case_id"] in {c.case_id for c in held_out_cases}
    ]
    if not calibration_data or not held_out_data:
        raise RuntimeError("Need both calibration and held-out cases.")

    selected_active = choose_threshold(calibration_data, ACTIVE_CANDIDATES, "active", pm.PROD_ACTIVE_NT)
    selected_storm = choose_threshold(calibration_data, STORM_CANDIDATES, "storm", pm.PROD_MINOR_STORM_NT)

    production_active = aggregate_cases(held_out_data, "active", pm.PROD_ACTIVE_NT)
    production_storm = aggregate_cases(held_out_data, "storm", pm.PROD_MINOR_STORM_NT)
    candidate_active = aggregate_cases(held_out_data, "active", selected_active)
    candidate_storm = aggregate_cases(held_out_data, "storm", selected_storm)

    result = {
        "suite": {
            "observatories": observatories,
            "years": years,
            "window_days": args.window_days,
            "cases_per_class": args.cases_per_class,
            "discovered_cases": [asdict(c) for c in cases],
            "calibration_cases": [asdict(c) for c in calibration_cases],
            "held_out_cases": [asdict(c) for c in held_out_cases],
            "successful_cases": len(all_reports),
            "failed_cases": len(failures),
        },
        "production_thresholds": {
            "active_nt": pm.PROD_ACTIVE_NT,
            "storm_nt": pm.PROD_MINOR_STORM_NT,
        },
        "calibration_selected_candidates": {
            "active_nt": selected_active,
            "storm_nt": selected_storm,
            "note": "Candidates are selected only from calibration cases and are not applied automatically.",
        },
        "held_out_test": {
            "production_thresholds": {
                "active": production_active,
                "storm": production_storm,
            },
            "calibration_selected_candidates": {
                "active": candidate_active,
                "storm": candidate_storm,
            },
            "coverage": {
                "reference_mean": float(np.mean([r["reference_coverage"] for r in held_out_data])),
                "kp_mean": float(np.mean([r["kp_coverage"] for r in held_out_data])),
                "dst_mean": float(np.mean([r["dst_coverage"] for r in held_out_data])),
            },
            "baseline": {
                "mean_mae_nt": float(np.mean([compact_report(r, selected_active, selected_storm)["baseline_quality_nt"]["mae"] for r in held_out_data])),
                "mean_rmse_nt": float(np.mean([compact_report(r, selected_active, selected_storm)["baseline_quality_nt"]["rmse"] for r in held_out_data])),
                "mean_p95_abs_nt": float(np.mean([compact_report(r, selected_active, selected_storm)["baseline_quality_nt"]["p95_absolute_error"] for r in held_out_data])),
            },
        },
        "failures": failures,
        "runtime_seconds": time.perf_counter() - suite_started,
    }

    path = output_dir / "magnetometer_production_validation.json"
    path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 88)
    print("MAGNETOMETER PRODUCTION VALIDATION — HELD-OUT TEST")
    print("=" * 88)
    print(f"Observatories:              {', '.join(observatories)}")
    print(f"Years:                      {min(years)}-{max(years)}")
    print(f"Cases discovered:           {len(cases)}")
    print(f"Calibration cases:          {len(calibration_cases)}")
    print(f"Held-out cases:             {len(held_out_cases)}")
    print(f"Successful cases:           {len(all_reports)}")
    print(f"Failed cases:               {len(failures)}")
    print("-" * 88)
    print("PRODUCTION THRESHOLDS — unchanged")
    print(f"  Active > {pm.PROD_ACTIVE_NT:.0f} nT")
    print(f"  Storm  > {pm.PROD_MINOR_STORM_NT:.0f} nT")
    print("-" * 88)
    print("CALIBRATION-SELECTED CANDIDATES — NOT APPLIED")
    print(f"  Active > {selected_active:.0f} nT")
    print(f"  Storm  > {selected_storm:.0f} nT")
    print("-" * 88)
    print("HELD-OUT TEST — PRODUCTION THRESHOLDS")
    print(f"  Active precision:         {production_active['precision']}")
    print(f"  Active recall:            {production_active['recall']}")
    print(f"  Active F1:                {production_active['f1']}")
    print(f"  Active false alarm:       {production_active['false_alarm_rate']}")
    print(f"  Storm precision:          {production_storm['precision']}")
    print(f"  Storm recall:             {production_storm['recall']}")
    print(f"  Storm F1:                 {production_storm['f1']}")
    print(f"  Storm false alarm:        {production_storm['false_alarm_rate']}")
    print("-" * 88)
    print("HELD-OUT TEST — CALIBRATION-SELECTED CANDIDATES")
    print(f"  Active precision:         {candidate_active['precision']}")
    print(f"  Active recall:            {candidate_active['recall']}")
    print(f"  Active F1:                {candidate_active['f1']}")
    print(f"  Storm precision:          {candidate_storm['precision']}")
    print(f"  Storm recall:             {candidate_storm['recall']}")
    print(f"  Storm F1:                 {candidate_storm['f1']}")
    print("-" * 88)
    print(f"Mean reference coverage:    {result['held_out_test']['coverage']['reference_mean'] * 100:.2f}%")
    print(f"Mean Kp coverage:           {result['held_out_test']['coverage']['kp_mean'] * 100:.2f}%")
    print(f"Mean Dst coverage:          {result['held_out_test']['coverage']['dst_mean'] * 100:.2f}%")
    print("-" * 88)
    print(f"JSON report:                {path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
