#!/usr/bin/env python3
"""Production-grade, held-out validation gate for the magnetometer pipeline.

Historical cases are cached on disk after their raw/reference preparation so
calibration and validation runs do not repeatedly redownload identical windows.
Independent cases may be prepared concurrently with a bounded worker pool.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_WORKERS = 4
DEFAULT_CACHE_DIR = REPO_ROOT / "magnetometer" / "data" / "case_cache"


# In-process caches prevent duplicate Kp/Dst downloads when several cases share
# a month or year. Disk caches make subsequent calibration runs reuse prepared
# historical cases across Python processes.
_KP_CACHE: Dict[Tuple[str, str], pd.Series] = {}
_DST_CACHE: Dict[Tuple[int, int], Optional[pd.Series]] = {}


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


def bootstrap_metric_ci(case_metrics: Sequence[Dict[str, Any]], metric: str, seed: int, iterations: int) -> Dict[str, float | None]:
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
            tp += int(row.get("tp", 0)); tn += int(row.get("tn", 0)); fp += int(row.get("fp", 0)); fn += int(row.get("fn", 0))
        result = metrics_from_counts(tp, tn, fp, fn).get(metric)
        if result is not None and math.isfinite(float(result)):
            values.append(float(result))
    if not values:
        return {"lower": None, "median": None, "upper": None}
    arr = np.asarray(values)
    return {"lower": float(np.percentile(arr, 2.5)), "median": float(np.percentile(arr, 50)), "upper": float(np.percentile(arr, 97.5))}


def split_years(years: Sequence[int]) -> Dict[str, List[int]]:
    years = sorted(set(int(y) for y in years))
    if len(years) < 3:
        raise ValueError("At least 3 distinct years are required for calibration/validation/final-test splits.")
    n = len(years); cal_n = max(1, n // 2); val_n = max(1, (n - cal_n) // 2)
    if cal_n + val_n >= n:
        val_n = 1; cal_n = n - 2
    return {"calibration": years[:cal_n], "validation": years[cal_n : cal_n + val_n], "test": years[cal_n + val_n :]}


def discover_cases_for_year(kp: pd.Series, year: int, class_name: str, per_year: int, window_days: int, split: str) -> List[Case]:
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
        selected.append(Case(
            case_id=f"{split}_{class_name}_{center_date.strftime('%Y%m%d')}",
            center_date=center_date.strftime("%Y-%m-%d"), start_date=start.strftime("%Y-%m-%d"),
            days=window_days, class_name=class_name, year=year, split=split,
        ))
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
                cases.extend(discover_cases_for_year(kp, year, class_name, cases_per_class_per_year, window_days, split))
    return splits, sorted(cases, key=lambda c: (c.split, c.year, c.class_name, c.center_date))


def _cache_path(cache_dir: Path, observatory: str, case: Case) -> Path:
    safe_case = case.case_id.replace("/", "_")
    return cache_dir / f"{observatory.upper()}_{safe_case}_{case.days}d.npz"


def _save_case_cache(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "residual": np.asarray(data["residual"], dtype=float),
        "known": np.asarray(data["refs"]["known"], dtype=bool),
        "active": np.asarray(data["refs"]["active"], dtype=bool),
        "storm": np.asarray(data["refs"]["storm"], dtype=bool),
        "kp_known": np.asarray(data["refs"]["kp_known"], dtype=bool),
        "dst_known": np.asarray(data["refs"]["dst_known"], dtype=bool),
        "cadence_s": np.asarray([data["cadence_s"]], dtype=float),
        "completeness": np.asarray([data["completeness"]], dtype=float),
        "kp_coverage": np.asarray([data["kp_coverage"]], dtype=float),
        "dst_coverage": np.asarray([data["dst_coverage"]], dtype=float),
        "reference_coverage": np.asarray([data["reference_coverage"]], dtype=float),
        "series": np.asarray(data["series"].to_numpy(dtype=float), dtype=float),
    }
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def _load_case_cache(path: Path, observatory: str, case: Case) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        residual = np.asarray(z["residual"], dtype=float)
        known = np.asarray(z["known"], dtype=bool)
        active = np.asarray(z["active"], dtype=bool)
        storm = np.asarray(z["storm"], dtype=bool)
        if not (residual.size == known.size == active.size == storm.size):
            return None
        refs = {
            "known": known, "active": active, "storm": storm,
            "kp_known": np.asarray(z["kp_known"], dtype=bool),
            "dst_known": np.asarray(z["dst_known"], dtype=bool),
        }
        return {
            "observatory": observatory, "case": asdict(case), "series": pd.Series(np.asarray(z["series"], dtype=float)),
            "residual": residual, "cadence_s": float(z["cadence_s"][0]),
            "completeness": float(z["completeness"][0]), "refs": refs,
            "kp_coverage": float(z["kp_coverage"][0]), "dst_coverage": float(z["dst_coverage"][0]),
            "reference_coverage": float(z["reference_coverage"][0]), "kp_error": None,
            "dst_months_requested": None, "dst_months_available": None, "cache_hit": True,
        }
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _fetch_kp_cached(start: str, end: str) -> pd.Series:
    key = (start, end)
    if key not in _KP_CACHE:
        _KP_CACHE[key] = fetch_kp_gfz(start, end)
    return _KP_CACHE[key]


def _fetch_dst_cached(year: int, month: int) -> Optional[pd.Series]:
    key = (int(year), int(month))
    if key not in _DST_CACHE:
        _DST_CACHE[key] = fetch_dst_kyoto(int(year), int(month))
    return _DST_CACHE[key]


def load_case(observatory: str, case: Case, cache_dir: Path | None = None) -> Dict[str, Any]:
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, observatory, case)
    cached = _load_case_cache(path, observatory, case)
    if cached is not None:
        return cached

    raw = fetch_intermagnet_iaga2002(observatory=observatory, start_date=case.start_date, duration_days=case.days, samples_per_day="Minute")
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty or "f_nt" not in df.columns:
        raise RuntimeError("No usable total-field data returned.")
    series = pd.to_numeric(df["f_nt"], errors="coerce")
    valid_count = int(series.notna().sum())
    expected = max(1, int(case.days * 24 * 60)); completeness = valid_count / expected
    if valid_count < max(24, int(expected * 0.50)):
        raise RuntimeError(f"Too few valid samples: {valid_count}/{expected} ({completeness:.1%}).")
    index = series.index
    cadence = index.to_series().diff().dropna().dt.total_seconds()
    cadence_s = float(cadence.median()) if not cadence.empty else 60.0
    if not math.isfinite(cadence_s) or cadence_s <= 0:
        raise RuntimeError("Invalid cadence in magnetometer data.")
    _, residual = pm.compute_qdc_baseline(series.to_numpy(dtype=float), cadence_s)

    kp_error = None
    try:
        kp = _fetch_kp_cached(index[0].strftime("%Y-%m-%d"), index[-1].strftime("%Y-%m-%d"))
    except Exception as exc:
        kp = pd.Series(dtype=float); kp_error = str(exc)

    dst_parts = []
    dst_months = 0; dst_ok = 0
    periods = pd.period_range(index[0].strftime("%Y-%m"), index[-1].strftime("%Y-%m"), freq="M")
    for period in periods:
        dst_months += 1
        try:
            part = _fetch_dst_cached(int(period.year), int(period.month))
        except Exception:
            part = None
        if part is not None and not part.empty:
            dst_ok += 1; dst_parts.append(part)
    dst = pd.concat(dst_parts).sort_index() if dst_parts else pd.Series(dtype=float)

    target = pd.date_range(index[0], periods=len(index), freq=pd.Timedelta(seconds=cadence_s), tz="UTC")
    tolerance = pd.Timedelta("3h")
    kp_aligned = kp.reindex(target, method="ffill", tolerance=tolerance) if not kp.empty else pd.Series(np.nan, index=target)
    dst_aligned = dst.reindex(target, method="ffill", tolerance=tolerance) if not dst.empty else pd.Series(np.nan, index=target)
    refs = pm.reference_masks(kp_aligned, dst_aligned)
    data = {
        "observatory": observatory, "case": asdict(case), "series": series.reset_index(drop=True),
        "residual": residual, "cadence_s": cadence_s, "completeness": completeness, "refs": refs,
        "kp_coverage": float(refs["kp_known"].mean()), "dst_coverage": float(refs["dst_known"].mean()),
        "reference_coverage": float(refs["known"].mean()), "kp_error": kp_error,
        "dst_months_requested": dst_months, "dst_months_available": dst_ok, "cache_hit": False,
    }
    _save_case_cache(path, data)
    return data


def score_case(data: Dict[str, Any], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    return pm.score_thresholds(data["residual"], data["refs"], data["cadence_s"], active_threshold, storm_threshold)


def choose_threshold(cases: Sequence[Dict[str, Any]], candidates: Sequence[float], kind: str, fixed_other: float) -> float:
    best: Tuple[float, float] | None = None
    for threshold in candidates:
        rows = []
        for data in cases:
            score = score_case(data, threshold if kind == "active" else fixed_other, threshold if kind == "storm" else fixed_other)
            rows.append(score[kind]["sample_level"])
        aggregate = aggregate_binary(rows)
        if aggregate["f1"] is None:
            continue
        candidate = (float(aggregate["f1"]), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[1] if best is not None else fixed_other


def aggregate_test(cases: Sequence[Dict[str, Any]], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    active_rows = []; storm_rows = []; baseline = []; coverage = []; completeness = []
    for data in cases:
        score = score_case(data, active_threshold, storm_threshold)
        active_rows.append(score["active"]["sample_level"]); storm_rows.append(score["storm"]["sample_level"])
        r = finite(data["residual"])
        baseline.append({"mae": float(np.mean(np.abs(r))), "rmse": float(np.sqrt(np.mean(r ** 2))), "p95": float(np.percentile(np.abs(r), 95))})
        coverage.append(data["reference_coverage"]); completeness.append(data["completeness"])
    return {
        "cases": len(cases), "active": aggregate_binary(active_rows), "storm": aggregate_binary(storm_rows),
        "baseline": {"mean_mae_nt": float(np.mean([x["mae"] for x in baseline])) if baseline else None, "mean_rmse_nt": float(np.mean([x["rmse"] for x in baseline])) if baseline else None, "mean_p95_abs_nt": float(np.mean([x["p95"] for x in baseline])) if baseline else None},
        "coverage": {"mean_reference": float(np.mean(coverage)) if coverage else None, "min_reference": float(np.min(coverage)) if coverage else None, "mean_completeness": float(np.mean(completeness)) if completeness else None, "min_completeness": float(np.min(completeness)) if completeness else None},
        "case_metrics": {"active": active_rows, "storm": storm_rows},
    }


def release_gate(test_result: Dict[str, Any], min_cases_per_class: int, min_reference_coverage: float, min_completeness: float, min_storm_precision: float, min_storm_recall: float, min_storm_f1: float, max_storm_far: float) -> Dict[str, Any]:
    checks = {
        "minimum_test_cases": test_result["cases"] >= min_cases_per_class * 3,
        "reference_coverage": (test_result["coverage"]["min_reference"] or 0.0) >= min_reference_coverage,
        "data_completeness": (test_result["coverage"]["min_completeness"] or 0.0) >= min_completeness,
        "storm_precision": (test_result["storm"]["precision"] or 0.0) >= min_storm_precision,
        "storm_recall": (test_result["storm"]["recall"] or 0.0) >= min_storm_recall,
        "storm_f1": (test_result["storm"]["f1"] or 0.0) >= min_storm_f1,
        "storm_false_alarm_rate": (test_result["storm"]["false_alarm_rate"] or 1.0) <= max_storm_far,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _load_one(arg: tuple[str, Case, Path]) -> tuple[str, Case, Dict[str, Any]]:
    observatory, case, cache_dir = arg
    return observatory, case, load_case(observatory, case, cache_dir)


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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "magnetometer" / "data"))
    parser.add_argument("--keep-case-json", action="store_true")
    args = parser.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(args.workers), 8))
    started = time.perf_counter()

    splits, cases = discover_suite(years, args.cases_per_class_per_year, args.window_days)
    by_split = {"calibration": [], "validation": [], "test": []}; loaded: List[Dict[str, Any]] = []; failures: List[Dict[str, Any]] = []

    print("\n" + "=" * 88); print("MAGNETOMETER PRODUCTION-GRADE VALIDATION"); print("=" * 88)
    print(f"Observatories: {', '.join(observatories)}"); print(f"Years: {min(years)}-{max(years)}"); print(f"Window: {args.window_days} days")
    print(f"Discovered cases: {len(cases)}"); print(f"Calibration years: {splits['calibration']}"); print(f"Validation years:  {splits['validation']}"); print(f"Final-test years:  {splits['test']}")
    print(f"Case workers: {workers}"); print(f"Case cache: {cache_dir}")

    jobs = [(obs, case, cache_dir) for obs in observatories for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_load_one, job): job for job in jobs}
        for future in as_completed(future_map):
            observatory, case, _ = future_map[future]
            try:
                obs, completed_case, data = future.result()
                loaded.append(data); by_split[completed_case.split].append(data)
                cache_label = "CACHE" if data.get("cache_hit") else "FETCH"
                print(f"[OK {cache_label}] {obs} {completed_case.case_id} ref={data['reference_coverage']:.1%} data={data['completeness']:.1%}", flush=True)
            except Exception as exc:
                failures.append({"observatory": observatory, "case": asdict(case), "error": str(exc)})
                print(f"[FAIL] {observatory} {case.case_id}: {exc}", flush=True)

    calibration = by_split["calibration"]; validation = by_split["validation"]; test = by_split["test"]
    if not calibration or not validation or not test:
        raise RuntimeError("Production gate requires successful calibration, validation, and final-test cases.")

    selected_active = choose_threshold(calibration, ACTIVE_CANDIDATES, "active", pm.PROD_ACTIVE_NT)
    selected_storm = choose_threshold(calibration, STORM_CANDIDATES, "storm", pm.PROD_MINOR_STORM_NT)
    validation_production = aggregate_test(validation, pm.PROD_ACTIVE_NT, pm.PROD_MINOR_STORM_NT)
    validation_candidate = aggregate_test(validation, selected_active, selected_storm)
    test_production = aggregate_test(test, pm.PROD_ACTIVE_NT, pm.PROD_MINOR_STORM_NT)
    test_candidate = aggregate_test(test, selected_active, selected_storm)
    test_active_ci = bootstrap_metric_ci(test_production["active"]["case_metrics"], "f1", 101, args.bootstrap_iterations)
    test_storm_ci = bootstrap_metric_ci(test_production["storm"]["case_metrics"], "f1", 202, args.bootstrap_iterations)
    gate = release_gate(test_production, args.min_test_cases_per_class, args.min_reference_coverage, args.min_completeness, args.min_storm_precision, args.min_storm_recall, args.min_storm_f1, args.max_storm_false_alarm_rate)
    test_counts = {name: sum(1 for row in test if row["case"]["class_name"] == name) for name in ("quiet", "active", "storm")}
    gate["checks"]["test_cases_per_class"] = all(count >= args.min_test_cases_per_class for count in test_counts.values())
    gate["passed"] = all(gate["checks"].values())

    result = {
        "release_status": "PASS" if gate["passed"] else "FAIL", "release_gate": gate,
        "suite": {"observatories": observatories, "years": years, "splits": splits, "window_days": args.window_days, "cases_per_class_per_year": args.cases_per_class_per_year, "discovered_cases": len(cases), "successful_cases": len(loaded), "failed_cases": len(failures), "test_case_counts_by_class": test_counts},
        "performance": {"workers": workers, "cache_dir": str(cache_dir), "runtime_seconds": time.perf_counter() - started, "cache_hits": int(sum(bool(d.get("cache_hit")) for d in loaded)), "cache_misses": int(sum(not bool(d.get("cache_hit")) for d in loaded))},
        "production_thresholds": {"active_nt": pm.PROD_ACTIVE_NT, "storm_nt": pm.PROD_MINOR_STORM_NT},
        "selected_on_calibration_only": {"active_nt": selected_active, "storm_nt": selected_storm},
        "validation_years": {"production_thresholds": validation_production, "calibration_selected_candidate": validation_candidate},
        "final_test_years": {"production_thresholds": test_production, "calibration_selected_candidate": test_candidate, "confidence_intervals_95pct": {"active_f1": test_active_ci, "storm_f1": test_storm_ci}},
        "reference_sources": {"primary": "GFZ Kp", "secondary": "Kyoto Dst when available", "dst_available_fraction": float(np.mean([r["dst_coverage"] for r in test])) if test else 0.0, "note": "Kp is a coarse global reference, not local station ground truth."},
        "failures": failures,
    }
    path = output_dir / "magnetometer_production_grade_validation.json"; path.write_text(json.dumps(result, indent=2))
    print("\n" + "-" * 88); print("FINAL HELD-OUT TEST — PRODUCTION THRESHOLDS")
    print(f"Active precision:         {test_production['active']['precision']:.3f}"); print(f"Active recall:            {test_production['active']['recall']:.3f}"); print(f"Active F1:                {test_production['active']['f1']:.3f}")
    print(f"Storm precision:          {test_production['storm']['precision']:.3f}"); print(f"Storm recall:             {test_production['storm']['recall']:.3f}"); print(f"Storm F1:                 {test_production['storm']['f1']:.3f}"); print(f"Storm false alarm rate:   {test_production['storm']['false_alarm_rate']:.3f}")
    print(f"Calibration-selected active threshold: {selected_active:.0f} nT"); print(f"Calibration-selected storm threshold:  {selected_storm:.0f} nT")
    print(f"Release gate:              {'PASS' if gate['passed'] else 'FAIL'}"); print(f"Runtime:                   {result['performance']['runtime_seconds']:.1f}s"); print(f"Cache hits/misses:         {result['performance']['cache_hits']}/{result['performance']['cache_misses']}"); print(f"Report: {path}"); print("=" * 88)


if __name__ == "__main__":
    main()
