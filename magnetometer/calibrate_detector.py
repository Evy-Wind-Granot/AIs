#!/usr/bin/env python3
"""Strict chronological calibration for the production magnetometer detector.

Calibration uses the same causal detector implementation as live inference.
The final-test split is never prepared or scored here. Station data-quality
failures are backfilled from a larger deterministic Kp candidate pool so each
required class/year has the requested number of usable cases.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg
from detector_core import DetectorProfile


PARAMETER_GRID = {
    "active_nt": (10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 35.0),
    "storm_nt": (30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0),
    "active_fast_ratio": (0.90, 1.00, 1.10, 1.25, 1.40),
    "active_peak_ratio": (1.25, 1.50, 1.75, 2.00, 2.25, 2.50),
    "active_peak_medium_ratio": (0.10, 0.20, 0.25, 0.35, 0.50),
    "storm_fast_ratio": (1.20, 1.40, 1.60, 1.80, 2.00),
    "storm_peak_ratio": (1.20, 1.40, 1.60, 1.80, 2.00, 2.25),
    "storm_peak_medium_ratio": (0.10, 0.20, 0.30, 0.40, 0.55),
    "storm_upper_ratio": (0.90, 1.00, 1.10, 1.20),
    "storm_release_ratio": (0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
    "peak_window_minutes": (3.0, 5.0, 7.0, 10.0),
    "active_on_minutes": (1.0, 2.0, 3.0, 5.0, 10.0),
    "active_off_minutes": (15.0, 30.0, 45.0, 60.0),
    "storm_on_minutes": (2.0, 3.0, 5.0, 10.0, 15.0),
    "storm_off_minutes": (60.0, 90.0, 120.0, 180.0),
}

MIN_PRECISION = 0.85
MIN_RECALL = 0.80
MIN_F1 = 0.82
MAX_STORM_FAR = 0.01
DEFAULT_WORKERS = 6


class PreparedCase:
    __slots__ = (
        "residual", "cadence_s", "known", "active_ref", "storm_ref",
    )

    def __init__(self, data: dict) -> None:
        residual = np.asarray(data["residual"], dtype=float)
        if residual.ndim != 1:
            raise ValueError("residual must be one-dimensional")
        self.residual = residual
        self.cadence_s = float(data["cadence_s"])
        refs = data["refs"]
        self.known = np.asarray(refs["known"], dtype=bool)
        self.active_ref = np.asarray(refs["active"], dtype=bool)
        self.storm_ref = np.asarray(refs["storm"], dtype=bool)


def _binary(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _counts(rows: Iterable[dict]) -> dict:
    tp = tn = fp = fn = 0
    for row in rows:
        tp += int(row["tp"])
        tn += int(row["tn"])
        fp += int(row["fp"])
        fn += int(row["fn"])
    return _binary_from_counts(tp, tn, fp, fn)


def _binary_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1, "far": far}


def _rolling_events(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(x & ~np.r_[False, x[:-1]])
    ends = np.flatnonzero(x & ~np.r_[x[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def _event_metrics(pred: np.ndarray, truth: np.ndarray, tolerance_samples: int = 5) -> dict:
    predicted = _rolling_events(pred)
    reference = _rolling_events(truth)
    matched_ref: set[int] = set()
    matched_pred = 0
    for ps, pe in predicted:
        for idx, (rs, re) in enumerate(reference):
            if idx in matched_ref:
                continue
            if pe + tolerance_samples >= rs and ps - tolerance_samples <= re:
                matched_ref.add(idx)
                matched_pred += 1
                break
    matched = matched_pred
    precision = matched / len(predicted) if predicted else None
    recall = matched / len(reference) if reference else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "reference_events": len(reference),
        "predicted_events": len(predicted),
        "matched_events": matched,
        "missed_events": max(0, len(reference) - matched),
        "false_positive_events": max(0, len(predicted) - matched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _merge_event_metrics(rows: Iterable[dict]) -> dict:
    ref = pred = matched = missed = false = 0
    for row in rows:
        ref += int(row["reference_events"])
        pred += int(row["predicted_events"])
        matched += int(row["matched_events"])
        missed += int(row["missed_events"])
        false += int(row["false_positive_events"])
    precision = matched / pred if pred else None
    recall = matched / ref if ref else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"reference_events": ref, "predicted_events": pred, "matched_events": matched, "missed_events": missed, "false_positive_events": false, "precision": precision, "recall": recall, "f1": f1}


def _score_case(case: PreparedCase, profile: DetectorProfile) -> tuple[dict, dict, dict, dict]:
    from detector_core import detect_activity_masks

    active, storm, major, severe, _ = detect_activity_masks(case.residual, cadence_s=case.cadence_s, profile=profile, include_anomaly=False)
    known = case.known & np.isfinite(case.residual)
    active_sample = _binary(active[known], case.active_ref[known])
    storm_sample = _binary(storm[known], case.storm_ref[known])
    active_event = _event_metrics(active[known], case.active_ref[known])
    storm_event = _event_metrics(storm[known], case.storm_ref[known])
    return active_sample, storm_sample, active_event, storm_event


def _evaluate(cases: list[PreparedCase], profile: DetectorProfile) -> dict:
    active_rows = []
    storm_rows = []
    active_events = []
    storm_events = []
    for case in cases:
        a, s, ae, se = _score_case(case, profile)
        active_rows.append(a)
        storm_rows.append(s)
        active_events.append(ae)
        storm_events.append(se)
    return {
        "active": _counts(active_rows),
        "storm": _counts(storm_rows),
        "active_event": _merge_event_metrics(active_events),
        "storm_event": _merge_event_metrics(storm_events),
    }


def _objective(score: dict) -> float:
    a = score["active"]
    s = score["storm"]
    ae = score["active_event"]
    se = score["storm_event"]
    values = (a["f1"], s["f1"], a["precision"], s["precision"], s["recall"])
    if any(v is None for v in values):
        return -1e9
    far = s["far"] if s["far"] is not None else 1.0
    if far > 0.025:
        return -1e9
    precision_floor = min(a["precision"], s["precision"])
    event_f1 = min(ae.get("f1") or 0.0, se.get("f1") or 0.0)
    # Maximize detection quality while strongly rewarding recall and event coverage.
    return float(
        0.25 * a["f1"]
        + 0.35 * s["f1"]
        + 0.15 * a["recall"]
        + 0.15 * s["recall"]
        + 0.10 * precision_floor
        + 0.05 * event_f1
        - 2.5 * far
    )


def _coordinate_descent(cases: list[PreparedCase], base: DetectorProfile) -> DetectorProfile:
    profile = base
    for name in PARAMETER_GRID:
        best = profile
        best_obj = _objective(_evaluate(cases, profile))
        for value in PARAMETER_GRID[name]:
            if name == "storm_nt" and value <= profile.active_nt:
                continue
            if name == "active_nt" and value >= profile.storm_nt:
                continue
            if name == "active_off_minutes" and value < profile.active_on_minutes:
                continue
            if name == "storm_off_minutes" and value < profile.storm_on_minutes:
                continue
            candidate = replace(profile, **{name: value})
            try:
                candidate.validate()
            except ValueError:
                continue
            obj = _objective(_evaluate(cases, candidate))
            if obj > best_obj + 1e-12:
                best = candidate
                best_obj = obj
        profile = best
    return profile


def _validation_passes(score: dict) -> bool:
    for name in ("active", "storm"):
        row = score[name]
        if (row["precision"] or 0.0) < MIN_PRECISION:
            return False
        if (row["recall"] or 0.0) < MIN_RECALL:
            return False
        if (row["f1"] or 0.0) < MIN_F1:
            return False
    return (score["storm"]["far"] or 1.0) <= MAX_STORM_FAR


def _load_one(observatory: str, case: pg.Case) -> tuple[str, pg.Case, dict]:
    return observatory, case, pg.load_case(observatory, case)


def _required_key(case: pg.Case) -> tuple[str, int, str]:
    return case.split, int(case.year), case.class_name


def _select_exact_successes(
    successful: list[tuple[str, pg.Case, dict]],
    requested: int,
) -> tuple[list[dict], dict[str, list[dict]]]:
    selected: list[dict] = []
    counts = defaultdict(int)
    failures_by_key: dict[str, list[dict]] = defaultdict(list)
    for observatory, case, data in successful:
        key = (observatory, case.split, int(case.year), case.class_name)
        if counts[key] >= requested:
            continue
        selected.append(data)
        counts[key] += 1
    missing = {str(k): requested - v for k, v in counts.items() if v < requested}
    # A key is missing entirely when no usable case was returned.
    expected_keys = {
        (obs, split, int(year), cls)
        for obs in sorted({x[0] for x in successful})
        for split, years in (("calibration", []), ("validation", []))
        for year in years
        for cls in ("quiet", "active", "storm")
    }
    return selected, failures_by_key


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronological production calibration using the exact causal detector implementation.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    args = ap.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    if len(years) < 3:
        raise SystemExit("At least three chronological years are required.")
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Production calibration requires at least 10 cases per class per year.")
    workers = max(1, min(int(args.workers), 8))

    # Oversample the candidate pool so station outages can be backfilled without
    # touching the final-test year.
    pool_size = args.cases_per_class_per_year + max(5, args.cases_per_class_per_year // 2)
    splits, cases = pg.discover_suite(years, pool_size, args.window_days)
    calibration_cases = [c for c in cases if c.split != "test"]

    kp_start = f"{min(years):04d}-01-01"
    kp_end = f"{max(years):04d}-12-31"
    master_kp = pg._fetch_kp_cached(kp_start, kp_end)
    pg._fetch_kp_cached = lambda _start, _end: master_kp

    months = set()
    for case in calibration_cases:
        start_dt = pd.Timestamp(case.start_date, tz="UTC")
        end_dt = start_dt + pd.Timedelta(days=case.days - 1)
        months.update((p.year, p.month) for p in pd.period_range(start_dt.strftime("%Y-%m"), end_dt.strftime("%Y-%m"), freq="M"))
    print(f"Prefetching Dst once for {len(months)} calibration/validation months...", flush=True)
    for year, month in sorted(months):
        pg._fetch_dst_cached(int(year), int(month))

    print(
        f"Preparing {len(calibration_cases) * len(observatories)} calibration/validation candidates "
        f"with {workers} workers; final-test cases excluded; exact usable cases required: {args.cases_per_class_per_year}.",
        flush=True,
    )

    successes: list[tuple[str, pg.Case, dict]] = []
    failures: list[dict] = []
    tasks = [(obs, case) for obs in observatories for case in calibration_cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_load_one, obs, case): (obs, case) for obs, case in tasks}
        for completed, future in enumerate(as_completed(future_map), 1):
            obs, case = future_map[future]
            try:
                _, _, data = future.result()
                successes.append((obs, case, data))
                print(f"[{completed}/{len(tasks)}] {'CACHE' if data.get('cache_hit') else 'FETCH'} {obs} {case.case_id}", flush=True)
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
                print(f"[{completed}/{len(tasks)}] FAIL {obs} {case.case_id}: {exc}", flush=True)

    # Deterministically select the requested number of usable cases per
    # observatory/split/year/class. If there are not enough, certification is
    # blocked instead of silently changing the sampling requirements.
    by_key: dict[tuple[str, str, int, str], list[tuple[pg.Case, dict]]] = defaultdict(list)
    for obs, case, data in successes:
        by_key[(obs, case.split, int(case.year), case.class_name)].append((case, data))
    for items in by_key.values():
        items.sort(key=lambda item: item[0].center_date)

    selected: list[dict] = []
    shortages: list[dict] = []
    for obs in observatories:
        for split in ("calibration", "validation"):
            for year in splits[split]:
                for class_name in ("quiet", "active", "storm"):
                    items = by_key.get((obs, split, int(year), class_name), [])
                    if len(items) < args.cases_per_class_per_year:
                        shortages.append({"observatory": obs, "split": split, "year": year, "class": class_name, "usable": len(items), "required": args.cases_per_class_per_year})
                    selected.extend(data for _, data in items[: args.cases_per_class_per_year])

    if shortages:
        report = {"status": "blocked", "reason": "insufficient usable station cases after deterministic backfill", "shortages": shortages, "failures": failures}
        candidate_path = Path(args.profile_path).resolve().with_suffix(".blocked.json")
        candidate_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    calibration = [
        data for data in selected
        if data["case"]["split"] == "calibration"
    ]
    validation = [
        data for data in selected
        if data["case"]["split"] == "validation"
    ]
    print(f"Prepared {len(calibration)} calibration and {len(validation)} validation cases after station-aware backfill.", flush=True)
    print("Searching cached causal feature space with short-peak evidence...", flush=True)

    cal_prepared = [PreparedCase(data) for data in calibration]
    val_prepared = [PreparedCase(data) for data in validation]
    profile = _coordinate_descent(cal_prepared, DetectorProfile())
    profile.validate()
    cal_score = _evaluate(cal_prepared, profile)
    val_score = _evaluate(val_prepared, profile)
    passed = _validation_passes(val_score)

    output = {
        "status": "certified" if passed else "candidate",
        "profile": asdict(profile),
        "selection": {
            "method": "calibration-only coordinate descent using exact live detector implementation",
            "calibration_years": splits["calibration"],
            "validation_years": splits["validation"],
            "final_test_years": splits["test"],
            "final_test_used": False,
            "candidate_pool_per_class_per_year": pool_size,
            "station_aware_backfill": True,
        },
        "calibration": cal_score,
        "validation": val_score,
        "validation_floors": {
            "min_precision": MIN_PRECISION,
            "min_recall": MIN_RECALL,
            "min_f1": MIN_F1,
            "max_storm_far": MAX_STORM_FAR,
        },
        "passed_validation": passed,
        "failed_source_cases": failures,
        "shortages": shortages,
        "usable_case_count": {"calibration": len(calibration), "validation": len(validation)},
    }

    path = Path(args.profile_path).resolve()
    if passed:
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"CERTIFIED detector profile written to {path}", flush=True)
    else:
        candidate = path.with_suffix(".candidate.json")
        candidate.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Validation failed; certified profile was NOT replaced. Candidate: {candidate}", flush=True)

    print(json.dumps(output, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
