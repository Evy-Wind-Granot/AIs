#!/usr/bin/env python3
"""Fast chronological calibration for the production magnetometer detector.

The expensive path is deliberately separated from profile search. Calibration
prepares only calibration and validation cases; final-test cases remain
untouched and reserved for the release gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
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
    "active_nt": (15.0, 20.0, 25.0, 30.0, 35.0),
    "storm_nt": (35.0, 50.0, 60.0, 70.0, 80.0),
    "active_fast_ratio": (1.10, 1.25, 1.40),
    "storm_fast_ratio": (1.60, 1.80, 2.00),
    "storm_upper_ratio": (1.00, 1.10, 1.20),
    "storm_release_ratio": (0.60, 0.65, 0.70),
}
DEFAULT_WORKERS = 6

@dataclass(frozen=True)
class PreparedCase:
    residual: np.ndarray
    cadence_s: float
    known: np.ndarray
    active_ref: np.ndarray
    storm_ref: np.ndarray
    fast_5m: np.ndarray
    medium_15m: np.ndarray
    upper_30m: np.ndarray
    slow_60m: np.ndarray
    slow_3h: np.ndarray

    @staticmethod
    def _window(seconds: float, cadence_s: float, cap: int = 0) -> int:
        n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
        return min(n, cap) if cap else n

    def predict(self, p: DetectorProfile) -> tuple[np.ndarray, np.ndarray]:
        ready = (
            np.isfinite(self.fast_5m) & np.isfinite(self.medium_15m)
            & np.isfinite(self.upper_30m) & np.isfinite(self.slow_60m)
            & np.isfinite(self.slow_3h) & np.isfinite(self.residual)
        )
        active_evidence = ready & (
            (self.medium_15m >= p.active_nt)
            | ((self.slow_60m >= p.active_slow_ratio * p.active_nt) & (self.medium_15m >= p.active_medium_slow_ratio * p.active_nt))
            | ((self.slow_3h >= p.active_slow_3h_ratio * p.active_nt) & (self.medium_15m >= p.active_medium_slow_ratio * p.active_nt))
            | ((self.upper_30m >= p.active_upper_ratio * p.active_nt) & (self.medium_15m >= p.active_medium_upper_ratio * p.active_nt))
            | ((self.fast_5m >= p.active_fast_ratio * p.active_nt) & (self.medium_15m >= p.active_medium_slow_ratio * p.active_nt))
        )
        strong_short = (self.fast_5m >= p.storm_fast_ratio * p.storm_nt) & (self.medium_15m >= p.storm_fast_medium_ratio * p.storm_nt)
        strong_30m = (self.upper_30m >= p.storm_upper_ratio * p.storm_nt) & (self.medium_15m >= p.storm_upper_medium_ratio * p.storm_nt)
        sustained = (
            (self.medium_15m >= p.storm_nt)
            | ((self.slow_60m >= p.storm_nt) & (self.medium_15m >= p.storm_medium_ratio * p.storm_nt))
            | ((self.slow_3h >= p.storm_nt) & (self.medium_15m >= p.storm_medium_ratio * p.storm_nt))
        )
        storm_evidence = ready & (sustained | strong_short | strong_30m)
        active = _hysteresis_mask(active_evidence, ready & (self.medium_15m <= 0.60 * p.active_nt), self._window(p.active_on_minutes * 60, self.cadence_s), self._window(p.active_off_minutes * 60, self.cadence_s))
        storm = _hysteresis_mask(storm_evidence, ready & (self.medium_15m <= p.storm_release_ratio * p.storm_nt), self._window(p.storm_on_minutes * 60, self.cadence_s), self._window(p.storm_off_minutes * 60, self.cadence_s))
        return active & ready, storm & ready

def _hysteresis_mask(on: np.ndarray, off: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    on = np.asarray(on, dtype=bool); off = np.asarray(off, dtype=bool)
    if on.shape != off.shape:
        raise ValueError("on/off masks must have identical shapes")
    out = np.zeros(on.size, dtype=bool)
    state = False; candidate = 0; min_on = max(1, int(min_on)); min_off = max(1, int(min_off))
    for i in range(on.size):
        if not state:
            candidate = candidate + 1 if on[i] else 0
            if candidate >= min_on:
                state = True; candidate = 0
        else:
            candidate = candidate + 1 if off[i] else 0
            if candidate >= min_off:
                state = False; candidate = 0
        out[i] = state
    return out

def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).median().to_numpy(dtype=float, copy=False)

def _rolling_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).quantile(q).to_numpy(dtype=float, copy=False)

def _prepare_case(data: dict) -> PreparedCase:
    residual = np.asarray(data["residual"], dtype=float)
    cadence_s = float(data["cadence_s"])
    if residual.ndim != 1:
        raise ValueError("residual must be one-dimensional")
    safe = np.where(np.isfinite(residual), np.abs(residual), np.nan)
    def w(seconds: float, cap: int) -> int:
        return min(max(1, int(round(seconds / max(cadence_s, 1.0)))), cap)
    refs = data["refs"]
    return PreparedCase(
        residual=residual, cadence_s=cadence_s,
        known=np.asarray(refs["known"], dtype=bool), active_ref=np.asarray(refs["active"], dtype=bool), storm_ref=np.asarray(refs["storm"], dtype=bool),
        fast_5m=_rolling_median(safe, w(5 * 60, 31)),
        medium_15m=_rolling_median(safe, w(15 * 60, 61)),
        upper_30m=_rolling_quantile(safe, w(30 * 60, 121), 0.75),
        slow_60m=_rolling_median(safe, w(60 * 60, 181)),
        slow_3h=_rolling_median(safe, w(3 * 3600, 361)),
    )

def _binary(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    pred = np.asarray(pred, dtype=bool); truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth)); tn = int(np.sum(~pred & ~truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}

def _aggregate(rows: Iterable[tuple[dict, dict]]) -> dict:
    a = np.zeros(4, dtype=np.int64); s = np.zeros(4, dtype=np.int64)
    for active, storm in rows:
        a += [active["tp"], active["tn"], active["fp"], active["fn"]]; s += [storm["tp"], storm["tn"], storm["fp"], storm["fn"]]
    return {"active": _counts(*a), "storm": _counts(*s)}

def _counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1, "far": far}

def _score_case(case: PreparedCase, profile: DetectorProfile) -> tuple[dict, dict]:
    active, storm = case.predict(profile); known = case.known & np.isfinite(case.residual)
    return _binary(active[known], case.active_ref[known]), _binary(storm[known], case.storm_ref[known])

def _evaluate(cases: list[PreparedCase], profile: DetectorProfile) -> dict:
    return _aggregate(_score_case(case, profile) for case in cases)

def _objective(score: dict) -> float:
    a, s = score["active"], score["storm"]
    if a["f1"] is None or s["f1"] is None:
        return -1e9
    return float(0.35 * a["f1"] + 0.65 * s["f1"] - 1.50 * (s["far"] or 1.0))

def _coordinate_descent(cases: list[PreparedCase], base: DetectorProfile) -> DetectorProfile:
    profile = base
    for name in ("active_nt", "storm_nt", "active_fast_ratio", "storm_fast_ratio", "storm_upper_ratio", "storm_release_ratio"):
        best, best_obj = profile, _objective(_evaluate(cases, profile))
        for value in PARAMETER_GRID[name]:
            if name == "storm_nt" and value <= profile.active_nt: continue
            if name == "active_nt" and value >= profile.storm_nt: continue
            candidate = replace(profile, **{name: value})
            try: candidate.validate()
            except ValueError: continue
            obj = _objective(_evaluate(cases, candidate))
            if obj > best_obj + 1e-12: best, best_obj = candidate, obj
        profile = best
    return profile

def _passes(score: dict, min_precision: float, min_recall: float, min_f1: float, max_far: float) -> bool:
    return all((score[name]["precision"] or 0.0) >= min_precision and (score[name]["recall"] or 0.0) >= min_recall and (score[name]["f1"] or 0.0) >= min_f1 for name in ("active", "storm")) and (score["storm"]["far"] or 1.0) <= max_far

def _load_one(observatory: str, case: pg.Case) -> tuple[str, pg.Case, dict]:
    return observatory, case, pg.load_case(observatory, case)

def main() -> None:
    ap = argparse.ArgumentParser(description="Chronologically calibrate magnetometer detector heuristics with cached causal features.")
    ap.add_argument("--observatory", default="VIC,BOU"); ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10); ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--min-precision", type=float, default=0.85); ap.add_argument("--min-recall", type=float, default=0.80); ap.add_argument("--min-f1", type=float, default=0.82); ap.add_argument("--max-storm-far", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS); ap.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    args = ap.parse_args()
    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    workers = max(1, min(int(args.workers), 8))
    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    cases = [c for c in cases if c.split != "test"]

    # Reuse the Kp series already fetched by discover_suite for every shorter
    # case range, and prefetch Dst months before worker threads start.
    kp_start = f"{min(years):04d}-01-01"; kp_end = f"{max(years):04d}-12-31"
    master_kp = pg._fetch_kp_cached(kp_start, kp_end)
    pg._fetch_kp_cached = lambda _start, _end: master_kp
    months = set()
    for case in cases:
        start_dt = pd.Timestamp(case.start_date, tz="UTC")
        end_dt = start_dt + pd.Timedelta(days=case.days - 1)
        months.update((p.year, p.month) for p in pd.period_range(start_dt.strftime("%Y-%m"), end_dt.strftime("%Y-%m"), freq="M"))
    print(f"Prefetching Dst once for {len(months)} calibration/validation months...", flush=True)
    for year, month in sorted(months):
        pg._fetch_dst_cached(int(year), int(month))

    print(f"Preparing {len(cases) * len(observatories)} calibration/validation cases with {workers} workers; final-test cases excluded.", flush=True)
    cal: list[PreparedCase] = []; val: list[PreparedCase] = []; failures = []
    tasks = [(obs, case) for obs in observatories for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_load_one, obs, case): (obs, case) for obs, case in tasks}
        for completed, future in enumerate(as_completed(future_map), 1):
            obs, case = future_map[future]
            try:
                _, _, data = future.result(); prepared = _prepare_case(data)
                if case.split == "calibration": cal.append(prepared)
                elif case.split == "validation": val.append(prepared)
                print(f"[{completed}/{len(tasks)}] {'CACHE' if data.get('cache_hit') else 'FETCH'} {obs} {case.case_id}", flush=True)
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
                print(f"[{completed}/{len(tasks)}] FAIL {obs} {case.case_id}: {exc}", flush=True)
    if not cal or not val:
        raise SystemExit("Calibration requires successful calibration and validation cases.")
    print(f"Prepared {len(cal)} calibration and {len(val)} validation cases.", flush=True)
    print("Searching cached feature space...", flush=True)
    profile = _coordinate_descent(cal, DetectorProfile()); calibration_score = _evaluate(cal, profile); validation_score = _evaluate(val, profile)
    passed = _passes(validation_score, args.min_precision, args.min_recall, args.min_f1, args.max_storm_far)
    output = {"status": "certified" if passed else "candidate", "profile": asdict(profile), "selection": {"method": "chronological coordinate descent over cached causal features", "calibration_years": splits["calibration"], "validation_years": splits["validation"], "final_test_years": splits["test"], "final_test_used": False}, "calibration_score": calibration_score, "validation_score": validation_score, "validation_floors": {"min_precision": args.min_precision, "min_recall": args.min_recall, "min_f1": args.min_f1, "max_storm_far": args.max_storm_far}, "passed_validation": passed, "failed_cases": failures}
    path = Path(args.profile_path).resolve()
    if passed:
        path.write_text(json.dumps(output, indent=2) + "\n"); print(f"CERTIFIED detector profile written to {path}", flush=True)
    else:
        candidate = path.with_suffix(".candidate.json"); candidate.write_text(json.dumps(output, indent=2) + "\n"); print(f"Validation failed; certified profile was NOT replaced. Candidate: {candidate}", flush=True)
    print(json.dumps({"status": output["status"], "profile": output["profile"], "calibration": calibration_score, "validation": validation_score}, indent=2))
    raise SystemExit(0 if passed else 2)

if __name__ == "__main__":
    main()
