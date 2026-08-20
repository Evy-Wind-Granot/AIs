#!/usr/bin/env python3
"""Strict chronological calibration for the production magnetometer detector.

Calibration uses the same causal detector implementation as live inference.
The final-test split is never prepared or scored here. Station data-quality
failures are backfilled from a larger deterministic Kp candidate pool so each
required class/year has enough usable cases.

Parameter selection is safety-constrained: candidates are compared by hard
production-floor violations first, then by worst-group performance and only
then by aggregate quality. The optimizer is regularized toward the conservative
production profile so it cannot improve a scalar objective by creating a noisy,
over-sensitive detector with unacceptable false alarms.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg
from detector_core import DetectorProfile


# Production parameter search deliberately excludes ultra-sensitive values that
# were producing pathological false-alarm rates in earlier calibration runs.
PARAMETER_GRID = {
    "active_nt": (15.0, 17.5, 20.0, 25.0, 30.0, 35.0),
    "storm_nt": (40.0, 50.0, 60.0, 70.0, 80.0),
    "active_fast_ratio": (1.00, 1.10, 1.25, 1.40),
    "active_peak_ratio": (1.50, 1.75, 2.00, 2.25, 2.50),
    "active_peak_medium_ratio": (0.20, 0.25, 0.35, 0.50),
    "storm_fast_ratio": (1.40, 1.60, 1.80, 2.00),
    "storm_peak_ratio": (1.40, 1.60, 1.80, 2.00, 2.25),
    "storm_peak_medium_ratio": (0.20, 0.30, 0.40, 0.55),
    "storm_upper_ratio": (1.00, 1.10, 1.20),
    "storm_release_ratio": (0.60, 0.65, 0.70, 0.75),
    "peak_window_minutes": (3.0, 5.0, 7.0, 10.0),
    "active_on_minutes": (2.0, 3.0, 5.0, 10.0),
    "active_off_minutes": (30.0, 45.0, 60.0),
    "storm_on_minutes": (3.0, 5.0, 10.0, 15.0),
    "storm_off_minutes": (90.0, 120.0, 180.0),
}

MIN_PRECISION = 0.85
MIN_RECALL = 0.80
MIN_F1 = 0.82
MAX_STORM_FAR = 0.01
MIN_EVENT_PRECISION = 0.85
MIN_EVENT_RECALL = 0.90
MIN_EVENT_F1 = 0.87
DEFAULT_WORKERS = 6


class PreparedCase:
    __slots__ = (
        "residual", "cadence_s", "known", "active_ref", "storm_ref",
        "fast_5m", "medium_15m", "upper_30m", "slow_60m", "slow_3h",
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
        self.fast_5m, self.medium_15m, self.upper_30m, self.slow_60m, self.slow_3h = _prepare_case_features(residual, self.cadence_s)

    @property
    def n(self) -> int:
        return int(self.residual.size)

    @staticmethod
    def _window(seconds: float, cadence_s: float, cap: int = 0) -> int:
        n = max(1, int(round(seconds / max(float(cadence_s), 1.0))))
        return min(n, cap) if cap else n

    def predict(self, profile: DetectorProfile) -> tuple[np.ndarray, np.ndarray]:
        peak = _rolling_max(np.where(np.isfinite(self.residual), np.abs(self.residual), np.nan), self._window(profile.peak_window_minutes * 60, self.cadence_s, 61))
        history_ready = (
            np.isfinite(self.fast_5m)
            & np.isfinite(self.medium_15m)
            & np.isfinite(self.upper_30m)
            & np.isfinite(self.slow_60m)
            & np.isfinite(self.slow_3h)
            & np.isfinite(peak)
            & np.isfinite(self.residual)
        )
        active_evidence = history_ready & (
            (self.medium_15m >= profile.active_nt)
            | ((self.slow_60m >= profile.active_slow_ratio * profile.active_nt) & (self.medium_15m >= profile.active_medium_slow_ratio * profile.active_nt))
            | ((self.slow_3h >= profile.active_slow_3h_ratio * profile.active_nt) & (self.medium_15m >= profile.active_medium_slow_ratio * profile.active_nt))
            | ((self.upper_30m >= profile.active_upper_ratio * profile.active_nt) & (self.medium_15m >= profile.active_medium_upper_ratio * profile.active_nt))
            | ((self.fast_5m >= profile.active_fast_ratio * profile.active_nt) & (self.medium_15m >= profile.active_medium_slow_ratio * profile.active_nt))
            | ((peak >= profile.active_peak_ratio * profile.active_nt) & (self.medium_15m >= profile.active_peak_medium_ratio * profile.active_nt))
        )
        strong_short = (self.fast_5m >= profile.storm_fast_ratio * profile.storm_nt) & (self.medium_15m >= profile.storm_fast_medium_ratio * profile.storm_nt)
        strong_peak = (peak >= profile.storm_peak_ratio * profile.storm_nt) & (self.medium_15m >= profile.storm_peak_medium_ratio * profile.storm_nt)
        strong_30m = (self.upper_30m >= profile.storm_upper_ratio * profile.storm_nt) & (self.medium_15m >= profile.storm_upper_medium_ratio * profile.storm_nt)
        sustained = (
            (self.medium_15m >= profile.storm_nt)
            | ((self.slow_60m >= profile.storm_nt) & (self.medium_15m >= profile.storm_medium_ratio * profile.storm_nt))
            | ((self.slow_3h >= profile.storm_nt) & (self.medium_15m >= profile.storm_medium_ratio * profile.storm_nt))
        )
        storm_evidence = history_ready & (sustained | strong_short | strong_peak | strong_30m)

        active = _hysteresis_mask_fast(
            active_evidence,
            history_ready & (self.medium_15m <= 0.60 * profile.active_nt),
            self._window(profile.active_on_minutes * 60, self.cadence_s),
            self._window(profile.active_off_minutes * 60, self.cadence_s),
        )
        storm = _hysteresis_mask_fast(
            storm_evidence,
            history_ready & (self.medium_15m <= profile.storm_release_ratio * profile.storm_nt),
            self._window(profile.storm_on_minutes * 60, self.cadence_s),
            self._window(profile.storm_off_minutes * 60, self.cadence_s),
        )
        active &= history_ready
        storm &= history_ready
        return active, storm


def _prepare_case(data: dict) -> PreparedCase:
    """Compatibility factory that caches profile-independent case evidence."""
    return PreparedCase(data)


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


def _hysteresis_mask_fast(on: np.ndarray, off: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    """Small linear state machine used after evidence has been cached."""
    on = np.asarray(on, dtype=bool)
    off = np.asarray(off, dtype=bool)
    if on.shape != off.shape:
        raise ValueError("on/off masks must have identical shapes")
    out = np.zeros(on.size, dtype=bool)
    state = False
    candidate = 0
    min_on = max(1, int(min_on))
    min_off = max(1, int(min_off))
    for i in range(on.size):
        if not state:
            candidate = candidate + 1 if on[i] else 0
            if candidate >= min_on:
                state = True
                candidate = 0
            out[i] = state
        else:
            out[i] = state
            candidate = candidate + 1 if off[i] else 0
            if candidate >= min_off:
                state = False
                candidate = 0
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).median().to_numpy(dtype=float, copy=False)


def _rolling_quantile(values: np.ndarray, window: int, quantile: float) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).quantile(quantile).to_numpy(dtype=float, copy=False)


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window, min_periods=window).max().to_numpy(dtype=float, copy=False)


def _prepare_case_features(residual: np.ndarray, cadence_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute profile-independent rolling evidence once per calibration case."""
    if cadence_s <= 0 or not np.isfinite(cadence_s):
        raise ValueError("cadence_s must be a positive finite number")
    magnitude = np.abs(np.asarray(residual, dtype=float))
    safe = np.where(np.isfinite(magnitude), magnitude, np.nan)

    def w(seconds: float, cap: int) -> int:
        return min(max(1, int(round(seconds / max(cadence_s, 1.0)))), cap)

    fast = _rolling_median(safe, w(5 * 60, 31))
    medium = _rolling_median(safe, w(15 * 60, 61))
    upper = _rolling_quantile(safe, w(30 * 60, 121), 0.75)
    slow = _rolling_median(safe, w(60 * 60, 181))
    slow3 = _rolling_median(safe, w(3 * 3600, 361))
    return fast, medium, upper, slow, slow3


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
    precision = matched_pred / len(predicted) if predicted else None
    recall = matched_pred / len(reference) if reference else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "reference_events": len(reference),
        "predicted_events": len(predicted),
        "matched_events": matched_pred,
        "missed_events": max(0, len(reference) - matched_pred),
        "false_positive_events": max(0, len(predicted) - matched_pred),
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
    active, storm = case.predict(profile)
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


def _production_violation(score: dict) -> float:
    """Continuous violation score; zero means every production floor passes."""
    active = score["active"]
    storm = score["storm"]
    event = score["storm_event"]
    terms = [
        max(0.0, MIN_PRECISION - float(active["precision"] or 0.0)),
        max(0.0, MIN_RECALL - float(active["recall"] or 0.0)),
        max(0.0, MIN_F1 - float(active["f1"] or 0.0)),
        max(0.0, MIN_PRECISION - float(storm["precision"] or 0.0)),
        max(0.0, MIN_RECALL - float(storm["recall"] or 0.0)),
        max(0.0, MIN_F1 - float(storm["f1"] or 0.0)),
        max(0.0, float(storm["far"] or 1.0) - MAX_STORM_FAR),
        max(0.0, MIN_EVENT_PRECISION - float(event.get("precision") or 0.0)),
        max(0.0, MIN_EVENT_RECALL - float(event.get("recall") or 0.0)),
        max(0.0, MIN_EVENT_F1 - float(event.get("f1") or 0.0)),
    ]
    return float(sum(terms))


def _profile_distance(profile: DetectorProfile, reference: DetectorProfile) -> float:
    """Safety regularizer; thresholds/timings weigh more heavily than ratios."""
    names = (
        "active_nt", "storm_nt", "active_slow_ratio", "active_slow_3h_ratio",
        "active_upper_ratio", "active_fast_ratio", "active_medium_slow_ratio",
        "active_medium_upper_ratio", "active_peak_ratio", "active_peak_medium_ratio",
        "storm_fast_ratio", "storm_fast_medium_ratio", "storm_upper_ratio",
        "storm_upper_medium_ratio", "storm_medium_ratio", "storm_release_ratio",
        "storm_peak_ratio", "storm_peak_medium_ratio", "peak_window_minutes",
        "active_on_minutes", "active_off_minutes", "storm_on_minutes", "storm_off_minutes",
    )
    scale = {
        "active_nt": 20.0, "storm_nt": 50.0, "peak_window_minutes": 5.0,
        "active_on_minutes": 5.0, "active_off_minutes": 30.0,
        "storm_on_minutes": 10.0, "storm_off_minutes": 120.0,
    }
    total = 0.0
    for name in names:
        denom = scale.get(name, 1.0)
        total += abs(float(getattr(profile, name)) - float(getattr(reference, name))) / denom
    return total


def _group_metrics(cases: Sequence[PreparedCase], profile: DetectorProfile) -> list[dict]:
    """Return per-case metrics so a single easy year/station cannot hide failures."""
    out = []
    for case in cases:
        active, storm, active_event, storm_event = _score_case(case, profile)
        out.append({"active": active, "storm": storm, "active_event": active_event, "storm_event": storm_event})
    return out


def _worst_group_penalty(cases: Sequence[PreparedCase], profile: DetectorProfile) -> float:
    if not cases:
        return 1.0
    grouped = []
    for row in _group_metrics(cases, profile):
        for name in ("active", "storm"):
            m = row[name]
            grouped.append(min(float(m["precision"] or 0.0), float(m["recall"] or 0.0), float(m["f1"] or 0.0)))
        e = row["storm_event"]
        grouped.append(min(float(e.get("precision") or 0.0), float(e.get("recall") or 0.0), float(e.get("f1") or 0.0)))
    return float(max(0.0, 0.75 - min(grouped)))


def _candidate_key(score: dict, profile: DetectorProfile, cases: Sequence[PreparedCase], reference: DetectorProfile) -> tuple[float, float, float, float]:
    """Lexicographic selection: feasibility -> robustness -> aggregate quality -> regularization."""
    violation = _production_violation(score)
    worst_penalty = _worst_group_penalty(cases, profile)
    aggregate_f1 = np.mean([
        float(score["active"]["f1"] or 0.0),
        float(score["storm"]["f1"] or 0.0),
        float(score["active_event"]["f1"] or 0.0),
        float(score["storm_event"]["f1"] or 0.0),
    ])
    distance = _profile_distance(profile, reference)
    return (round(violation, 12), round(worst_penalty, 12), -round(float(aggregate_f1), 12), round(distance, 12))


def _coordinate_descent(cases: list[PreparedCase], base: DetectorProfile) -> DetectorProfile:
    """Coordinate search with hard production constraints and conservative priors."""
    profile = base
    reference = DetectorProfile()
    best_key = _candidate_key(_evaluate(cases, profile), profile, cases, reference)
    for name in PARAMETER_GRID:
        best = profile
        best_obj = best_key
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
            score = _evaluate(cases, candidate)
            key = _candidate_key(score, candidate, cases, reference)
            if key < best_obj:
                best = candidate
                best_obj = key
        profile = best
        best_key = best_obj
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
    if (score["storm"]["far"] or 1.0) > MAX_STORM_FAR:
        return False
    event = score["storm_event"]
    return (
        (event.get("precision") or 0.0) >= MIN_EVENT_PRECISION
        and (event.get("recall") or 0.0) >= MIN_EVENT_RECALL
        and (event.get("f1") or 0.0) >= MIN_EVENT_F1
    )


def _load_one(observatory: str, case: pg.Case) -> tuple[str, pg.Case, dict]:
    return observatory, case, pg.load_case(observatory, case)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronological production detector calibration with safety-constrained parameter selection.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    args = ap.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    if len(years) < 3:
        raise SystemExit("At least three chronological years are required.")
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Production calibration requires at least 10 target cases per class per year.")
    workers = max(1, min(int(args.workers), 8))

    pool_size = max(args.cases_per_class_per_year, args.cases_per_class_per_year * 2)
    splits, cases = pg.discover_suite(years, pool_size, args.window_days)
    cases = [c for c in cases if c.split != "test"]

    master_kp = pg._fetch_kp_cached(f"{min(years):04d}-01-01", f"{max(years):04d}-12-31")
    pg._fetch_kp_cached = lambda _start, _end: master_kp

    months = set()
    for case in cases:
        start = pd.Timestamp(case.start_date, tz="UTC")
        end = start + pd.Timedelta(days=case.days - 1)
        months.update((p.year, p.month) for p in pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"))
    print(f"Prefetching Dst once for {len(months)} calibration/validation months...", flush=True)
    for y, m in sorted(months):
        pg._fetch_dst_cached(int(y), int(m))

    tasks = [(obs, case) for obs in observatories for case in cases]
    print(
        f"Preparing {len(tasks)} calibration/validation candidates; final-test cases excluded; "
        f"target cap={args.cases_per_class_per_year}; station minimum=4.", flush=True,
    )
    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one, obs, case): (obs, case) for obs, case in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            obs, case = futures[future]
            try:
                _, _, data = future.result()
                successes.append((obs, case, data))
                print(f"[{i}/{len(tasks)}] {'CACHE' if data.get('cache_hit') else 'FETCH'} {obs} {case.case_id}", flush=True)
            except Exception as exc:
                failures.append({"observatory": obs, "case": asdict(case), "error": str(exc)})
                print(f"[{i}/{len(tasks)}] FAIL {obs} {case.case_id}: {exc}", flush=True)

    from collections import Counter
    counts = Counter((obs, case.split, int(case.year), case.class_name) for obs, case, _data in successes)
    shortages = []
    for obs in observatories:
        for split in ("calibration", "validation"):
            for year in splits[split]:
                for cls in ("quiet", "active", "storm"):
                    usable = counts[(obs, split, int(year), cls)]
                    if usable < 4:
                        shortages.append({"type": "station_minimum", "observatory": obs, "split": split, "year": year, "class": cls, "usable": usable, "required": 4})
    for split in ("calibration", "validation"):
        for year in splits[split]:
            for cls in ("quiet", "active", "storm"):
                usable = sum(counts[(obs, split, int(year), cls)] for obs in observatories)
                if usable < 8:
                    shortages.append({"type": "pooled_year_minimum", "split": split, "year": year, "class": cls, "usable": usable, "required": 8})
    for split in ("calibration", "validation"):
        for cls in ("quiet", "active", "storm"):
            usable = sum(counts[(obs, split, int(year), cls)] for obs in observatories for year in splits[split])
            if usable < 12:
                shortages.append({"type": "pooled_split_minimum", "split": split, "class": cls, "usable": usable, "required": 12})

    if shortages:
        report = {
            "status": "blocked",
            "reason": "insufficient independent event coverage after data-quality filtering",
            "policy": {
                "target_cases_per_station_year": args.cases_per_class_per_year,
                "minimum_station_cases": 4,
                "minimum_pooled_cases_per_year": 8,
                "minimum_cases_per_class_per_split": 12,
                "under_supplied_years_retain_all_independent_usable_cases": True,
            },
            "shortages": shortages,
            "failures": failures,
        }
        path = Path(args.profile_path).resolve().with_suffix(".blocked.json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(2)

    used = Counter()
    selected = []
    for obs, case, data in sorted(successes, key=lambda x: (x[0], x[1].split, x[1].year, x[1].class_name, x[1].center_date)):
        key = (obs, case.split, int(case.year), case.class_name)
        if used[key] >= args.cases_per_class_per_year:
            continue
        selected.append({"observatory": obs, "case": asdict(case), **data})
        used[key] += 1

    calibration = [x for x in selected if x["case"]["split"] == "calibration"]
    validation = [x for x in selected if x["case"]["split"] == "validation"]
    print(f"Prepared {len(calibration)} calibration and {len(validation)} validation cases using pooled independent-event sufficiency.", flush=True)

    cal_prepared = [PreparedCase(x) for x in calibration]
    val_prepared = [PreparedCase(x) for x in validation]
    profile = _coordinate_descent(cal_prepared, DetectorProfile())
    profile.validate()
    cal_score = _evaluate(cal_prepared, profile)
    val_score = _evaluate(val_prepared, profile)
    cal_passed = _validation_passes(cal_score)
    passed = cal_passed and _validation_passes(val_score)

    output = {
        "status": "certified" if passed else "candidate",
        "profile": asdict(profile),
        "sampling_policy": {
            "target_cases_per_station_year": args.cases_per_class_per_year,
            "minimum_station_cases": 4,
            "minimum_pooled_cases_per_year": 8,
            "minimum_cases_per_class_per_split": 12,
            "under_supplied_years_retain_all_independent_usable_cases": True,
        },
        "selection": {
            "calibration_years": splits["calibration"],
            "validation_years": splits["validation"],
            "final_test_years": splits["test"],
            "final_test_used": False,
            "candidate_pool_per_class_per_year": pool_size,
        },
        "calibration": cal_score,
        "calibration_passed": cal_passed,
        "validation": val_score,
        "failed_source_cases": failures,
        "passed_validation": passed,
        "certification_policy": {
            "search": "feasible-first, worst-group, aggregate-quality, safety-regularized",
            "production_floors": {
                "sample_precision": MIN_PRECISION,
                "sample_recall": MIN_RECALL,
                "sample_f1": MIN_F1,
                "storm_false_alarm_rate": MAX_STORM_FAR,
                "storm_event_precision": MIN_EVENT_PRECISION,
                "storm_event_recall": MIN_EVENT_RECALL,
                "storm_event_f1": MIN_EVENT_F1,
            },
            "calibration_must_also_pass": True,
        },
    }
    path = Path(args.profile_path).resolve()
    if passed:
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"CERTIFIED detector profile written to {path}", flush=True)
    else:
        candidate = path.with_suffix(".candidate.json")
        candidate.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Certification blocked; certified profile was NOT replaced. Candidate: {candidate}", flush=True)
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
