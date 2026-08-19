#!/usr/bin/env python3
"""Calibrate detector heuristics without touching the final test set.

Calibration is deliberately chronological:
  historical years -> calibration optimization -> validation checkpoint.

The final-test years are never used here.  A profile is written as
``certified`` only when the selected parameters clear the validation floors.
The live detector refuses non-certified profiles.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg
from detector_core import DetectorProfile, detect_activity_masks


PARAMETER_GRID = {
    "active_nt": (15.0, 20.0, 25.0, 30.0, 35.0),
    "storm_nt": (35.0, 50.0, 60.0, 70.0, 80.0),
    "active_fast_ratio": (1.10, 1.25, 1.40),
    "storm_fast_ratio": (1.60, 1.80, 2.00),
    "storm_upper_ratio": (1.00, 1.10, 1.20),
    "storm_release_ratio": (0.60, 0.65, 0.70),
}


def _binary(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth)); tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _score_case(data: dict, profile: DetectorProfile) -> tuple[dict, dict]:
    active, storm, _major, _severe, _ = detect_activity_masks(
        data["residual"], cadence_s=data["cadence_s"], profile=profile
    )
    known = data["refs"]["known"] & np.isfinite(data["residual"])
    return (
        _binary(active[known], data["refs"]["active"][known]),
        _binary(storm[known], data["refs"]["storm"][known]),
    )


def _aggregate(rows: Iterable[tuple[dict, dict]]) -> dict:
    a = []; s = []
    for active, storm in rows:
        a.append(active); s.append(storm)
    def agg(items):
        keys = ("tp", "tn", "fp", "fn")
        counts = {k: sum(int(x[k]) for x in items) for k in keys}
        return _binary_from_counts(**counts)
    return {"active": agg(a), "storm": agg(s)}


def _binary_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _objective(score: dict) -> float:
    a = score["active"]; s = score["storm"]
    if a["f1"] is None or s["f1"] is None:
        return -1e9
    # Optimize useful detection while making false alarms materially expensive.
    return float(0.35 * a["f1"] + 0.65 * s["f1"] - 1.50 * (s["far"] or 1.0))


def _evaluate(cases: list[dict], profile: DetectorProfile) -> dict:
    return _aggregate(_score_case(c, profile) for c in cases)


def _coordinate_descent(cases: list[dict], base: DetectorProfile) -> DetectorProfile:
    profile = base
    ordered = (
        "active_nt", "storm_nt", "active_fast_ratio",
        "storm_fast_ratio", "storm_upper_ratio", "storm_release_ratio",
    )
    for name in ordered:
        best = profile
        best_obj = _objective(_evaluate(cases, profile))
        for value in PARAMETER_GRID[name]:
            if name == "storm_nt" and value <= profile.active_nt:
                continue
            if name == "active_nt" and value >= profile.storm_nt:
                continue
            candidate = replace(profile, **{name: value})
            try:
                candidate.validate()
            except ValueError:
                continue
            obj = _objective(_evaluate(cases, candidate))
            if obj > best_obj + 1e-12:
                best, best_obj = candidate, obj
        profile = best
    return profile


def _passes_validation(score: dict, min_precision: float, min_recall: float, min_f1: float, max_far: float) -> bool:
    for name in ("active", "storm"):
        m = score[name]
        if (m["precision"] or 0.0) < min_precision:
            return False
        if (m["recall"] or 0.0) < min_recall:
            return False
        if (m["f1"] or 0.0) < min_f1:
            return False
    return (score["storm"]["far"] or 1.0) <= max_far


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronologically calibrate the magnetometer detector heuristics.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--min-precision", type=float, default=0.85)
    ap.add_argument("--min-recall", type=float, default=0.80)
    ap.add_argument("--min-f1", type=float, default=0.82)
    ap.add_argument("--max-storm-far", type=float, default=0.01)
    ap.add_argument("--profile-path", default=str(HERE / "detector_profile.json"))
    args = ap.parse_args()

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted(set(int(x.strip()) for x in args.years.split(",") if x.strip()))
    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    calibration_cases = []
    validation_cases = []
    failures = []

    for observatory in observatories:
        for case in cases:
            try:
                data = pg.load_case(observatory, case)
                if case.split == "calibration":
                    calibration_cases.append(data)
                elif case.split == "validation":
                    validation_cases.append(data)
            except Exception as exc:
                failures.append({"observatory": observatory, "case": asdict(case), "error": str(exc)})

    if not calibration_cases or not validation_cases:
        raise SystemExit("Calibration requires successful calibration and validation cases.")

    profile = _coordinate_descent(calibration_cases, DetectorProfile())
    calibration_score = _evaluate(calibration_cases, profile)
    validation_score = _evaluate(validation_cases, profile)
    passed = _passes_validation(validation_score, args.min_precision, args.min_recall, args.min_f1, args.max_storm_far)

    output = {
        "status": "certified" if passed else "candidate",
        "profile": asdict(profile),
        "selection": {
            "method": "chronological coordinate descent",
            "calibration_years": splits["calibration"],
            "validation_years": splits["validation"],
            "final_test_years": splits["test"],
            "final_test_used": False,
        },
        "calibration_score": calibration_score,
        "validation_score": validation_score,
        "validation_floors": {
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "min_f1": args.min_f1,
            "max_storm_far": args.max_storm_far,
        },
        "passed_validation": passed,
        "failed_cases": failures,
    }
    path = Path(args.profile_path).resolve()
    if passed:
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"CERTIFIED detector profile written to {path}")
    else:
        candidate = path.with_suffix(".candidate.json")
        candidate.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Validation failed; certified profile was NOT replaced. Candidate: {candidate}")
    print(json.dumps({"status": output["status"], "profile": output["profile"], "calibration": calibration_score, "validation": validation_score}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
