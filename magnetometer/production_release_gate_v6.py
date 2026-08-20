#!/usr/bin/env python3
"""Strict production release gate for the causal detector.

Calibration creates only a candidate profile. This gate evaluates that exact
candidate on chronological validation and then on the untouched final-test
split. Only a full validation + final-test pass promotes the candidate to the
certified profile used by live inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import causal_baseline
from . import production_grade_validation as pg
from . import production_release_gate_v5 as v5
from .detecting import calibrate as canonical_calibration
from .detector_core import DetectorProfile

pg.pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline
pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data" / "case_cache_causal_v10"

# Importing the canonical calibrator patches PreparedCase.predict to the exact
# live detector implementation, preventing calibration/release drift.
cd = canonical_calibration._impl


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _load_candidate(path: Path) -> tuple[dict, DetectorProfile]:
    if not path.exists():
        raise SystemExit(f"FAIL: candidate profile missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "candidate":
        raise SystemExit("FAIL: release gate requires a candidate profile; certified profile must never be used as a calibration candidate")
    profile = DetectorProfile.from_dict(payload.get("profile", payload))
    profile.validate()
    return payload, profile


def _gate(rows, observatories, splits, split_name, profile):
    accepted, rejected = v5._quality_filter(rows)
    shortages = v5._shortages(accepted, observatories, splits, split_name)
    if shortages:
        return accepted, rejected, {"passed": False, "score": None, "checks": {"coverage_policy": False}, "shortages": shortages}
    scores = v5._per_case_scores(accepted, profile)
    result = v5._passes(scores)
    result["coverage_policy"] = v5._coverage_ok(accepted)
    result["passed"] = bool(result["passed"] and result["coverage_policy"])
    return accepted, rejected, result


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict causal detector production release gate v6.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--candidate-profile-path", default=str(Path(__file__).resolve().with_name("detector_profile.candidate.json")))
    ap.add_argument("--certified-profile-path", default=str(Path(__file__).resolve().with_name("detector_profile.json")))
    args = ap.parse_args()
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Release gate requires --cases-per-class-per-year >= 10.")

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    if len(years) < 3:
        raise SystemExit("Release gate requires at least three chronological years.")

    candidate_path = Path(args.candidate_profile_path).resolve()
    certified_path = Path(args.certified_profile_path).resolve()
    candidate_payload, profile = _load_candidate(candidate_path)

    splits, cases = pg.discover_suite(years, args.cases_per_class_per_year, args.window_days)
    pretest = [c for c in cases if c.split != "test"]
    final_test = [c for c in cases if c.split == "test"]

    print("=" * 100)
    print("MAGNETOMETER STRICT CAUSAL PRODUCTION RELEASE GATE v6")
    print("=" * 100)
    print(f"Candidate: {candidate_path}")
    print(f"Certified target: {certified_path}")
    print(f"Calibration years: {splits['calibration']}")
    print(f"Validation years: {splits['validation']}")
    print(f"Final-test years: {splits['test']}")
    print("Final-test access: BLOCKED until validation passes")

    pre_rows, pre_failures = v5._load(observatories, pretest)
    validation_rows_raw = [r for r in pre_rows if r["case"]["split"] == "validation"]
    validation_rows, validation_rejected, validation_gate = _gate(validation_rows_raw, observatories, splits, "validation", profile)

    if not validation_gate["passed"]:
        report = {
            "schema_version": "7.0-candidate-promotion-gate",
            "release_status": "FAIL",
            "candidate": {"path": str(candidate_path), "profile": candidate_payload},
            "splits": splits,
            "validation": validation_gate,
            "final_test": {"accessed": False, "reason": "validation did not pass; holdout not fetched"},
            "data_quality": {"pretest_failures": pre_failures, "validation_rejected": validation_rejected, "accepted_validation_cases": len(validation_rows)},
        }
        out = Path(__file__).resolve().parent / "data" / "magnetometer_exact_profile_release_gate.json"
        _atomic_json(out, report)
        print("Validation gate: FAIL — final-test holdout untouched")
        print(f"Report: {out}")
        raise SystemExit(2)

    test_raw, test_failures = v5._load(observatories, final_test)
    test_rows, test_rejected, final_gate = _gate(test_raw, observatories, splits, "test", profile)
    report = {
        "schema_version": "7.0-candidate-promotion-gate",
        "release_status": "PASS" if final_gate["passed"] else "FAIL",
        "candidate": {"path": str(candidate_path), "profile": candidate_payload},
        "splits": splits,
        "validation": validation_gate,
        "final_test": {"accessed": True, **final_gate},
        "data_quality": {"pretest_failures": pre_failures, "validation_rejected": validation_rejected, "final_test_failures": test_failures, "final_test_rejected": test_rejected, "accepted_validation_cases": len(validation_rows), "accepted_final_test_cases": len(test_rows)},
    }
    out = Path(__file__).resolve().parent / "data" / "magnetometer_exact_profile_release_gate.json"
    _atomic_json(out, report)

    if final_gate["passed"]:
        promoted = dict(candidate_payload)
        promoted["status"] = "certified"
        promoted["certification"] = {
            "release_gate": "v6",
            "validation_passed": True,
            "final_test_passed": True,
            "final_test_years": splits["test"],
        }
        _atomic_json(certified_path, promoted)
        print("FINAL RELEASE GATE: PASS")
        print(f"Certified profile promoted atomically to {certified_path}")
        raise SystemExit(0)

    print("FINAL RELEASE GATE: FAIL")
    print("Certified profile was NOT replaced.")
    print(f"Report: {out}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
