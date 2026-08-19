"""Canonical production calibrator with statistically valid pooled event sufficiency.

Ten cases per class/year is a target cap, not an assumption that ten independent
storm episodes exist every year. The calibrator blocks only when minimum
independent station and pooled event coverage is unavailable. Final-test cases
are never loaded here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import calibrate_detector as cd
from .. import production_grade_validation as pg
from ..detector_core import DetectorProfile

MIN_STATION_CASES = 4
MIN_POOLED_CASES_PER_YEAR = 8
MIN_CASES_PER_CLASS_PER_SPLIT = 12
DEFAULT_WORKERS = 6


def _load_one(obs: str, case: pg.Case):
    return obs, case, pg.load_case(obs, case)


def _check_sufficiency(successes, observatories, splits):
    counts = Counter((obs, case.split, int(case.year), case.class_name) for obs, case, _data in successes)
    shortages = []
    for obs in observatories:
        for split in ("calibration", "validation"):
            for year in splits[split]:
                for cls in ("quiet", "active", "storm"):
                    n = counts[(obs, split, int(year), cls)]
                    if n < MIN_STATION_CASES:
                        shortages.append({"type":"station_minimum","observatory":obs,"split":split,"year":year,"class":cls,"usable":n,"required":MIN_STATION_CASES})
    for split in ("calibration", "validation"):
        for year in splits[split]:
            for cls in ("quiet", "active", "storm"):
                n = sum(counts[(obs, split, int(year), cls)] for obs in observatories)
                if n < MIN_POOLED_CASES_PER_YEAR:
                    shortages.append({"type":"pooled_year_minimum","split":split,"year":year,"class":cls,"usable":n,"required":MIN_POOLED_CASES_PER_YEAR})
    for split in ("calibration", "validation"):
        for cls in ("quiet", "active", "storm"):
            n = sum(counts[(obs, split, int(year), cls)] for obs in observatories for year in splits[split])
            if n < MIN_CASES_PER_CLASS_PER_SPLIT:
                shortages.append({"type":"pooled_split_minimum","split":split,"class":cls,"usable":n,"required":MIN_CASES_PER_CLASS_PER_SPLIT})
    return shortages


def _validation_ok(score):
    for name in ("active", "storm"):
        m = score[name]
        if (m["precision"] or 0.0) < 0.85 or (m["recall"] or 0.0) < 0.80 or (m["f1"] or 0.0) < 0.82:
            return False
    if (score["storm"]["far"] or 1.0) > 0.01:
        return False
    e = score.get("storm_event", {})
    return (e.get("precision") or 0.0) >= 0.85 and (e.get("recall") or 0.0) >= 0.90 and (e.get("f1") or 0.0) >= 0.87


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronological production detector calibration with pooled independent-event sufficiency.")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year", type=int, default=10)
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--profile-path", default=str(Path(__file__).resolve().parents[1] / "detector_profile.json"))
    args = ap.parse_args()
    if args.cases_per_class_per_year < 10:
        raise SystemExit("Production calibration requires --cases-per-class-per-year >= 10.")

    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
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
    print(f"Preparing {len(tasks)} calibration/validation candidates; final-test cases excluded; target cap={args.cases_per_class_per_year}; station minimum={MIN_STATION_CASES}.", flush=True)
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
                failures.append({"observatory":obs,"case":asdict(case),"error":str(exc)})
                print(f"[{i}/{len(tasks)}] FAIL {obs} {case.case_id}: {exc}", flush=True)

    shortages = _check_sufficiency(successes, observatories, splits)
    if shortages:
        report = {"status":"blocked","reason":"insufficient independent event coverage after data-quality filtering","policy":{"target_cases_per_station_year":args.cases_per_class_per_year,"minimum_station_cases":MIN_STATION_CASES,"minimum_pooled_cases_per_year":MIN_POOLED_CASES_PER_YEAR,"minimum_cases_per_class_per_split":MIN_CASES_PER_CLASS_PER_SPLIT},"shortages":shortages,"failures":failures}
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
        selected.append({"observatory":obs,"case":asdict(case),**data})
        used[key] += 1

    calibration = [x for x in selected if x["case"]["split"] == "calibration"]
    validation = [x for x in selected if x["case"]["split"] == "validation"]
    print(f"Prepared {len(calibration)} calibration and {len(validation)} validation cases using pooled independent-event sufficiency.", flush=True)
    cal_prepared = [cd.PreparedCase(x) for x in calibration]
    val_prepared = [cd.PreparedCase(x) for x in validation]
    profile = cd._coordinate_descent(cal_prepared, DetectorProfile())
    profile.validate()
    cal_score = cd._evaluate(cal_prepared, profile)
    val_score = cd._evaluate(val_prepared, profile)
    passed = _validation_ok(val_score)

    output = {"status":"certified" if passed else "candidate","profile":asdict(profile),"sampling_policy":{"target_cases_per_station_year":args.cases_per_class_per_year,"minimum_station_cases":MIN_STATION_CASES,"minimum_pooled_cases_per_year":MIN_POOLED_CASES_PER_YEAR,"minimum_cases_per_class_per_split":MIN_CASES_PER_CLASS_PER_SPLIT,"under_supplied_years_retain_all_independent_usable_cases":True},"selection":{"calibration_years":splits["calibration"],"validation_years":splits["validation"],"final_test_years":splits["test"],"final_test_used":False,"candidate_pool_per_class_per_year":pool_size},"calibration":cal_score,"validation":val_score,"failed_source_cases":failures,"passed_validation":passed}
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
