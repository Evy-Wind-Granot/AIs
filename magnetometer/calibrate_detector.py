#!/usr/bin/env python3
"""Production calibration for the causal-disturbance-v2 detector.

This module deliberately avoids monkey-patching and large ratio searches.
Only the parameters that materially define the operating point are calibrated:
absolute active/storm thresholds, robust-normalized evidence thresholds, and
state-machine persistence. Calibration never loads or scores the final-test
split.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import production_grade_validation as pg
from detector_core import DETECTOR_VERSION, DetectorProfile, detect_activity_masks

MIN_PRECISION = 0.85
MIN_RECALL = 0.80
MIN_F1 = 0.82
MAX_STORM_FAR = 0.01
MIN_EVENT_PRECISION = 0.85
MIN_EVENT_RECALL = 0.90
MIN_EVENT_F1 = 0.87
DEFAULT_WORKERS = 6
CACHE_NAMESPACE = "case_cache_causal_v7"
ACTIVE_THRESHOLDS = (10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0, 35.0, 40.0)
STORM_THRESHOLDS = (25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 75.0, 100.0)
ACTIVE_Z = (2.5, 3.0, 3.5, 4.0)
STORM_Z = (3.5, 4.0, 4.5, 5.0, 6.0)
ACTIVE_ON = (1.0, 2.0, 3.0, 5.0)
STORM_ON = (5.0, 10.0, 15.0, 20.0)


class PreparedCase:
    __slots__ = ("observatory", "case", "residual", "cadence_s", "known", "active_ref", "storm_ref")

    def __init__(self, observatory: str, case: pg.Case, data: dict[str, Any]) -> None:
        self.observatory = observatory
        self.case = case
        self.residual = np.asarray(data["residual"], dtype=float)
        self.cadence_s = float(data["cadence_s"])
        refs = data["refs"]
        self.known = np.asarray(refs["known"], dtype=bool)
        self.active_ref = np.asarray(refs["active"], dtype=bool)
        self.storm_ref = np.asarray(refs["storm"], dtype=bool)


def _binary(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth)); tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _binary_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    far = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "far": far}


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _binary_from_counts(
        sum(int(x["tp"]) for x in rows), sum(int(x["tn"]) for x in rows),
        sum(int(x["fp"]) for x in rows), sum(int(x["fn"]) for x in rows),
    )


def _score_case(case: PreparedCase, profile: DetectorProfile) -> tuple[dict, dict, dict, dict]:
    active, storm, _major, _severe, _diag = detect_activity_masks(case.residual, cadence_s=case.cadence_s, profile=profile, include_anomaly=False)
    known = case.known & np.isfinite(case.residual)
    active_sample = _binary(active[known], case.active_ref[known])
    storm_sample = _binary(storm[known], case.storm_ref[known])
    active_pred_events = pg.pm.bool_events(active & known, case.cadence_s, 1800, 300)
    active_ref_events = pg.pm.bool_events(case.active_ref & known, case.cadence_s, 21600, 10800)
    storm_pred_events = pg.pm.bool_events(storm & known, case.cadence_s, 1800, 300)
    storm_ref_events = pg.pm.bool_events(case.storm_ref & known, case.cadence_s, 21600, 10800)
    return active_sample, storm_sample, pg.pm.match_events(active_pred_events, active_ref_events, case.cadence_s), pg.pm.match_events(storm_pred_events, storm_ref_events, case.cadence_s)


def _evaluate(cases: Sequence[PreparedCase], profile: DetectorProfile) -> dict[str, Any]:
    active_rows=[]; storm_rows=[]; active_events=[]; storm_events=[]
    for case in cases:
        a,s,ae,se = _score_case(case, profile)
        active_rows.append(a); storm_rows.append(s); active_events.append(ae); storm_events.append(se)
    def merge(rows: Sequence[dict]) -> dict:
        ref=sum(int(r["reference_events"]) for r in rows); pred=sum(int(r["predicted_events"]) for r in rows); matched=sum(int(r["matched_events"]) for r in rows)
        precision=matched/pred if pred else None; recall=matched/ref if ref else None
        f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
        return {"reference_events":ref,"predicted_events":pred,"matched_events":matched,"missed_events":ref-matched,"false_positive_events":pred-matched,"precision":precision,"recall":recall,"f1":f1}
    return {"active":_aggregate(active_rows),"storm":_aggregate(storm_rows),"active_event":merge(active_events),"storm_event":merge(storm_events)}


def _violation(score: dict[str, Any]) -> float:
    a,s,e=score["active"],score["storm"],score["storm_event"]
    return float(sum((
        max(0.0,MIN_PRECISION-(a["precision"] or 0.0)),max(0.0,MIN_RECALL-(a["recall"] or 0.0)),max(0.0,MIN_F1-(a["f1"] or 0.0)),
        max(0.0,MIN_PRECISION-(s["precision"] or 0.0)),max(0.0,MIN_RECALL-(s["recall"] or 0.0)),max(0.0,MIN_F1-(s["f1"] or 0.0)),
        max(0.0,(s["far"] or 1.0)-MAX_STORM_FAR),max(0.0,MIN_EVENT_PRECISION-(e["precision"] or 0.0)),
        max(0.0,MIN_EVENT_RECALL-(e["recall"] or 0.0)),max(0.0,MIN_EVENT_F1-(e["f1"] or 0.0)),
    )))


def _candidate_key(score: dict[str, Any], profile: DetectorProfile, cases: Sequence[PreparedCase]) -> tuple:
    violation=_violation(score); worst=1.0
    for case in cases:
        a,s,_ae,se=_score_case(case,profile)
        worst=min(worst,*[float(a[k] or 0.0) for k in ("precision","recall","f1")],*[float(s[k] or 0.0) for k in ("precision","recall","f1")],*[float(se[k] or 0.0) for k in ("precision","recall","f1")])
    quality=float(np.mean([float(score["active"]["f1"] or 0.0),float(score["storm"]["f1"] or 0.0),float(score["active_event"]["f1"] or 0.0),float(score["storm_event"]["f1"] or 0.0)]))
    conservative=(-float(profile.active_nt),-float(profile.storm_nt),-float(profile.active_on_minutes),-float(profile.storm_on_minutes))
    return (round(violation,12),round(max(0.0,0.75-worst),12),-round(quality,12),conservative)


def _search(cases: Sequence[PreparedCase], base: DetectorProfile) -> DetectorProfile:
    candidates=[]
    for active_nt in ACTIVE_THRESHOLDS:
        for storm_nt in STORM_THRESHOLDS:
            if storm_nt <= active_nt: continue
            p=replace(base,active_nt=active_nt,storm_nt=storm_nt)
            candidates.append((_candidate_key(_evaluate(cases,p),p,cases),p))
    if not candidates: return base
    profile=min(candidates,key=lambda x:x[0])[1]
    for name,values in (("active_z",ACTIVE_Z),("storm_z",STORM_Z),("active_on_minutes",ACTIVE_ON),("storm_on_minutes",STORM_ON)):
        best=profile; best_key=_candidate_key(_evaluate(cases,profile),profile,cases)
        for value in values:
            candidate=replace(profile,**{name:value})
            try: candidate.validate()
            except ValueError: continue
            key=_candidate_key(_evaluate(cases,candidate),candidate,cases)
            if key < best_key: best,best_key=candidate,key
        profile=best
    return profile


def _passes(score: dict[str, Any]) -> bool:
    for name in ("active","storm"):
        row=score[name]
        if (row["precision"] or 0.0)<MIN_PRECISION or (row["recall"] or 0.0)<MIN_RECALL or (row["f1"] or 0.0)<MIN_F1: return False
    if (score["storm"]["far"] or 1.0)>MAX_STORM_FAR: return False
    e=score["storm_event"]
    return (e["precision"] or 0.0)>=MIN_EVENT_PRECISION and (e["recall"] or 0.0)>=MIN_EVENT_RECALL and (e["f1"] or 0.0)>=MIN_EVENT_F1


def _load_one(observatory: str, case: pg.Case) -> tuple[str, pg.Case, dict]:
    return observatory, case, pg.load_case(observatory, case)


def main() -> None:
    ap=argparse.ArgumentParser(description="Production calibration for the causal-disturbance-v2 detector.")
    ap.add_argument("--observatory",default="VIC,BOU"); ap.add_argument("--years",default="2022,2023,2024,2025")
    ap.add_argument("--cases-per-class-per-year",type=int,default=10); ap.add_argument("--window-days",type=int,default=7)
    ap.add_argument("--workers",type=int,default=DEFAULT_WORKERS); ap.add_argument("--profile-path",default=str(HERE/"detector_profile.json"))
    args=ap.parse_args()
    if args.cases_per_class_per_year<10: raise SystemExit("Production calibration requires at least 10 target cases per class per year.")
    observatories=[x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    years=sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
    if len(years)<3: raise SystemExit("At least three chronological years are required.")
    pg.DEFAULT_CACHE_DIR=HERE/"data"/CACHE_NAMESPACE; pg.DEFAULT_CACHE_DIR.mkdir(parents=True,exist_ok=True)
    splits,cases=pg.discover_suite(years,args.cases_per_class_per_year*2,args.window_days)
    usable_cases=[c for c in cases if c.split in ("calibration","validation")]
    kp=pg._fetch_kp_cached(f"{min(years):04d}-01-01",f"{max(years):04d}-12-31"); pg._fetch_kp_cached=lambda _start,_end:kp
    months=set()
    for case in usable_cases:
        start=pd.Timestamp(case.start_date,tz="UTC"); end=start+pd.Timedelta(days=case.days-1)
        months.update((p.year,p.month) for p in pd.period_range(start.strftime("%Y-%m"),end.strftime("%Y-%m"),freq="M"))
    for year,month in sorted(months): pg._fetch_dst_cached(int(year),int(month))
    tasks=[(obs,case) for obs in observatories for case in usable_cases]; successes=[]; failures=[]
    with ThreadPoolExecutor(max_workers=max(1,min(int(args.workers),8))) as pool:
        futures={pool.submit(_load_one,obs,case):(obs,case) for obs,case in tasks}
        for future in as_completed(futures):
            obs,case=futures[future]
            try:
                _,_,data=future.result(); successes.append((obs,case,data)); print(f"OK {obs} {case.case_id}",flush=True)
            except Exception as exc:
                failures.append({"observatory":obs,"case":asdict(case),"error":str(exc)}); print(f"SKIP {obs} {case.case_id}: {exc}",flush=True)
    from collections import Counter
    counts=Counter((obs,c.split,int(c.year),c.class_name) for obs,c,_ in successes); shortages=[]
    for obs in observatories:
        for split in ("calibration","validation"):
            for year in splits[split]:
                for cls in ("quiet","active","storm"):
                    n=counts[(obs,split,int(year),cls)]
                    if n<4: shortages.append({"type":"station_year_class","observatory":obs,"split":split,"year":year,"class":cls,"usable":n,"required":4})
    for split in ("calibration","validation"):
        for year in splits[split]:
            for cls in ("quiet","active","storm"):
                n=sum(counts[(obs,split,int(year),cls)] for obs in observatories)
                if n<8: shortages.append({"type":"pooled_year_class","split":split,"year":year,"class":cls,"usable":n,"required":8})
    if shortages:
        output={"status":"blocked","detector_version":DETECTOR_VERSION,"reason":"insufficient independent event coverage after data-quality filtering","shortages":shortages,"failed_source_cases":failures}
        Path(args.profile_path).resolve().with_suffix(".blocked.json").write_text(json.dumps(output,indent=2)+"\n"); print(json.dumps(output,indent=2)); raise SystemExit(2)
    used=Counter(); selected=[]
    for obs,case,data in sorted(successes,key=lambda x:(x[0],x[1].split,x[1].year,x[1].class_name,x[1].center_date)):
        key=(obs,case.split,int(case.year),case.class_name)
        if used[key]<args.cases_per_class_per_year: selected.append((obs,case,data)); used[key]+=1
    calibration=[PreparedCase(o,c,d) for o,c,d in selected if c.split=="calibration"]
    validation=[PreparedCase(o,c,d) for o,c,d in selected if c.split=="validation"]
    print(f"Prepared {len(calibration)} calibration and {len(validation)} validation cases.",flush=True)
    profile=_search(calibration,DetectorProfile()); profile.validate()
    cal_score=_evaluate(calibration,profile); val_score=_evaluate(validation,profile)
    cal_passed=_passes(cal_score); val_passed=_passes(val_score); certified=cal_passed and val_passed
    output={"status":"certified" if certified else "candidate","detector_version":DETECTOR_VERSION,"profile":asdict(profile),"sampling_policy":{"target_cases_per_station_year":args.cases_per_class_per_year,"minimum_station_cases":4,"minimum_pooled_cases_per_year":8,"under_supplied_years_retain_all_independent_usable_cases":False},"selection":{"calibration_years":splits["calibration"],"validation_years":splits["validation"],"final_test_years":splits["test"],"final_test_used":False,"candidate_pool_per_class_per_year":args.cases_per_class_per_year*2},"calibration":cal_score,"calibration_passed":cal_passed,"validation":val_score,"passed_validation":val_passed,"failed_source_cases":failures,"certification_policy":{"search":"joint-threshold search then bounded normalized-evidence/persistence refinement","production_floors":{"sample_precision":MIN_PRECISION,"sample_recall":MIN_RECALL,"sample_f1":MIN_F1,"storm_false_alarm_rate":MAX_STORM_FAR,"storm_event_precision":MIN_EVENT_PRECISION,"storm_event_recall":MIN_EVENT_RECALL,"storm_event_f1":MIN_EVENT_F1},"calibration_must_pass":True,"validation_must_pass":True,"final_test_used":False}}
    path=Path(args.profile_path).resolve()
    if certified:
        path.write_text(json.dumps(output,indent=2)+"\n"); print(f"CERTIFIED detector profile written to {path}",flush=True); raise SystemExit(0)
    candidate=path.with_suffix(".candidate.json"); candidate.write_text(json.dumps(output,indent=2)+"\n"); print(f"Certification blocked; certified profile was NOT replaced. Candidate: {candidate}",flush=True); print(json.dumps(output,indent=2)); raise SystemExit(2)


if __name__=="__main__": main()
