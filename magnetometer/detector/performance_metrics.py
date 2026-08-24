#!/usr/bin/env python3
"""Production detector benchmark: local station reference first, Kp second."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from magnetometer_demo import fetch_kp_gfz, handle_gaps, parse_iaga2002_to_dataframe, run_analysis
from intermagnet_client import fetch_intermagnet_long_range
from adaptive_detector import detect_adaptive
from local_event_benchmark import build_local_reference, compare_events


@dataclass
class BinaryMetrics:
    samples: int
    positive_samples: int
    predicted_positive: int
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    false_alarm_rate: float
    balanced_accuracy: float
    prevalence: float


def _safe(a, b):
    return float(a / b) if b else 0.0


def binary_metrics(reference, prediction):
    r = np.asarray(reference, bool)
    p = np.asarray(prediction, bool)
    if len(r) != len(p):
        raise ValueError("reference and prediction lengths differ")
    tp = int(np.sum(r & p)); tn = int(np.sum(~r & ~p))
    fp = int(np.sum(~r & p)); fn = int(np.sum(r & ~p))
    precision = _safe(tp, tp + fp); recall = _safe(tp, tp + fn)
    f1 = _safe(2 * precision * recall, precision + recall)
    specificity = _safe(tn, tn + fp)
    return BinaryMetrics(
        len(r), int(r.sum()), int(p.sum()), tp, tn, fp, fn,
        precision, recall, f1, specificity, _safe(fp, fp + tn),
        0.5 * (recall + specificity), _safe(int(r.sum()), len(r)),
    )


def _runs(mask):
    x = np.asarray(mask, bool)
    if not len(x):
        return []
    p = np.concatenate(([False], x, [False]))
    s = np.flatnonzero(~p[:-1] & p[1:])
    e = np.flatnonzero(p[:-1] & ~p[1:])
    return list(zip(s.tolist(), e.tolist()))


def stability_metrics(flags, cadence_s):
    v = np.asarray(flags, object)
    active = v != "quiet"
    changes = int(np.sum(v[1:] != v[:-1])) if len(v) > 1 else 0
    flagged_runs = _runs(active); quiet_runs = _runs(~active)
    days = len(v) * cadence_s / 86400
    return {
        "samples": len(v),
        "state_changes": changes,
        "state_changes_per_day": _safe(changes, days),
        "flagged_samples": int(active.sum()),
        "flagged_fraction": _safe(int(active.sum()), len(v)),
        "longest_flagged_minutes": max((e - s for s, e in flagged_runs), default=0) * cadence_s / 60,
        "longest_quiet_minutes": max((e - s for s, e in quiet_runs), default=0) * cadence_s / 60,
    }


def _align_kp(index, kp):
    k = kp.copy()
    k.index = pd.to_datetime(k.index, utc=True)
    return k.reindex(index, method="ffill", tolerance=pd.Timedelta("6h"))


def evaluate_period(observatory, start_date, days, column="f_nt", cadence_s=60, chunk_days=7):
    text = fetch_intermagnet_long_range(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
        chunk_days=chunk_days,
    )
    series = parse_iaga2002_to_dataframe(text)[column]
    df = handle_gaps(series, max_gap_samples=3).dropna().to_frame(column)
    if len(df) < 100:
        raise RuntimeError(f"Insufficient valid magnetometer samples: {len(df)}")

    result = run_analysis(
        df[column].to_numpy(), cadence_s,
        label=f"benchmark {observatory} {start_date}",
        start_time=df.index.min().to_pydatetime(),
    )
    residual = np.asarray(result["residual"], float)

    # PRIMARY: station-local reference. It uses no Kp/Dst and no detector
    # thresholds. This is the metric used to redesign the detector.
    reference, reference_diag = build_local_reference(residual, cadence_s)
    flags, detector_diag = detect_adaptive(residual, cadence_s)
    predicted = flags != "quiet"
    local = {
        "sample": asdict(binary_metrics(reference, predicted)),
        "event": compare_events(reference, predicted, cadence_s),
    }

    # SECONDARY: global context only. Failure here must never invalidate the
    # local benchmark because the production detector does not depend on Kp.
    global_context: dict[str, Any] = {"available": False}
    try:
        kp = fetch_kp_gfz(
            pd.Timestamp(df.index.min()).strftime("%Y-%m-%d"),
            pd.Timestamp(df.index.max()).strftime("%Y-%m-%d"),
        )
        ka = _align_kp(df.index, kp).to_numpy(float)
        valid = np.isfinite(ka)
        kflags = flags[valid]; ka = ka[valid]
        active_ref = ka >= 4
        storm_ref = ka >= 5
        active_pred = np.isin(kflags, ["active", "minor_storm", "major_storm", "severe_storm", "anomaly"])
        storm_pred = np.isin(kflags, ["minor_storm", "major_storm", "severe_storm"])
        global_context = {
            "available": True,
            "kp_range": [float(np.nanmin(ka)), float(np.nanmax(ka))],
            "active_vs_kp_ge_4": asdict(binary_metrics(active_ref, active_pred)),
            "storm_vs_kp_ge_5": asdict(binary_metrics(storm_ref, storm_pred)),
        }
    except Exception as exc:
        global_context = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "benchmark_version": "local-v2",
        "reference_status": "station-local reference; not Kp ground truth",
        "observatory": observatory,
        "start_date": start_date,
        "days": days,
        "column": column,
        "cadence_s": cadence_s,
        "chunk_days": chunk_days,
        "valid_samples": len(residual),
        "local_reference": reference_diag,
        "local_detector": detector_diag,
        "local": local,
        "global_context": global_context,
        "stability": stability_metrics(flags, cadence_s),
        "flag_counts": {str(k): int(v) for k, v in zip(*np.unique(flags, return_counts=True))},
    }


def _print(title, metrics):
    print(f"\n{title}\n{'-' * len(title)}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:34s}: {v:.4f}")
        else:
            print(f"{k:34s}: {v}")


def main():
    ap = argparse.ArgumentParser(description="Local ground-truth/reference magnetometer detector benchmark")
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--column", default="f_nt", choices=["x_nt", "y_nt", "z_nt", "f_nt"])
    ap.add_argument("--cadence-s", type=int, default=60)
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--output")
    a = ap.parse_args()
    r = evaluate_period(a.observatory, a.start_date, a.days, a.column, a.cadence_s, a.chunk_days)

    print("\n=== MAGNETOMETER DETECTOR PERFORMANCE ===")
    print(f"Observatory: {r['observatory']} | Period: {r['start_date']} | Days: {r['days']}")
    print("PRIMARY: LOCAL STATION REFERENCE (NO Kp/Dst)")
    _print("Local sample metrics", r["local"]["sample"])
    _print("Local event metrics", r["local"]["event"])
    _print("Adaptive detector diagnostics", r["local_detector"])
    _print("Operational stability", r["stability"])

    print("\nSECONDARY: GLOBAL Kp AGREEMENT (NOT GROUND TRUTH)")
    if r["global_context"].get("available"):
        print(f"Kp range: {r['global_context']['kp_range']}")
        _print("Active vs Kp >= 4", r["global_context"]["active_vs_kp_ge_4"])
        _print("Storm vs Kp >= 5", r["global_context"]["storm_vs_kp_ge_5"])
    else:
        print(f"Kp unavailable: {r['global_context'].get('error', 'unknown error')}")

    print("\nFlag counts:")
    for k, v in r["flag_counts"].items():
        print(f"  {k:16s}: {v}")

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"\nJSON report written to {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
