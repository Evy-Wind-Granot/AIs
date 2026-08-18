#!/usr/bin/env python3
"""
Magnetometer performance benchmark.

Measures:
  - data completeness
  - QDC/baseline residual quality (MAE, RMSE/RMS, bias, P95)
  - quiet-period residual quality
  - activity/storm classification agreement against Kp/Dst when available
  - false-alarm and miss rates
  - processing throughput
  - daily stability of residual RMS

This is intentionally separate from magnetometer_demo.py so the production
pipeline stays focused on inference while the benchmark can evolve freely.

Examples:
    python magnetometer/performance_metrics.py --observatory VIC \
        --start-date 2024-01-01 --days 7

    python magnetometer/performance_metrics.py --observatory VIC \
        --start-date 2024-01-01 --days 30 --output-dir magnetometer/data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Allow execution from repository root or from inside magnetometer/.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from magnetometer_demo import (  # noqa: E402
    build_design_matrix,
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    parse_iaga2002_to_dataframe,
    robust_harmonic_baseline,
)


# Keep classification thresholds consistent with magnetometer_demo.py.
QUIET_LIMIT_NT = 15.0
ACTIVE_LIMIT_NT = 35.0
MINOR_STORM_LIMIT_NT = 70.0
MAJOR_STORM_LIMIT_NT = 150.0
SEVERE_STORM_LIMIT_NT = 300.0



def finite_values(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.isfinite(x)]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, Optional[float]]:
    """Return confusion matrix and standard binary classification metrics."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    valid = np.isfinite(pred.astype(float)) & np.isfinite(truth.astype(float))
    pred = pred[valid]
    truth = truth[valid]

    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    total = tp + tn + fp + fn

    def div(a: float, b: float) -> Optional[float]:
        return float(a / b) if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    accuracy = div(tp + tn, total)
    balanced = (
        float((recall + specificity) / 2)
        if recall is not None and specificity is not None
        else None
    )
    false_alarm_rate = div(fp, fp + tn)
    miss_rate = div(fn, fn + tp)

    return {
        "samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


def compute_qdc_baseline(x: np.ndarray, cadence_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate the production rolling QDC construction used by the demo."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    baseline = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)

    window_samples = max(1, int(24 * 3600 / cadence_s))
    step_samples = max(1, window_samples // 2)
    t_global = np.arange(n, dtype=float) * cadence_s / 3600.0
    t_min = float(t_global.min()) if n else 0.0
    t_max = float(t_global.max()) if n else 0.0
    last_good_coeffs = None

    for start in range(0, max(1, n - step_samples), step_samples):
        end = min(start + window_samples, n)
        if end - start < step_samples // 2:
            break

        segment = x[start:end]
        t_seg = t_global[start:end]
        if np.isfinite(segment).sum() < (end - start) * 0.5:
            continue

        seg_base, coeffs = robust_harmonic_baseline(
            segment,
            cadence_s,
            t_hours=t_seg,
            t_ref_min=t_min,
            t_ref_max=t_max,
        )

        seg_res = segment - seg_base
        storm_frac = float(np.mean(np.abs(seg_res) > 50.0))
        if storm_frac > 0.05 and last_good_coeffs is not None:
            seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good_coeffs
        elif storm_frac <= 0.05:
            last_good_coeffs = coeffs

        w_win = np.hanning(end - start)
        baseline[start:end] += seg_base * w_win
        weights[start:end] += w_win

    mask = weights > 0
    baseline[mask] /= weights[mask]
    finite_x = finite_values(x)
    fallback = float(np.nanmedian(finite_x)) if finite_x.size else 0.0
    baseline[~mask] = fallback
    residual = x - baseline
    return baseline, residual


def prepare_global_reference(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    n: int,
    cadence_s: float,
) -> Tuple[pd.Series, pd.Series]:
    """Fetch and align Kp and Dst to the magnetometer cadence."""
    kp = pd.Series(dtype=float)
    dst = pd.Series(dtype=float)

    try:
        kp = fetch_kp_gfz(start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
    except Exception as exc:  # benchmark should remain useful when one source is down
        print(f"[WARN] Kp unavailable: {exc}")

    months = pd.period_range(start=start_time.strftime("%Y-%m"), end=end_time.strftime("%Y-%m"), freq="M")
    dst_parts = []
    for period in months:
        series = fetch_dst_kyoto(int(period.year), int(period.month))
        if series is not None and not series.empty:
            dst_parts.append(series)
    if dst_parts:
        dst = pd.concat(dst_parts).sort_index()

    target_index = pd.date_range(start=start_time, periods=n, freq=pd.Timedelta(seconds=cadence_s), tz="UTC")
    kp_aligned = kp.reindex(target_index, method="ffill", tolerance=pd.Timedelta("3h")) if not kp.empty else pd.Series(np.nan, index=target_index)
    dst_aligned = dst.reindex(target_index, method="ffill", tolerance=pd.Timedelta("3h")) if not dst.empty else pd.Series(np.nan, index=target_index)
    return kp_aligned, dst_aligned


def activity_reference(kp: pd.Series, dst: pd.Series) -> Dict[str, np.ndarray]:
    kp_values = kp.to_numpy(dtype=float)
    dst_values = dst.to_numpy(dtype=float)
    has_global = np.isfinite(kp_values) | np.isfinite(dst_values)

    active = ((np.isfinite(kp_values) & (kp_values >= 4.0)) |
              (np.isfinite(dst_values) & (dst_values < -30.0))) & has_global
    storm = ((np.isfinite(kp_values) & (kp_values >= 6.0)) |
             (np.isfinite(dst_values) & (dst_values < -50.0))) & has_global
    return {"has_global": has_global, "active": active, "storm": storm}


def daily_rms(residual: np.ndarray, index: pd.DatetimeIndex) -> Dict[str, float]:
    series = pd.Series(residual, index=index).dropna()
    if series.empty:
        return {}
    values = series.groupby(series.index.floor("D")).apply(lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy())))))
    return {str(day.date()): round(float(value), 6) for day, value in values.items()}


def benchmark(
    observatory: str,
    start_date: str,
    days: int,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_start = time.perf_counter()
    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
        samples_per_day="Minute",
    )
    fetch_seconds = time.perf_counter() - fetch_start

    parse_start = time.perf_counter()
    df = parse_iaga2002_to_dataframe(raw)
    parse_seconds = time.perf_counter() - parse_start
    if df.empty:
        raise RuntimeError("INTERMAGNET returned no usable samples.")

    # F is the total-field series and is the same quantity used by the demo.
    field = "f_nt"
    series = pd.to_numeric(df[field], errors="coerce")
    valid = series.notna()
    valid_series = series.loc[valid]
    if len(valid_series) < 24:
        raise RuntimeError(f"Only {len(valid_series)} valid {field} samples available; need at least 24.")

    index = series.index
    expected = max(1, int(days * 24 * 60))
    valid_count = int(valid.sum())
    completeness = valid_count / expected
    cadence_seconds = float(index.to_series().diff().dropna().dt.total_seconds().median())

    analysis_start = time.perf_counter()
    baseline, residual = compute_qdc_baseline(series.to_numpy(dtype=float), cadence_seconds)
    analysis_seconds = time.perf_counter() - analysis_start

    finite_resid = finite_values(residual)
    abs_resid = np.abs(finite_resid)
    rms = float(np.sqrt(np.mean(finite_resid ** 2)))
    mae = float(np.mean(abs_resid))
    median_abs = float(np.median(abs_resid))
    p95_abs = float(np.percentile(abs_resid, 95))
    bias = float(np.mean(finite_resid))
    residual_std = float(np.std(finite_resid))

    quiet_mask = np.isfinite(residual) & (np.abs(residual) <= QUIET_LIMIT_NT)
    quiet_resid = residual[quiet_mask]
    quiet_rms = float(np.sqrt(np.mean(quiet_resid ** 2))) if quiet_resid.size else None
    quiet_mae = float(np.mean(np.abs(quiet_resid))) if quiet_resid.size else None

    predicted_active = np.isfinite(residual) & (np.abs(residual) > ACTIVE_LIMIT_NT)
    predicted_storm = np.isfinite(residual) & (np.abs(residual) > MINOR_STORM_LIMIT_NT)
    anomaly = np.isfinite(residual) & (np.abs(np.diff(residual, prepend=residual[0])) > 100.0)

    storm_flags = np.full(len(residual), "quiet", dtype=object)
    magnitude = np.abs(residual)
    storm_flags[magnitude > QUIET_LIMIT_NT] = "unsettled"
    storm_flags[magnitude > ACTIVE_LIMIT_NT] = "active"
    storm_flags[magnitude > MINOR_STORM_LIMIT_NT] = "minor_storm"
    storm_flags[magnitude > MAJOR_STORM_LIMIT_NT] = "major_storm"
    storm_flags[magnitude > SEVERE_STORM_LIMIT_NT] = "severe_storm"
    storm_flags[anomaly] = "anomaly"

    global_fetch_start = time.perf_counter()
    kp_aligned, dst_aligned = prepare_global_reference(
        index[0], index[-1], len(index), int(round(cadence_seconds))
    )
    global_fetch_seconds = time.perf_counter() - global_fetch_start
    refs = activity_reference(kp_aligned, dst_aligned)
    has_global = refs["has_global"] & np.isfinite(residual)

    active_metrics = None
    storm_metrics = None
    if int(has_global.sum()) > 0:
        active_metrics = binary_metrics(predicted_active[has_global], refs["active"][has_global])
        storm_metrics = binary_metrics(predicted_storm[has_global], refs["storm"][has_global])

    total_runtime = fetch_seconds + parse_seconds + analysis_seconds + global_fetch_seconds
    throughput = valid_count / analysis_seconds if analysis_seconds > 0 else None

    daily = daily_rms(residual, index)
    daily_values = list(daily.values())

    report: Dict[str, Any] = {
        "benchmark": {
            "observatory": observatory,
            "start_date": start_date,
            "days": days,
            "field": field,
            "samples": len(series),
            "valid_samples": valid_count,
            "expected_samples": expected,
            "completeness": safe_float(completeness),
            "cadence_seconds": safe_float(cadence_seconds),
        },
        "baseline_quality_nt": {
            "bias": safe_float(bias),
            "mae": safe_float(mae),
            "rmse": safe_float(rms),
            "rms": safe_float(rms),
            "residual_std": safe_float(residual_std),
            "median_absolute_error": safe_float(median_abs),
            "p95_absolute_error": safe_float(p95_abs),
            "quiet_rms": safe_float(quiet_rms),
            "quiet_mae": safe_float(quiet_mae),
        },
        "activity": {
            "quiet_fraction": safe_float(float(np.mean(storm_flags == "quiet"))),
            "unsettled_fraction": safe_float(float(np.mean(storm_flags == "unsettled"))),
            "active_or_higher_fraction": safe_float(float(np.mean(predicted_active))),
            "storm_or_higher_fraction": safe_float(float(np.mean(predicted_storm))),
            "anomaly_fraction": safe_float(float(np.mean(anomaly))),
            "counts": {
                "quiet": int(np.sum(storm_flags == "quiet")),
                "unsettled": int(np.sum(storm_flags == "unsettled")),
                "active": int(np.sum(storm_flags == "active")),
                "minor_storm": int(np.sum(storm_flags == "minor_storm")),
                "major_storm": int(np.sum(storm_flags == "major_storm")),
                "severe_storm": int(np.sum(storm_flags == "severe_storm")),
                "anomaly": int(np.sum(storm_flags == "anomaly")),
            },
        },
        "global_validation": {
            "reference_samples": int(has_global.sum()),
            "kp_available_samples": int(np.isfinite(kp_aligned.to_numpy()).sum()),
            "dst_available_samples": int(np.isfinite(dst_aligned.to_numpy()).sum()),
            "active_event_metrics": active_metrics,
            "storm_event_metrics": storm_metrics,
        },
        "performance": {
            "fetch_seconds": safe_float(fetch_seconds),
            "parse_seconds": safe_float(parse_seconds),
            "analysis_seconds": safe_float(analysis_seconds),
            "global_index_seconds": safe_float(global_fetch_seconds),
            "total_seconds": safe_float(total_runtime),
            "analysis_samples_per_second": safe_float(throughput),
            "analysis_samples_per_second_of_realtime": safe_float(
                throughput / (1.0 / cadence_seconds) if throughput is not None and cadence_seconds > 0 else None
            ),
        },
        "daily_rms_nt": daily,
        "daily_rms_summary_nt": {
            "mean": safe_float(float(np.mean(daily_values))) if daily_values else None,
            "std": safe_float(float(np.std(daily_values))) if daily_values else None,
            "min": safe_float(float(np.min(daily_values))) if daily_values else None,
            "max": safe_float(float(np.max(daily_values))) if daily_values else None,
        },
    }

    json_path = output_dir / f"magnetometer_performance_{observatory}_{start_date}_{days}d.json"
    json_path.write_text(json.dumps(report, indent=2))

    rows = [
        ("completeness", report["benchmark"]["completeness"]),
        ("bias_nt", report["baseline_quality_nt"]["bias"]),
        ("mae_nt", report["baseline_quality_nt"]["mae"]),
        ("rmse_nt", report["baseline_quality_nt"]["rmse"]),
        ("median_abs_error_nt", report["baseline_quality_nt"]["median_absolute_error"]),
        ("p95_abs_error_nt", report["baseline_quality_nt"]["p95_absolute_error"]),
        ("quiet_rms_nt", report["baseline_quality_nt"]["quiet_rms"]),
        ("quiet_mae_nt", report["baseline_quality_nt"]["quiet_mae"]),
        ("active_fraction", report["activity"]["active_or_higher_fraction"]),
        ("storm_fraction", report["activity"]["storm_or_higher_fraction"]),
        ("anomaly_fraction", report["activity"]["anomaly_fraction"]),
        ("analysis_seconds", report["performance"]["analysis_seconds"]),
        ("analysis_samples_per_second", report["performance"]["analysis_samples_per_second"]),
    ]
    csv_path = output_dir / f"magnetometer_performance_{observatory}_{start_date}_{days}d.csv"
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(csv_path, index=False)

    print("\n" + "=" * 72)
    print("MAGNETOMETER PERFORMANCE REPORT")
    print("=" * 72)
    print(f"Observatory:              {observatory}")
    print(f"Period:                   {start_date} ({days} days)")
    print(f"Samples:                  {len(series):,}")
    print(f"Valid samples:            {valid_count:,} ({completeness * 100:.2f}%)")
    print(f"Cadence:                  {cadence_seconds:.1f} s")
    print("-" * 72)
    print(f"Residual MAE:             {mae:.3f} nT")
    print(f"Residual RMSE/RMS:        {rms:.3f} nT")
    print(f"Residual bias:            {bias:.3f} nT")
    print(f"Residual median |e|:      {median_abs:.3f} nT")
    print(f"Residual P95 |e|:         {p95_abs:.3f} nT")
    print(f"Quiet-period RMS:         {quiet_rms:.3f} nT" if quiet_rms is not None else "Quiet-period RMS:         N/A")
    print(f"Quiet-period MAE:         {quiet_mae:.3f} nT" if quiet_mae is not None else "Quiet-period MAE:         N/A")
    print("-" * 72)
    print(f"Detected active+ fraction:{float(np.mean(predicted_active)) * 100:.3f}%")
    print(f"Detected storm+ fraction: {float(np.mean(predicted_storm)) * 100:.3f}%")
    print(f"Anomaly fraction:         {float(np.mean(anomaly)) * 100:.3f}%")

    if active_metrics:
        print(f"Active precision:         {active_metrics['precision']:.3f}" if active_metrics['precision'] is not None else "Active precision:         N/A")
        print(f"Active recall:            {active_metrics['recall']:.3f}" if active_metrics['recall'] is not None else "Active recall:            N/A")
        print(f"Active F1:                {active_metrics['f1']:.3f}" if active_metrics['f1'] is not None else "Active F1:                N/A")
        print(f"Active false alarm rate:  {active_metrics['false_alarm_rate']:.3f}" if active_metrics['false_alarm_rate'] is not None else "Active false alarm rate:  N/A")
    else:
        print("Global active-event scores: N/A (no Kp/Dst coverage)")

    if storm_metrics:
        print(f"Storm precision:          {storm_metrics['precision']:.3f}" if storm_metrics['precision'] is not None else "Storm precision:          N/A")
        print(f"Storm recall:             {storm_metrics['recall']:.3f}" if storm_metrics['recall'] is not None else "Storm recall:             N/A")
        print(f"Storm F1:                 {storm_metrics['f1']:.3f}" if storm_metrics['f1'] is not None else "Storm F1:                 N/A")
        print(f"Storm false alarm rate:   {storm_metrics['false_alarm_rate']:.3f}" if storm_metrics['false_alarm_rate'] is not None else "Storm false alarm rate:   N/A")
    else:
        print("Global storm-event scores: N/A (no Kp/Dst coverage)")

    print("-" * 72)
    print(f"Analysis time:            {analysis_seconds:.3f} s")
    print(f"Analysis throughput:      {throughput:,.0f} samples/s" if throughput is not None else "Analysis throughput:      N/A")
    print(f"Total benchmark time:     {total_runtime:.3f} s")
    print(f"JSON report:              {json_path}")
    print(f"CSV summary:              {csv_path}")
    print("=" * 72)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the magnetometer QDC/activity pipeline.")
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "magnetometer" / "data"),
        help="Directory for JSON/CSV benchmark reports.",
    )
    args = parser.parse_args()

    benchmark(
        observatory=args.observatory,
        start_date=args.start_date,
        days=args.days,
        output_dir=Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    main()
