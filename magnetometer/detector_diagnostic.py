#!/usr/bin/env python3
"""Single-window production detector diagnostic.

This tool is for fast detector development and operational inspection.  It uses
exactly the same QDC/residual path, reference masks, persistence rules and
production threshold scoring primitives as the production validation suite,
but evaluates one explicit time window instead of a multi-year case suite.

Important: Kp/Dst-derived labels are *reference proxies*, not local ground
truth.  Metrics are therefore reported as ``reference-comparison`` metrics and
must not be confused with the strict held-out certification result.

Examples
--------
10-day historical window::

    python magnetometer/detector_diagnostic.py \
        --observatory VIC \
        --start-date 2025-01-01 \
        --days 10 \
        --fetch-real-data

Latest available window::

    python magnetometer/detector_diagnostic.py \
        --observatory VIC \
        --days 10 \
        --latest \
        --fetch-real-data

The command prints a compact operational report and writes a complete JSON
report containing sample-level metrics, event-level metrics, coverage,
residual quality, class counts, transitions and individual detected/reference
events.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from magnetometer_demo import (  # noqa: E402
    ANOMALY_DELTA_NT,
    PROD_ACTIVE_NT,
    PROD_MAJOR_STORM_NT,
    PROD_MINOR_STORM_NT,
    PROD_SEVERE_STORM_NT,
    PROD_UNSETTLED_NT,
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    flag_activity,
    parse_iaga2002_to_dataframe,
)
import performance_metrics as pm  # noqa: E402

logger = logging.getLogger("detector_diagnostic")

DEFAULT_OUTPUT_DIR = HERE / "data"
CLASS_ORDER = (
    "quiet",
    "unsettled",
    "active",
    "minor_storm",
    "major_storm",
    "severe_storm",
    "anomaly",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:.{digits}f}%"


def _event_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _merge_events(
    events: Sequence[Tuple[int, int]],
    max_gap_samples: int,
    min_duration_samples: int = 1,
) -> List[Tuple[int, int]]:
    if not events:
        return []
    merged: List[List[int]] = []
    for start, end in events:
        if end - start < min_duration_samples:
            continue
        if merged and start - merged[-1][1] <= max_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _event_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _match_events(
    predicted: Sequence[Tuple[int, int]],
    reference: Sequence[Tuple[int, int]],
    cadence_s: float,
) -> Dict[str, Any]:
    candidates: List[Tuple[int, int, int]] = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            overlap = _event_overlap(pred, ref)
            if overlap > 0:
                candidates.append((overlap, pi, ri))
    candidates.sort(reverse=True)

    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches: List[Dict[str, Any]] = []
    for overlap, pi, ri in candidates:
        if pi in used_pred or ri in used_ref:
            continue
        used_pred.add(pi)
        used_ref.add(ri)
        matches.append(
            {
                "predicted_index": pi,
                "reference_index": ri,
                "overlap_seconds": float(overlap * cadence_s),
            }
        )

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "reference_events": len(reference),
        "predicted_events": len(predicted),
        "matched_events": tp,
        "missed_events": fn,
        "false_positive_events": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def _event_records(
    events: Sequence[Tuple[int, int]],
    index: pd.DatetimeIndex,
    residual: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for number, (start, end) in enumerate(events, start=1):
        values = np.asarray(residual[start:end], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            peak_idx = int(np.argmax(np.abs(finite)))
            peak_abs = float(np.abs(finite[peak_idx]))
            peak_signed = float(finite[np.argmax(np.abs(finite))])
        else:
            peak_abs = None
            peak_signed = None
        label = None
        if labels is not None and end > start:
            vals, counts = np.unique(labels[start:end], return_counts=True)
            label = str(vals[int(np.argmax(counts))]) if len(vals) else None
        records.append(
            {
                "event": number,
                "start": index[start].isoformat(),
                "end": index[min(end - 1, len(index) - 1)].isoformat(),
                "duration_minutes": float((end - start) * (index[1] - index[0]).total_seconds() / 60.0)
                if len(index) > 1
                else None,
                "peak_abs_residual_nt": peak_abs,
                "peak_signed_residual_nt": peak_signed,
                "class": label,
            }
        )
    return records


def _reference_series(
    index: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    kp = pd.Series(dtype=float)
    dst_parts: List[pd.Series] = []
    errors: List[str] = []

    try:
        kp = fetch_kp_gfz(start_date, end_date)
    except Exception as exc:  # network/reference failures are reported, not fatal to detector-only metrics
        errors.append(f"Kp: {exc}")

    periods = pd.period_range(
        pd.Timestamp(index[0]).strftime("%Y-%m"),
        pd.Timestamp(index[-1]).strftime("%Y-%m"),
        freq="M",
    )
    for period in periods:
        try:
            dst = fetch_dst_kyoto(int(period.year), int(period.month))
        except Exception as exc:
            errors.append(f"Dst {period}: {exc}")
            dst = None
        if dst is not None and not dst.empty:
            dst_parts.append(dst)

    dst = pd.concat(dst_parts).sort_index() if dst_parts else pd.Series(dtype=float)
    target = pd.DatetimeIndex(index)
    tolerance = pd.Timedelta("3h")
    kp_aligned = (
        kp.reindex(target, method="ffill", tolerance=tolerance)
        if not kp.empty
        else pd.Series(np.nan, index=target)
    )
    dst_aligned = (
        dst.reindex(target, method="ffill", tolerance=tolerance)
        if not dst.empty
        else pd.Series(np.nan, index=target)
    )
    refs = pm.reference_masks(kp_aligned, dst_aligned)
    return kp_aligned, dst_aligned, {
        "kp_coverage": float(refs["kp_known"].mean()),
        "dst_coverage": float(refs["dst_known"].mean()),
        "overall_coverage": float(refs["known"].mean()),
        "errors": errors,
        "refs": refs,
    }


def _class_metrics(predicted: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    """One-vs-rest metrics for every reference class represented by Kp/Dst."""
    rows: Dict[str, Any] = {}
    for cls, mask in reference.items():
        rows[cls] = pm.binary_metrics(predicted == cls, mask)
    return rows


def _reference_class_labels(kp: pd.Series, dst: pd.Series) -> np.ndarray:
    """Build mutually exclusive conservative reference classes from Kp/Dst.

    This is explicitly a proxy label. Storm takes precedence, then active,
    then quiet. Samples with neither reference index are marked ``unknown``.
    """
    kp_v = kp.to_numpy(dtype=float)
    dst_v = dst.to_numpy(dtype=float)
    known = np.isfinite(kp_v) | np.isfinite(dst_v)
    storm = (np.isfinite(kp_v) & (kp_v >= 6.0)) | (np.isfinite(dst_v) & (dst_v < -50.0))
    active = (np.isfinite(kp_v) & (kp_v >= 4.0)) | (np.isfinite(dst_v) & (dst_v < -30.0))
    labels = np.full(len(kp), "unknown", dtype=object)
    labels[known & ~active] = "quiet"
    labels[known & active & ~storm] = "active"
    labels[known & storm] = "storm"
    return labels


def evaluate(
    observatory: str,
    start_date: str,
    days: int,
    active_threshold: float,
    storm_threshold: float,
    samples_per_day: str = "Minute",
) -> Dict[str, Any]:
    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
        samples_per_day=samples_per_day,
    )
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty or "f_nt" not in df.columns:
        raise RuntimeError("INTERMAGNET returned no usable total-field data.")

    series = pd.to_numeric(df["f_nt"], errors="coerce")
    index = series.index
    valid = series.notna().to_numpy()
    if valid.sum() < 10:
        raise RuntimeError("Too few valid magnetometer samples for evaluation.")

    deltas = index.to_series().diff().dropna().dt.total_seconds()
    cadence_s = float(deltas.median()) if not deltas.empty else 60.0
    if not math.isfinite(cadence_s) or cadence_s <= 0:
        raise RuntimeError(f"Invalid inferred cadence: {cadence_s!r}")

    expected_samples = max(1, int(round(days * 86400.0 / cadence_s)))
    completeness = float(valid.sum() / expected_samples)

    # Use the exact production QDC/residual implementation.
    from performance_metrics import compute_qdc_baseline

    baseline, residual = compute_qdc_baseline(series.to_numpy(dtype=float), cadence_s)
    predicted_flags = flag_activity(residual, cadence_s)

    start = pd.Timestamp(index[0]).strftime("%Y-%m-%d")
    end = pd.Timestamp(index[-1]).strftime("%Y-%m-%d")
    kp, dst, reference_info = _reference_series(index, start, end)
    refs = reference_info["refs"]
    reference_labels = _reference_class_labels(kp, dst)
    known = refs["known"] & np.isfinite(residual)

    # Use the same production scoring primitive for exact threshold/persistence
    # behavior and independently expose its event matching details.
    threshold_score = pm.score_thresholds(
        residual,
        refs,
        cadence_s,
        float(active_threshold),
        float(storm_threshold),
    )

    active_pred, storm_pred = pm.production_detection_masks(
        residual,
        cadence_s,
        float(active_threshold),
        float(storm_threshold),
    )
    reference_active = refs["active"]
    reference_storm = refs["storm"]

    # Event extraction intentionally matches the production scorer's semantics.
    predicted_active_events = pm.bool_events(active_pred, cadence_s, 1800, 300)
    predicted_storm_events = pm.bool_events(storm_pred, cadence_s, 1800, 300)
    reference_active_events = pm.bool_events(reference_active & refs["known"], cadence_s, 21600, 10800)
    reference_storm_events = pm.bool_events(reference_storm & refs["known"], cadence_s, 21600, 10800)

    event_active = _match_events(predicted_active_events, reference_active_events, cadence_s)
    event_storm = _match_events(predicted_storm_events, reference_storm_events, cadence_s)

    # Residual quality excludes missing values.
    finite_residual = residual[np.isfinite(residual)]
    abs_residual = np.abs(finite_residual)
    derivative = np.diff(residual, prepend=residual[0])
    finite_derivative = derivative[np.isfinite(derivative)]

    counts = {cls: int(np.sum(predicted_flags == cls)) for cls in CLASS_ORDER}
    counts["unknown"] = int(np.sum(~np.isin(predicted_flags, CLASS_ORDER)))

    transitions: List[Dict[str, Any]] = []
    for i in range(1, len(predicted_flags)):
        if predicted_flags[i] != predicted_flags[i - 1]:
            transitions.append(
                {
                    "time": index[i].isoformat(),
                    "from": str(predicted_flags[i - 1]),
                    "to": str(predicted_flags[i]),
                }
            )

    # Per-class reference comparison. This is useful diagnostically but is not
    # the certification gate's event taxonomy.
    ref_class_metrics: Dict[str, Any] = {}
    for cls in ("quiet", "active", "storm"):
        truth = reference_labels == cls
        ref_class_metrics[cls] = pm.binary_metrics(
            (predicted_flags == cls) & known,
            truth & known,
        )

    # Direct detector-vs-reference masks use the production scoring path.
    report: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "detector_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_warning": (
            "Kp/Dst thresholds are reference proxies, not local geomagnetic ground truth. "
            "These metrics are diagnostic and do not certify the detector."
        ),
        "window": {
            "observatory": observatory,
            "requested_start_date": start_date,
            "requested_days": int(days),
            "actual_start": index[0].isoformat(),
            "actual_end": index[-1].isoformat(),
            "samples": int(len(series)),
            "valid_samples": int(valid.sum()),
            "cadence_seconds": cadence_s,
            "completeness": completeness,
        },
        "thresholds": {
            "active_nt": float(active_threshold),
            "storm_nt": float(storm_threshold),
            "unsettled_nt": PROD_UNSETTLED_NT,
            "major_storm_nt": PROD_MAJOR_STORM_NT,
            "severe_storm_nt": PROD_SEVERE_STORM_NT,
            "anomaly_delta_nt": ANOMALY_DELTA_NT,
        },
        "residual_quality": {
            "mae_nt": float(np.mean(abs_residual)) if abs_residual.size else None,
            "rmse_nt": float(np.sqrt(np.mean(finite_residual ** 2))) if finite_residual.size else None,
            "mean_nt": float(np.mean(finite_residual)) if finite_residual.size else None,
            "median_nt": float(np.median(finite_residual)) if finite_residual.size else None,
            "std_nt": float(np.std(finite_residual)) if finite_residual.size else None,
            "p95_abs_nt": float(np.percentile(abs_residual, 95)) if abs_residual.size else None,
            "p99_abs_nt": float(np.percentile(abs_residual, 99)) if abs_residual.size else None,
            "max_abs_nt": float(np.max(abs_residual)) if abs_residual.size else None,
            "min_nt": float(np.min(finite_residual)) if finite_residual.size else None,
            "max_nt": float(np.max(finite_residual)) if finite_residual.size else None,
            "max_abs_step_nt": float(np.max(np.abs(finite_derivative))) if finite_derivative.size else None,
        },
        "reference": {
            "kp_coverage": reference_info["kp_coverage"],
            "dst_coverage": reference_info["dst_coverage"],
            "overall_coverage": reference_info["overall_coverage"],
            "errors": reference_info["errors"],
            "class_counts": {
                "quiet": int(np.sum(reference_labels == "quiet")),
                "active": int(np.sum(reference_labels == "active")),
                "storm": int(np.sum(reference_labels == "storm")),
                "unknown": int(np.sum(reference_labels == "unknown")),
            },
        },
        "detector": {
            "class_counts": counts,
            "transitions": transitions,
            "transition_count": len(transitions),
            "active_events": len(predicted_active_events),
            "storm_events": len(predicted_storm_events),
        },
        "reference_comparison": {
            "sample_level": {
                "active": threshold_score["active"]["sample_level"],
                "storm": threshold_score["storm"]["sample_level"],
            },
            "event_level": {
                "active": event_active,
                "storm": event_storm,
            },
            "mutually_exclusive_proxy_classes": ref_class_metrics,
        },
        "production_scoring": threshold_score,
        "events": {
            "detected_active": _event_records(predicted_active_events, index, residual, predicted_flags),
            "detected_storm": _event_records(predicted_storm_events, index, residual, predicted_flags),
            "reference_active": _event_records(reference_active_events, index, residual, reference_labels),
            "reference_storm": _event_records(reference_storm_events, index, residual, reference_labels),
        },
    }
    return _json_safe(report)


def print_report(report: Dict[str, Any]) -> None:
    window = report["window"]
    ref = report["reference"]
    detector = report["detector"]
    comparison = report["reference_comparison"]
    residual = report["residual_quality"]
    storm = comparison["sample_level"]["storm"]
    active = comparison["sample_level"]["active"]
    storm_event = comparison["event_level"]["storm"]
    active_event = comparison["event_level"]["active"]

    print("\n" + "=" * 92)
    print("MAGNETOMETER SINGLE-WINDOW DETECTOR DIAGNOSTIC")
    print("=" * 92)
    print(f"Observatory:       {window['observatory']}")
    print(f"Window:            {window['actual_start']}  ->  {window['actual_end']}")
    print(f"Samples:           {window['samples']:,} ({window['cadence_seconds']:.1f}s cadence)")
    print(f"Completeness:      {_pct(window['completeness'])}")
    print(f"Reference coverage:{_pct(ref['overall_coverage'])}")
    print("-" * 92)
    print("RESIDUAL QUALITY")
    print(f"MAE:               {_fmt(residual['mae_nt'])} nT")
    print(f"RMSE:              {_fmt(residual['rmse_nt'])} nT")
    print(f"P95 |residual|:    {_fmt(residual['p95_abs_nt'])} nT")
    print(f"P99 |residual|:    {_fmt(residual['p99_abs_nt'])} nT")
    print(f"Max |residual|:    {_fmt(residual['max_abs_nt'])} nT")
    print(f"Max |step|:        {_fmt(residual['max_abs_step_nt'])} nT/sample")
    print("-" * 92)
    print("REFERENCE COMPARISON — Kp/Dst PROXY, NOT GROUND TRUTH")
    print("                         Precision     Recall        F1          FAR")
    print(
        f"Active                 {_fmt(active['precision'], 3):>9}    {_fmt(active['recall'], 3):>9}"
        f"    {_fmt(active['f1'], 3):>9}    {_fmt(active['false_alarm_rate'], 3):>9}"
    )
    print(
        f"Storm                  {_fmt(storm['precision'], 3):>9}    {_fmt(storm['recall'], 3):>9}"
        f"    {_fmt(storm['f1'], 3):>9}    {_fmt(storm['false_alarm_rate'], 3):>9}"
    )
    print("-" * 92)
    print("EVENT-LEVEL REFERENCE COMPARISON")
    print(
        f"Active:  precision={_fmt(active_event['precision'])}  recall={_fmt(active_event['recall'])}"
        f"  F1={_fmt(active_event['f1'])}  predicted={active_event['predicted_events']}"
        f"  reference={active_event['reference_events']}"
    )
    print(
        f"Storm:   precision={_fmt(storm_event['precision'])}  recall={_fmt(storm_event['recall'])}"
        f"  F1={_fmt(storm_event['f1'])}  predicted={storm_event['predicted_events']}"
        f"  reference={storm_event['reference_events']}"
    )
    print("-" * 92)
    print("DETECTOR OUTPUT")
    for cls in CLASS_ORDER:
        print(f"{cls:<18}: {detector['class_counts'].get(cls, 0):>8,}")
    print(f"Transitions:        {detector['transition_count']}")
    print(f"Active events:      {detector['active_events']}")
    print(f"Storm events:       {detector['storm_events']}")
    print("-" * 92)
    print("REFERENCE COVERAGE")
    print(f"Kp:                {_pct(ref['kp_coverage'])}")
    print(f"Dst:               {_pct(ref['dst_coverage'])}")
    print(f"Combined:          {_pct(ref['overall_coverage'])}")
    if ref["errors"]:
        print("Reference warnings:")
        for error in ref["errors"]:
            print(f"  - {error}")
    print("=" * 92)
    print("NOTE: This diagnostic does NOT certify a release. Use production_release_gate.py")
    print("      for the strict held-out certification decision.")
    print("=" * 92)


def _resolve_start_date(args: argparse.Namespace) -> str:
    if args.latest and args.start_date:
        raise SystemExit("Use either --start-date or --latest, not both.")
    if args.latest:
        # Start slightly before the requested window so the end is near the
        # latest UTC date. INTERMAGNET may return only completed days.
        return (datetime.now(timezone.utc).date() - timedelta(days=args.days)).isoformat()
    if not args.start_date:
        raise SystemExit("One of --start-date or --latest is required.")
    try:
        parsed = pd.Timestamp(args.start_date)
    except Exception as exc:
        raise SystemExit(f"Invalid --start-date: {exc}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC")
    return parsed.strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast, production-path detector diagnostic for one historical or latest window."
    )
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", help="UTC start date, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--latest", action="store_true", help="Evaluate a window ending near the latest UTC date")
    parser.add_argument(
        "--fetch-real-data",
        action="store_true",
        help="Required safety switch before network access is performed",
    )
    parser.add_argument("--active-threshold", type=float, default=PROD_ACTIVE_NT)
    parser.add_argument("--storm-threshold", type=float, default=PROD_MINOR_STORM_NT)
    parser.add_argument("--samples-per-day", default="Minute", choices=("Minute", "Hour"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-json", action="store_true", help="Do not write the JSON diagnostic report")
    args = parser.parse_args()

    if args.days < 1 or args.days > 31:
        parser.error("--days must be between 1 and 31")
    if args.active_threshold <= 0 or args.storm_threshold <= 0:
        parser.error("thresholds must be positive")
    if args.storm_threshold <= args.active_threshold:
        parser.error("--storm-threshold must be greater than --active-threshold")
    if not args.fetch_real_data:
        parser.error("Must supply --fetch-real-data before accessing INTERMAGNET/Kp/Dst")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    start_date = _resolve_start_date(args)
    observatory = args.observatory.strip().upper()
    report = evaluate(
        observatory=observatory,
        start_date=start_date,
        days=args.days,
        active_threshold=args.active_threshold,
        storm_threshold=args.storm_threshold,
        samples_per_day=args.samples_per_day,
    )
    print_report(report)

    if not args.no_json:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        actual_start = pd.Timestamp(report["window"]["actual_start"]).strftime("%Y%m%d")
        path = output_dir / f"detector_diagnostic_{observatory}_{actual_start}_{args.days}d.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
