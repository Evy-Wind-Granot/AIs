#!/usr/bin/env python3
"""Multi-year, multi-observatory training and certification for geomagnetic forecasting."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent / "magnetometer"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from forecast_release_gate import evaluate_forecast_release  # noqa: E402
from magnetometer_demo import (  # noqa: E402
    build_design_matrix,
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    handle_gaps,
    parse_iaga2002_to_dataframe,
    robust_harmonic_baseline,
)
from models.forecaster import ForecastConfig, GeomagneticForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magnetometer_forecaster_production")


def fetch_chunked(observatory: str, start: pd.Timestamp, days: int, chunk_days: int) -> pd.DataFrame:
    if days <= 0 or chunk_days <= 0:
        raise ValueError("days and chunk_days must be positive")
    parts: List[pd.DataFrame] = []
    cursor = start
    remaining = days
    while remaining:
        duration = min(chunk_days, remaining)
        logger.info("Fetching INTERMAGNET chunk: %s (%d days)", cursor.strftime("%Y-%m-%d"), duration)
        raw = fetch_intermagnet_iaga2002(
            observatory=observatory,
            start_date=cursor.strftime("%Y-%m-%d"),
            duration_days=duration,
            samples_per_day="Minute",
        )
        frame = parse_iaga2002_to_dataframe(raw)
        if frame.empty:
            raise RuntimeError(f"No INTERMAGNET samples for {observatory} {cursor.date()}")
        parts.append(frame)
        cursor += pd.Timedelta(days=duration)
        remaining -= duration
    merged = pd.concat(parts).sort_index()
    return merged[~merged.index.duplicated(keep="first")]


def compute_residual(frame: pd.DataFrame, column: str) -> pd.Series:
    series = handle_gaps(pd.to_numeric(frame[column], errors="coerce"), max_gap_samples=3)
    if int(series.notna().sum()) == 0:
        raise RuntimeError(f"Zero valid samples in {column!r}")
    cadence = frame.index.to_series().diff().dropna().dt.total_seconds().median()
    cadence_s = float(cadence) if np.isfinite(cadence) and cadence > 0 else 60.0
    values = series.to_numpy(dtype=float)
    n = len(values)
    baseline = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)
    window_samples = max(2, int(24 * 3600 / cadence_s))
    step_samples = max(1, window_samples // 2)
    t = np.arange(n, dtype=float) * cadence_s / 3600.0
    t_min, t_max = float(t.min()), float(t.max())
    last_good = None
    for offset in range(0, max(1, n - step_samples), step_samples):
        end = min(offset + window_samples, n)
        if end - offset < max(2, step_samples // 2):
            continue
        segment = values[offset:end]
        if np.isfinite(segment).sum() < 0.5 * len(segment):
            continue
        t_seg = t[offset:end]
        seg_base, coeffs = robust_harmonic_baseline(segment, cadence_s, t_hours=t_seg, t_ref_min=t_min, t_ref_max=t_max)
        seg_res = segment - seg_base
        if float(np.mean(np.abs(seg_res) > 50.0)) > 0.05 and last_good is not None:
            seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good
        else:
            last_good = coeffs
        w = np.hanning(end - offset)
        baseline[offset:end] += seg_base * w
        weights[offset:end] += w
    valid = weights > 0
    finite = values[np.isfinite(values)]
    fallback = float(np.median(finite)) if finite.size else 0.0
    baseline[valid] /= weights[valid]
    baseline[~valid] = fallback
    return pd.Series(values - baseline, index=frame.index, name="residual")


def fetch_dst_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    parts: List[pd.Series] = []
    for period in pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"):
        part = fetch_dst_kyoto(int(period.year), int(period.month))
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return pd.Series(index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    out = pd.concat(parts).sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out[~out.index.duplicated(keep="last")].astype(float)


def prepare_period(observatory: str, start: pd.Timestamp, days: int, column: str, chunk_days: int) -> pd.DataFrame:
    raw = fetch_chunked(observatory, start, days, chunk_days)
    if column not in raw.columns:
        raise ValueError(f"Column {column!r} missing from INTERMAGNET response")
    residual = compute_residual(raw, column)
    frame = pd.DataFrame({"residual": residual})
    kp = fetch_kp_gfz(frame.index[0].strftime("%Y-%m-%d"), frame.index[-1].strftime("%Y-%m-%d"))
    dst = fetch_dst_range(frame.index[0], frame.index[-1])
    tolerance = pd.Timedelta("3h")
    frame["kp"] = kp.reindex(frame.index, method="ffill", tolerance=tolerance) if not kp.empty else np.nan
    frame["dst"] = dst.reindex(frame.index, method="ffill", tolerance=tolerance) if not dst.empty else np.nan
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual"])


def parse_years(value: str) -> List[int]:
    years = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if len(years) < 2:
        raise ValueError("At least two chronological years are required")
    return years


def run_one(observatory: str, years: List[int], args: argparse.Namespace) -> Dict[str, Any]:
    frames: List[pd.DataFrame] = []
    for year in years:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        days = 366 if start.is_leap_year else 365
        logger.info("Preparing %s year %d", observatory, year)
        frames.append(prepare_period(observatory, start, days, args.column, args.chunk_days))
    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    if len(frame) < 2000:
        raise RuntimeError(f"{observatory}: insufficient training samples ({len(frame)})")

    config = ForecastConfig(
        backend=args.backend,
        regression_loss=args.regression_loss,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    model = GeomagneticForecaster(config=config)
    training = model.fit(frame)
    gate = evaluate_forecast_release(training["final_test"])

    model.training_metadata["observatory"] = observatory
    model.training_metadata["training_years"] = years
    model.training_metadata["release_gate"] = gate
    model_path = Path(args.model_root) / observatory / "magnetometer_forecaster"
    model.save_model(model_path)

    report = {
        "observatory": observatory,
        "years": years,
        "samples": int(len(frame)),
        "training": training,
        "release_gate": gate,
        "model_path": str(model_path.resolve()),
    }
    report_path = model_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Production training/certification for geomagnetic ML forecasting")
    ap.add_argument("--observatory", default="VIC,BOU")
    ap.add_argument("--years", default="2022,2023,2024,2025")
    ap.add_argument("--column", default="f_nt")
    ap.add_argument("--chunk-days", type=int, default=14)
    ap.add_argument("--backend", choices=("sklearn", "lightgbm"), default="sklearn")
    ap.add_argument("--regression-loss", choices=("absolute_error", "squared_error"), default="absolute_error")
    ap.add_argument("--model-root", default="magnetometer/data/models")
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        idx = pd.date_range("2025-01-01", periods=12000, freq="min", tz="UTC")
        rng = np.random.default_rng(42)
        residual = rng.normal(0.0, 4.0, len(idx))
        for start, stop, amplitude in ((4000, 5000, 90.0), (9800, 10200, 75.0), (11200, 11600, 100.0)):
            residual[start:stop] += amplitude
        frame = pd.DataFrame({"residual": residual, "kp": 2.0, "dst": -5.0}, index=idx)
        model = GeomagneticForecaster(ForecastConfig(validation_fraction=args.validation_fraction, test_fraction=args.test_fraction, regression_loss=args.regression_loss, max_iter=120, min_samples_leaf=10))
        training = model.fit(frame)
        gate = evaluate_forecast_release(training["final_test"])
        print(json.dumps({"training": training, "release_gate": gate}, indent=2))
        raise SystemExit(0 if gate["passed"] else 2)

    years = parse_years(args.years)
    observatories = [x.strip().upper() for x in args.observatory.split(",") if x.strip()]
    if not observatories:
        raise SystemExit("At least one observatory is required")

    result: Dict[str, Any] = {"release_status": "PASS", "observatories": {}}
    for observatory in observatories:
        report = run_one(observatory, years, args)
        result["observatories"][observatory] = report
        if not report["release_gate"]["passed"]:
            result["release_status"] = "FAIL"
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["release_status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
