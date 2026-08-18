#!/usr/bin/env python3
"""Train and evaluate the hybrid geomagnetic short-horizon forecaster."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent / "magnetometer"
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from feature_engineering import build_supervised_dataset  # noqa: E402
from magnetometer_demo import (  # noqa: E402
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    parse_iaga2002_to_dataframe,
    robust_harmonic_baseline,
)
from models.forecaster import ForecastConfig, GeomagneticForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magnetometer_forecaster_train")


def _fetch_dst_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    parts: List[pd.Series] = []
    months = pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M")
    for period in months:
        part = fetch_dst_kyoto(int(period.year), int(period.month))
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return pd.Series(dtype=float)
    result = pd.concat(parts).sort_index()
    result.index = pd.to_datetime(result.index, utc=True)
    return result.astype(float)


def _fetch_chunked(
    observatory: str,
    start_date: str,
    days: int,
    *,
    chunk_days: int = 14,
) -> pd.DataFrame:
    """Fetch long INTERMAGNET windows in bounded chunks to avoid transfer truncation."""
    if days <= 0:
        raise ValueError("days must be positive")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")

    start = pd.Timestamp(start_date, tz="UTC")
    remaining = int(days)
    chunks: List[pd.DataFrame] = []
    while remaining > 0:
        duration = min(chunk_days, remaining)
        chunk_start = start
        logger.info("Fetching INTERMAGNET chunk: %s (%d days)", chunk_start.strftime("%Y-%m-%d"), duration)
        raw = fetch_intermagnet_iaga2002(
            observatory=observatory,
            start_date=chunk_start.strftime("%Y-%m-%d"),
            duration_days=duration,
            samples_per_day="Minute",
        )
        df = parse_iaga2002_to_dataframe(raw)
        if df.empty:
            raise RuntimeError(
                f"INTERMAGNET returned no samples for chunk starting {chunk_start.date()}."
            )
        chunks.append(df)
        start += pd.Timedelta(days=duration)
        remaining -= duration

    merged = pd.concat(chunks).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


def _prepare_window(observatory: str, start_date: str, days: int, column: str, chunk_days: int) -> pd.DataFrame:
    df = _fetch_chunked(observatory, start_date, days, chunk_days=chunk_days)
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not present in INTERMAGNET response")
    series = pd.to_numeric(df[column], errors="coerce")
    valid = int(series.notna().sum())
    if valid == 0:
        raise RuntimeError(
            f"INTERMAGNET returned {len(series)} timestamps but zero valid samples in {column!r}."
        )

    cadence = df.index.to_series().diff().dropna().dt.total_seconds().median()
    cadence = float(cadence) if np.isfinite(cadence) and cadence > 0 else 60.0
    values = series.to_numpy(dtype=float)

    # Reproduce the production deterministic QDC/Harmonic layer so the ML model
    # is trained on exactly the residual semantics used during inference.
    n = len(values)
    baseline = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)
    window_samples = max(2, int(24 * 3600 / cadence))
    step_samples = max(1, window_samples // 2)
    t_global = np.arange(n, dtype=float) * cadence / 3600.0
    t_min, t_max = t_global.min(), t_global.max()
    last_good = None
    from magnetometer_demo import build_design_matrix
    for offset in range(0, max(1, n - step_samples), step_samples):
        end = min(offset + window_samples, n)
        if end - offset < max(2, step_samples // 2):
            continue
        segment = values[offset:end]
        if np.isfinite(segment).sum() < (end - offset) * 0.5:
            continue
        t_seg = t_global[offset:end]
        seg_base, coeffs = robust_harmonic_baseline(
            segment,
            cadence,
            t_hours=t_seg,
            t_ref_min=t_min,
            t_ref_max=t_max,
        )
        seg_res = segment - seg_base
        storm_frac = np.mean(np.abs(seg_res) > 50.0)
        if storm_frac > 0.05 and last_good is not None:
            seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good
        elif storm_frac <= 0.05:
            last_good = coeffs
        w = np.hanning(end - offset)
        baseline[offset:end] += seg_base * w
        weights[offset:end] += w

    mask = weights > 0
    finite_values = values[np.isfinite(values)]
    fallback = float(np.median(finite_values)) if finite_values.size else 0.0
    baseline[mask] /= weights[mask]
    baseline[~mask] = fallback
    residual = values - baseline

    frame = pd.DataFrame(index=df.index)
    frame["residual"] = residual
    kp = fetch_kp_gfz(df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
    dst = _fetch_dst_range(df.index[0], df.index[-1])
    tolerance = pd.Timedelta("3h")
    frame["kp"] = kp.reindex(frame.index, method="ffill", tolerance=tolerance) if not kp.empty else np.nan
    frame["dst"] = dst.reindex(frame.index, method="ffill", tolerance=tolerance) if not dst.empty else np.nan
    return frame


def _evaluate_validation_tail(model: GeomagneticForecaster, frame: pd.DataFrame, cadence_s: float) -> dict:
    """Evaluate only the chronological validation tail, retaining causal context."""
    features, targets = build_supervised_dataset(
        frame,
        cadence_s=cadence_s,
        windows_minutes=model.config.windows_minutes,
        lags_minutes=model.config.lags_minutes,
        horizons_hours=model.config.horizons_hours,
        storm_threshold_nt=model.config.storm_threshold_nt,
    )
    validation_start = pd.Timestamp(model.training_metadata["validation_start"])
    evaluation_mask = features.index >= validation_start
    evaluation_frame = frame.loc[frame.index >= validation_start]
    from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
    report = {"horizons": {}}
    x = model._sanitize_features(features.loc[:, model.feature_names])
    for horizon in model.config.horizons_hours:
        peak_col = f"target_peak_abs_{horizon}h"
        storm_col = f"target_storm_{horizon}h"
        mask = evaluation_mask & targets[peak_col].notna() & targets[storm_col].notna()
        if int(mask.sum()) == 0:
            raise ValueError(f"No validation targets available for {horizon}h horizon.")
        pred_peak = np.clip(model.regressors[int(horizon)].predict(x.loc[mask]), 0.0, None)
        pred_prob = model.classifiers[int(horizon)].predict_proba(x.loc[mask])[:, 1]
        y_peak = targets.loc[mask, peak_col].to_numpy(dtype=float)
        y_storm = targets.loc[mask, storm_col].to_numpy(dtype=int)
        current_abs = np.abs(features.loc[mask, "residual"].to_numpy(dtype=float))
        pred_class = pred_prob >= model.config.probability_threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_storm, pred_class, average="binary", zero_division=0
        )
        tn = int(np.sum(~pred_class & ~(y_storm.astype(bool))))
        fp = int(np.sum(pred_class & ~(y_storm.astype(bool))))
        far = float(fp / (fp + tn)) if fp + tn else None
        report["horizons"][str(horizon)] = {
            "samples": int(mask.sum()),
            "rmse_nt": float(np.sqrt(mean_squared_error(y_peak, pred_peak))),
            "mae_nt": float(mean_absolute_error(y_peak, pred_peak)),
            "storm": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_alarm_rate": far,
                "threshold": float(model.config.probability_threshold),
                "validation_positive_rate": float(np.mean(y_storm)),
            },
            "persistence_rmse_nt": float(np.sqrt(mean_squared_error(y_peak, current_abs))),
            "persistence_mae_nt": float(mean_absolute_error(y_peak, current_abs)),
            "persistence_context_samples": int(len(evaluation_frame)),
            "beats_persistence_rmse": bool(
                np.sqrt(mean_squared_error(y_peak, pred_peak))
                < np.sqrt(mean_squared_error(y_peak, current_abs))
            ),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate geomagnetic ML forecaster")
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--column", default="f_nt")
    ap.add_argument("--chunk-days", type=int, default=14)
    ap.add_argument("--backend", choices=("sklearn", "lightgbm"), default="sklearn")
    ap.add_argument("--model-path", default="magnetometer/data/models/magnetometer_forecaster")
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.chunk_days <= 0 or args.chunk_days > args.days:
        raise SystemExit("--chunk-days must be > 0 and <= --days")

    if args.self_test:
        idx = pd.date_range("2025-01-01", periods=12_000, freq="min", tz="UTC")
        rng = np.random.default_rng(7)
        t = np.arange(len(idx), dtype=float)
        residual = 4.0 * rng.normal(size=len(idx)) + 2.0 * np.sin(t / 90.0)
        # Put examples into the validation tail as well as the training area.
        for start, stop, amplitude in ((4000, 5000, 90.0), (9800, 10200, 75.0), (11200, 11600, 100.0)):
            ramp = np.linspace(0.0, amplitude, min(100, stop - start))
            residual[start : start + len(ramp)] += ramp
            residual[start + len(ramp) : stop] += amplitude
        frame = pd.DataFrame({"residual": residual, "kp": 2.0, "dst": -5.0}, index=idx)
        config = ForecastConfig(backend="sklearn", min_samples_leaf=10, max_iter=120)
        model = GeomagneticForecaster(config=config)
        model.fit(frame, cadence_s=60.0)
        report = _evaluate_validation_tail(model, frame, 60.0)
        model_path = Path(args.model_path)
        model.save_model(model_path)
        loaded = GeomagneticForecaster.load_model(model_path)
        prediction = loaded.predict(frame.tail(720), cadence_s=60.0, current_rule_tier="quiet")
        print(json.dumps({"training": report, "prediction": prediction.horizons}, indent=2))
        return

    logger.info("Fetching %s days of %s data for %s", args.days, args.start_date, args.observatory)
    frame = _prepare_window(args.observatory.upper(), args.start_date, args.days, args.column, args.chunk_days)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual"])
    if len(frame) < 1000:
        raise SystemExit("Not enough usable samples after residual preparation.")

    config = ForecastConfig(backend=args.backend, validation_fraction=args.validation_fraction)
    model = GeomagneticForecaster(config=config)
    training_report = model.fit(frame, cadence_s=60.0)
    evaluation = _evaluate_validation_tail(model, frame, 60.0)
    model.save_model(Path(args.model_path))

    report = {
        "observatory": args.observatory.upper(),
        "start_date": args.start_date,
        "days": args.days,
        "chunk_days": args.chunk_days,
        "training": training_report,
        "evaluation": evaluation,
        "model_path": str(Path(args.model_path).resolve()),
    }
    report_path = Path(args.model_path).with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
