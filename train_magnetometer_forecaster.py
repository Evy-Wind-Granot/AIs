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
from models.forecaster import ForecastConfig, GeomagneticForecaster, evaluate_forecast  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magnetometer_forecaster_train")


def _fetch_dst_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    parts: List[pd.Series] = []
    months = pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M")
    for period in months:
        part = fetch_dst_kyoto(int(period.year), int(period.month))
        if part is not None and not part.empty:
            parts.append(part)
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def _prepare_window(observatory: str, start_date: str, days: int, column: str) -> pd.DataFrame:
    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
        samples_per_day="Minute",
    )
    df = parse_iaga2002_to_dataframe(raw)
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not present in INTERMAGNET response")
    series = pd.to_numeric(df[column], errors="coerce")
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
    for start in range(0, max(1, n - step_samples), step_samples):
        end = min(start + window_samples, n)
        if end - start < max(2, step_samples // 2):
            continue
        segment = values[start:end]
        if np.isfinite(segment).sum() < (end - start) * 0.5:
            continue
        t_seg = t_global[start:end]
        seg_base, coeffs = robust_harmonic_baseline(segment, cadence, t_hours=t_seg, t_ref_min=t_min, t_ref_max=t_max)
        seg_res = segment - seg_base
        storm_frac = np.mean(np.abs(seg_res) > 50.0)
        if storm_frac > 0.05 and last_good is not None:
            from magnetometer_demo import build_design_matrix
            seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good
        elif storm_frac <= 0.05:
            last_good = coeffs
        w = np.hanning(end - start)
        baseline[start:end] += seg_base * w
        weights[start:end] += w

    mask = weights > 0
    fallback = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
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
    # Build metrics on the holdout rows only. The full frame above is used to
    # construct causal rolling context, but no pre-holdout row enters the score.
    from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
    report = {"horizons": {}}
    x = model._sanitize_features(features.loc[:, model.feature_names])
    for horizon in model.config.horizons_hours:
        peak_col = f"target_peak_abs_{horizon}h"
        storm_col = f"target_storm_{horizon}h"
        mask = evaluation_mask & targets[peak_col].notna() & targets[storm_col].notna()
        pred_peak = np.clip(model.regressors[int(horizon)].predict(x.loc[mask]), 0.0, None)
        pred_prob = model.classifiers[int(horizon)].predict_proba(x.loc[mask])[:, 1]
        y_peak = targets.loc[mask, peak_col].to_numpy(dtype=float)
        y_storm = targets.loc[mask, storm_col].to_numpy(dtype=int)
        current_abs = np.abs(features.loc[mask, "residual"].to_numpy(dtype=float))
        pred_class = pred_prob >= model.config.probability_threshold
        precision, recall, f1, _ = precision_recall_fscore_support(y_storm, pred_class, average="binary", zero_division=0)
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
            },
            "persistence_rmse_nt": float(np.sqrt(mean_squared_error(y_peak, current_abs))),
            "persistence_mae_nt": float(mean_absolute_error(y_peak, current_abs)),
            "persistence_context_samples": int(len(evaluation_frame)),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate geomagnetic ML forecaster")
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--column", default="f_nt")
    ap.add_argument("--backend", choices=("sklearn", "lightgbm"), default="sklearn")
    ap.add_argument("--model-path", default="magnetometer/data/models/magnetometer_forecaster")
    ap.add_argument("--validation-fraction", type=float, default=0.20)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        idx = pd.date_range("2025-01-01", periods=12_000, freq="min", tz="UTC")
        rng = np.random.default_rng(7)
        t = np.arange(len(idx), dtype=float)
        residual = 4.0 * rng.normal(size=len(idx)) + 2.0 * np.sin(t / 90.0)
        residual[4000:4300] += np.linspace(0.0, 90.0, 300)
        residual[4300:4700] += 90.0
        residual[4700:5000] += np.linspace(90.0, 0.0, 5000 - 4700)
        frame = pd.DataFrame({"residual": residual, "kp": 2.0, "dst": -5.0}, index=idx)
        config = ForecastConfig(backend="sklearn", min_samples_leaf=10, max_iter=120)
        model = GeomagneticForecaster(config=config)
        model.fit(frame, cadence_s=60.0)
        report = _evaluate_validation_tail(model, frame, 60.0)
        model.save_model(Path(args.model_path))
        loaded = GeomagneticForecaster.load_model(Path(args.model_path))
        prediction = loaded.predict(frame.tail(720), cadence_s=60.0, current_rule_tier="quiet")
        print(json.dumps({"training": report, "prediction": prediction.horizons}, indent=2))
        return

    logger.info("Fetching %s days of %s data for %s", args.days, args.start_date, args.observatory)
    frame = _prepare_window(args.observatory.upper(), args.start_date, args.days, args.column)
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
