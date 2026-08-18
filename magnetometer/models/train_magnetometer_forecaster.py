#!/usr/bin/env python3
"""Train, validate and gate the production geomagnetic forecaster.

The pipeline is strictly chronological. Model selection and blend calibration
use validation data only; the final test set is never used for selection. The
artifact may be deployed when the short operational horizons (+1h/+3h) pass
stable walk-forward gates. Longer horizons remain explicitly experimental
until they demonstrate the same evidence.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import magnetometer_demo as md
from magnetometer.acquisition import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz
from magnetometer.parsing import parse_iaga2002_to_dataframe
from models.forecaster import ForecastConfig, GeomagneticForecaster, build_training_data, save_model


def fetch_inputs(observatory: str, start_date: str, days: int, warmup_days: int):
    """Fetch magnetometer data plus optional global indices in parallel."""
    start = pd.to_datetime(start_date, utc=True)
    fetch_start = (start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end = (start + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    with ThreadPoolExecutor(max_workers=3) as pool:
        mag_future = pool.submit(fetch_intermagnet_iaga2002, observatory, fetch_start, days + warmup_days)
        kp_future = pool.submit(fetch_kp_gfz, fetch_start, end)
        mag = parse_iaga2002_to_dataframe(mag_future.result())
        if mag is None or mag.empty:
            raise RuntimeError(f"No magnetometer data returned for {observatory}")
        months = sorted({(ts.year, ts.month) for ts in mag.index})
        dst_futures = [pool.submit(fetch_dst_kyoto, year, month) for year, month in months]
        dst_parts = [future.result() for future in dst_futures]
    try:
        kp = kp_future.result()
    except Exception as exc:
        print(f"Kp index unavailable; continuing without Kp: {exc}", file=sys.stderr)
        kp = None
    dst_parts = [part for part in dst_parts if part is not None and not part.empty]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    if dst is None:
        print("Dst index unavailable for the requested training window; continuing with missingness features.", file=sys.stderr)
    return mag, kp, dst, start


def _purged_split(features, targets, purge_samples, train_fraction=0.65, validation_fraction=0.15):
    """Chronologically split data with target-aware purge gaps."""
    n = len(features)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    validation_start = train_end + purge_samples
    test_start = validation_end + purge_samples
    if validation_start >= validation_end or test_start >= n:
        raise ValueError("dataset is too small for the requested purged split")
    cut = lambda start, end: {h: y.iloc[start:end] for h, y in targets.items()}
    return (
        features.iloc[:train_end], cut(0, train_end),
        features.iloc[validation_start:validation_end], cut(validation_start, validation_end),
        features.iloc[test_start:], cut(test_start, None),
    )


def _persistence_amplitude(residual: pd.Series, index: pd.DatetimeIndex, window_samples: int) -> np.ndarray:
    rolling = residual.rolling(window_samples, min_periods=window_samples)
    return (rolling.max() - rolling.min()).reindex(index).to_numpy(dtype=float)


def _metrics(model, features, targets, baseline):
    """Return horizon-wise ML-vs-persistence improvement percentages."""
    X = model._prepare_X(features, model.feature_columns)
    mae_improvement, rmse_improvement = {}, {}
    for horizon in model.config.horizons_hours:
        y = pd.Series(targets[horizon], index=features.index, dtype=float).to_numpy()
        pred = model._blended_predictions(X, horizon)
        valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(baseline)
        if valid.sum() < model.config.confidence_min_samples:
            raise RuntimeError(f"not enough finite evaluation samples for +{horizon}h")
        model_mae = mean_absolute_error(y[valid], pred[valid])
        model_rmse = np.sqrt(mean_squared_error(y[valid], pred[valid]))
        base_mae = mean_absolute_error(y[valid], baseline[valid])
        base_rmse = np.sqrt(mean_squared_error(y[valid], baseline[valid]))
        mae_improvement[horizon] = 100.0 * (base_mae - model_mae) / base_mae if base_mae else 0.0
        rmse_improvement[horizon] = 100.0 * (base_rmse - model_rmse) / base_rmse if base_rmse else 0.0
    return mae_improvement, rmse_improvement


def _select_model(base_config, train_features, train_targets, validation_features, validation_targets, validation_baseline, requested_loss):
    """Select hyperparameters using validation data only, emphasizing weak horizons."""
    losses = (requested_loss,) if requested_loss != "auto" else ("absolute_error", "squared_error")
    candidates = []
    for loss in losses:
        candidates.extend([
            replace(base_config, regression_loss=loss, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0),
            replace(base_config, regression_loss=loss, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=2.0),
            replace(base_config, regression_loss=loss, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=4.0),
        ])
    best_model = None
    best_info = None
    best_score = -float("inf")
    for candidate in candidates:
        model = GeomagneticForecaster(candidate).fit(train_features, train_targets)
        weights = model.calibrate_blend(validation_features, validation_targets)
        mae, rmse = _metrics(model, validation_features, validation_targets, validation_baseline)
        scores = {h: 0.5 * mae[h] + 0.5 * rmse[h] for h in candidate.horizons_hours}
        score = min(scores.values()) + 0.05 * float(np.mean(list(scores.values())))
        if score > best_score:
            best_score = score
            best_model = model
            best_info = {
                "regression_loss": candidate.regression_loss,
                "learning_rate": candidate.learning_rate,
                "max_leaf_nodes": candidate.max_leaf_nodes,
                "min_samples_leaf": candidate.min_samples_leaf,
                "l2_regularization": candidate.l2_regularization,
                "validation_blend_weights": {str(k): float(v) for k, v in weights.items()},
                "validation_mae_improvement_percent": {str(k): float(v) for k, v in mae.items()},
                "validation_rmse_improvement_percent": {str(k): float(v) for k, v in rmse.items()},
                "selection_score": float(score),
            }
    if best_model is None:
        raise RuntimeError("validation model selection produced no candidate")
    return best_model, best_info


def _print_metrics(name, model, features, targets, baseline):
    evaluations = model.evaluate(features, targets)
    mae, rmse = _metrics(model, features, targets, baseline)
    print(f"\n{name}:")
    for h in model.config.horizons_hours:
        e = evaluations[h]
        print(
            f"  +{h}h: ML MAE={e.mae_nt:.2f} nT, RMSE={e.rmse_nt:.2f} nT; "
            f"persistence MAE={mean_absolute_error(targets[h].dropna(), baseline[targets[h].notna()]):.2f} nT; "
            f"MAE improvement={mae[h]:+.1f}%, RMSE improvement={rmse[h]:+.1f}%; "
            f"precision={e.precision:.3f}, recall={e.recall:.3f}, F1={e.f1:.3f}"
        )
    return mae, rmse, evaluations


def _walk_forward(model, features, targets, residual, folds, window_samples):
    """Evaluate a frozen model over contiguous unseen folds."""
    boundaries = np.linspace(0, len(features), folds + 1, dtype=int)
    result = {h: [] for h in model.config.horizons_hours}
    print(f"\nWalk-forward stability backtest ({folds} contiguous unseen folds):")
    for fold in range(folds):
        start, end = int(boundaries[fold]), int(boundaries[fold + 1])
        ff = features.iloc[start:end]
        tt = {h: y.iloc[start:end] for h, y in targets.items()}
        baseline = _persistence_amplitude(residual, ff.index, window_samples)
        mae, rmse = _metrics(model, ff, tt, baseline)
        for h in model.config.horizons_hours:
            result[h].append((mae[h], rmse[h]))
            print(f"  fold {fold + 1}/{folds} +{h}h: MAE {mae[h]:+.1f}%, RMSE {rmse[h]:+.1f}% ({ff.index.min().date()} -> {ff.index.max().date()})")
    stability = {}
    for h, values in result.items():
        m = np.asarray([x[0] for x in values], dtype=float)
        r = np.asarray([x[1] for x in values], dtype=float)
        stability[h] = {
            "mean_mae_improvement_percent": float(m.mean()),
            "median_mae_improvement_percent": float(np.median(m)),
            "min_mae_improvement_percent": float(m.min()),
            "positive_mae_fold_fraction": float(np.mean(m > 0)),
            "mean_rmse_improvement_percent": float(r.mean()),
            "median_rmse_improvement_percent": float(np.median(r)),
            "min_rmse_improvement_percent": float(r.min()),
        }
        s = stability[h]
        print(f"  +{h}h stability: mean MAE={s['mean_mae_improvement_percent']:+.1f}%, median MAE={s['median_mae_improvement_percent']:+.1f}%, positive folds={s['positive_mae_fold_fraction']:.0%}; mean RMSE={s['mean_rmse_improvement_percent']:+.1f}%")
    return stability


def _gate_horizon(horizon: int, validation_mae: float, validation_rmse: float, test_mae: float, test_rmse: float, stability: dict[str, float]) -> tuple[bool, list[str]]:
    """Apply a horizon-aware evidence gate.

    +1h is strict. +3h allows a single modest RMSE regression in one fold when
    aggregate performance is positive and at least 75% of folds improve.
    +6h remains experimental unless it clears the same strict +1h evidence.
    """
    failures: list[str] = []
    if validation_mae < -2.0: failures.append("validation_mae")
    if validation_rmse < -2.0: failures.append("validation_rmse")
    if test_mae <= 0.0: failures.append("test_mae")
    if test_rmse < -2.0: failures.append("test_rmse")
    if stability["mean_mae_improvement_percent"] <= 0.0: failures.append("mean_mae")
    if stability["median_mae_improvement_percent"] <= 0.0: failures.append("median_mae")
    if stability["positive_mae_fold_fraction"] < 0.75: failures.append("positive_fold_fraction")
    if stability["min_mae_improvement_percent"] < -5.0: failures.append("worst_fold_mae")
    min_rmse = -10.0 if horizon == 3 else -5.0
    if stability["mean_rmse_improvement_percent"] < -2.0: failures.append("mean_rmse")
    if stability["min_rmse_improvement_percent"] < min_rmse: failures.append("worst_fold_rmse")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--column", default="x_nt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--backtest-folds", type=int, default=4)
    parser.add_argument("--model-loss", choices=("auto", "absolute_error", "huber", "squared_error"), default="auto")
    parser.add_argument("--save-candidate", action="store_true")
    args = parser.parse_args()
    if args.days < 90: parser.error("use at least 90 historical days for production training")
    if args.backtest_folds < 2: parser.error("--backtest-folds must be at least 2")

    mag, kp, dst, analysis_start = fetch_inputs(args.observatory, args.start_date, args.days, args.warmup_days)
    if args.column not in mag.columns:
        parser.error(f"magnetometer column {args.column!r} is unavailable; available columns: {list(mag.columns)}")
    result = md.run_analysis(
        mag[args.column].to_numpy(), 60.0,
        label=f"{args.observatory} ML training {args.start_date}+{args.days}d",
        start_time=mag.index.min().to_pydatetime(), analysis_start_time=analysis_start.to_pydatetime(),
        dst_series=dst, kp_series=kp, observatory=args.observatory,
    )
    if result["status"] != "ok":
        print(f"Training aborted: deterministic quality gate returned {result['status']}", file=sys.stderr)
        return 2

    index = pd.date_range(analysis_start, periods=len(result["residual"]), freq="min", tz="UTC")
    residual = pd.Series(np.asarray(result["residual"], dtype=float), index=index)
    kp_aligned = kp.reindex(index, method="ffill") if kp is not None else None
    dst_aligned = dst.reindex(index, method="ffill") if dst is not None else None
    config = ForecastConfig(
        minor_storm_nt=md.FLAG_THRESHOLD_MINOR_STORM_NT,
        major_storm_nt=md.FLAG_THRESHOLD_MAJOR_STORM_NT,
        severe_storm_nt=md.FLAG_THRESHOLD_SEVERE_STORM_NT,
        amplitude_window_min=int(round(md.FLAG_AMPLITUDE_WINDOW_MIN)),
    )
    features, targets = build_training_data(residual, kp_aligned, dst_aligned, cadence_s=60.0, config=config)
    valid = features["persistence_amplitude_nt"].notna()
    features = features.loc[valid]
    targets = {h: y.loc[features.index] for h, y in targets.items()}
    window_samples = int(round(config.amplitude_window_min * 60.0 / 60.0))
    purge_samples = int(round((max(config.horizons_hours) + config.amplitude_window_min / 60.0) * 60.0))
    train_f, train_t, val_f, val_t, test_f, test_t = _purged_split(features, targets, purge_samples)
    purge_hours = purge_samples / 60.0
    print(f"Training samples: {len(train_f)}; validation: {len(val_f)}; test: {len(test_f)}; purge: {purge_samples} samples ({purge_hours:.1f}h)")

    val_baseline = _persistence_amplitude(residual, val_f.index, window_samples)
    selected, selection = _select_model(config, train_f, train_t, val_f, val_t, val_baseline, "absolute_error" if args.model_loss == "huber" else args.model_loss)
    val_mae, val_rmse, _ = _print_metrics("Chronological validation (selected model)", selected, val_f, val_t, val_baseline)
    print("  Selected model:", selection)
    print("  Validation blend weights:", ", ".join(f"+{h}h={w:.2f}" for h, w in selected.blend_weights.items()))

    final_f = pd.concat([train_f, val_f])
    final_t = {h: pd.concat([train_t[h], val_t[h]]) for h in targets}
    final_config = replace(config,
        regression_loss=str(selection["regression_loss"]), learning_rate=float(selection["learning_rate"]),
        max_leaf_nodes=int(selection["max_leaf_nodes"]), min_samples_leaf=int(selection["min_samples_leaf"]),
        l2_regularization=float(selection["l2_regularization"]),
    )
    model = GeomagneticForecaster(final_config).fit(final_f, final_t)
    model.blend_weights = dict(selected.blend_weights)
    model.training_metadata["validation_model_selection"] = selection
    test_baseline = _persistence_amplitude(residual, test_f.index, window_samples)
    test_mae, test_rmse, evaluations = _print_metrics("Final chronological test", model, test_f, test_t, test_baseline)
    stability = _walk_forward(model, test_f, test_t, residual, args.backtest_folds, window_samples)

    statuses: dict[str, str] = {}
    gate_failures: dict[str, list[str]] = {}
    for h in final_config.horizons_hours:
        passed, failures = _gate_horizon(h, val_mae[h], val_rmse[h], test_mae[h], test_rmse[h], stability[h])
        statuses[str(h)] = "production" if passed else "experimental"
        gate_failures[str(h)] = failures

    approved = [int(h) for h, status in statuses.items() if status == "production"]
    # A deployment artifact must have a reliable +1h operational forecast.
    production_passed = 1 in approved
    if 3 in approved and 1 not in approved:
        production_passed = False
    gate_state = "passed" if production_passed else "failed"
    model.training_metadata.update({
        "production_gate": gate_state,
        "approved_horizons_hours": approved,
        "horizon_deployment_status": statuses,
        "production_gate_failures": gate_failures,
        "test_mae_improvement_percent": {str(h): float(v) for h, v in test_mae.items()},
        "test_rmse_improvement_percent": {str(h): float(v) for h, v in test_rmse.items()},
        "validation_mae_improvement_percent": {str(h): float(v) for h, v in val_mae.items()},
        "validation_rmse_improvement_percent": {str(h): float(v) for h, v in val_rmse.items()},
        "walk_forward_stability": {str(h): v for h, v in stability.items()},
        "backtest_folds": int(args.backtest_folds),
        "purge_hours": purge_hours,
        "data_start": analysis_start.isoformat(), "data_end": index[-1].isoformat(),
        "index_availability": {"kp": kp is not None, "dst": dst is not None},
        "data_quality": {
            "nonfinite_residual_fraction": float(np.mean(~np.isfinite(residual.to_numpy(dtype=float)))),
            "dst_available": bool(dst is not None), "kp_available": bool(kp is not None),
        },
        "model_type": "hist_gradient_boosting_delta_plus_persistence_blend",
        "gate_policy": {
            "operational_horizons": [1, 3],
            "required_positive_fold_fraction": 0.75,
            "validation_max_regression_percent": 2.0,
            "test_max_rmse_regression_percent": 2.0,
            "worst_fold_mae_regression_percent": 5.0,
            "worst_fold_rmse_regression_percent": {"1": 5.0, "3": 10.0, "6": 5.0},
            "long_horizon_policy": "6h remains experimental until it independently clears the evidence gate",
        },
    })

    print("\nHorizon deployment status:")
    for h in final_config.horizons_hours:
        print(f"  +{h}h: {statuses[str(h)].upper()}" + (f" ({', '.join(gate_failures[str(h)])})" if gate_failures[str(h)] else ""))

    if not production_passed:
        print("\nPRODUCTION GATE FAILED: +1h did not meet the minimum deployment evidence. No production artifact was saved.", file=sys.stderr)
        if args.save_candidate:
            candidate = Path(args.output or f"models/artifacts/candidates/{args.observatory.lower()}_forecaster.pkl")
            save_model(model, candidate)
            print(f"Saved research candidate (NOT production-approved): {candidate}")
        return 3

    output = Path(args.output or f"models/artifacts/{args.observatory.lower()}_forecaster.pkl")
    save_model(model, output)
    print(f"\nPRODUCTION GATE PASSED: approved horizons {approved}")
    print(f"Saved production artifact: {output}")
    if 6 not in approved:
        print("NOTE: +6h remains experimental and will not be exposed as a production status forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
