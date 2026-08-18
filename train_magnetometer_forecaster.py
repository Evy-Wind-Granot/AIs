#!/usr/bin/env python3
"""Train, backtest and gate the production geomagnetic forecaster.

The protocol is strictly chronological and deliberately harder than a single
train/test split. Model-family selection and blend calibration use validation
data only. Production performance is then measured on multiple contiguous,
unseen walk-forward test windows. A model is published only when its
performance is positive in aggregate, stable across folds, and does not show
a material RMSE regression against persistence.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import magnetometer_demo as md
from magnetometer.acquisition import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz
from magnetometer.parsing import parse_iaga2002_to_dataframe
from models.forecaster import ForecastConfig, GeomagneticForecaster, build_training_data, save_model


def fetch_inputs(
    observatory: str, start_date: str, days: int, warmup_days: int
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, pd.Timestamp]:
    """Fetch a historical magnetometer window plus optional Kp/Dst."""
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
        print(
            "Dst index unavailable for the requested training window; continuing with missingness features.",
            file=sys.stderr,
        )
    return mag, kp, dst, start


def _purged_three_way_split(
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    *,
    purge_samples: int,
    train_fraction: float = 0.65,
    validation_fraction: float = 0.15,
):
    """Split chronologically with target-aware gaps between every partition."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("invalid chronological split fractions")
    n = len(features)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    validation_start = train_end + purge_samples
    test_start = validation_end + purge_samples
    if validation_start >= validation_end or test_start >= n:
        raise ValueError("dataset is too small for the requested purged split")

    def cut(start: int, end: int | None) -> dict[int, pd.Series]:
        return {h: target.iloc[start:end] for h, target in targets.items()}

    return (
        features.iloc[:train_end], cut(0, train_end),
        features.iloc[validation_start:validation_end], cut(validation_start, validation_end),
        features.iloc[test_start:], cut(test_start, None),
    )


def _persistence_amplitude(
    residual: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    amplitude_window_samples: int,
) -> np.ndarray:
    """Predict future amplitude by persistence of the current causal window."""
    rolling = residual.rolling(amplitude_window_samples, min_periods=amplitude_window_samples)
    current = rolling.max() - rolling.min()
    return current.reindex(target_index).to_numpy(dtype=float)


def _metric_pair(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return model MAE/RMSE and persistence MAE/RMSE on identical samples."""
    valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(baseline)
    if int(valid.sum()) < 100:
        raise RuntimeError("not enough finite samples for reliable evaluation")
    model_mae = float(mean_absolute_error(target[valid], prediction[valid]))
    model_rmse = float(np.sqrt(mean_squared_error(target[valid], prediction[valid])))
    base_mae = float(mean_absolute_error(target[valid], baseline[valid]))
    base_rmse = float(np.sqrt(mean_squared_error(target[valid], baseline[valid])))
    return model_mae, model_rmse, base_mae, base_rmse


def _evaluate_improvements(
    model: GeomagneticForecaster,
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    baseline: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return MAE/RMSE improvements without printing candidate diagnostics."""
    X = model._prepare_X(features, model.feature_columns)
    mae_improvements: dict[int, float] = {}
    rmse_improvements: dict[int, float] = {}
    for horizon in model.config.horizons_hours:
        target = targets[horizon].to_numpy(dtype=float)
        prediction = model._blended_predictions(X, horizon)
        model_mae, model_rmse, base_mae, base_rmse = _metric_pair(target, prediction, baseline)
        mae_improvements[horizon] = 100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else 0.0
        rmse_improvements[horizon] = 100.0 * (base_rmse - model_rmse) / base_rmse if base_rmse > 0 else 0.0
    return mae_improvements, rmse_improvements


def _select_validation_model(
    base_config: ForecastConfig,
    train_features: pd.DataFrame,
    train_targets: dict[int, pd.Series],
    validation_features: pd.DataFrame,
    validation_targets: dict[int, pd.Series],
    validation_baseline: np.ndarray,
    requested_loss: str,
) -> tuple[GeomagneticForecaster, dict[str, object]]:
    """Select model hyperparameters using validation data only.

    The final test set is never consulted here. Selection emphasizes the
    weakest horizon by maximizing the minimum horizon score, which prevents a
    strong +1h result from hiding an unstable +6h model.
    """
    if requested_loss != "auto":
        losses = (requested_loss,)
    else:
        losses = ("absolute_error", "huber", "squared_error")

    candidates: list[ForecastConfig] = []
    for loss in losses:
        candidates.extend(
            [
                replace(base_config, regression_loss=loss, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0),
                replace(base_config, regression_loss=loss, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=2.0),
            ]
        )

    best: GeomagneticForecaster | None = None
    best_info: dict[str, object] | None = None
    best_score = -float("inf")

    for candidate in candidates:
        model = GeomagneticForecaster(candidate).fit(train_features, train_targets)
        weights = model.calibrate_blend(validation_features, validation_targets)
        mae, rmse = _evaluate_improvements(model, validation_features, validation_targets, validation_baseline)
        horizon_scores = {
            h: 0.5 * mae[h] + 0.5 * rmse[h] for h in candidate.horizons_hours
        }
        worst_horizon_score = min(horizon_scores.values())
        mean_score = float(np.mean(list(horizon_scores.values())))
        score = worst_horizon_score + 0.05 * mean_score
        if score > best_score:
            best_score = score
            best = model
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

    if best is None or best_info is None:
        raise RuntimeError("validation model selection produced no candidate")
    return best, best_info


def _print_metrics(
    name: str,
    model: GeomagneticForecaster,
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    baseline: np.ndarray,
) -> tuple[dict[int, float], dict[int, float], dict[int, object]]:
    """Print ML-vs-persistence metrics and return MAE/RMSE improvements."""
    evaluations = model.evaluate(features, targets)
    mae_improvements: dict[int, float] = {}
    rmse_improvements: dict[int, float] = {}
    print(f"\n{name}:")
    X = model._prepare_X(features, model.feature_columns)
    for horizon, evaluation in evaluations.items():
        target = targets[horizon].to_numpy(dtype=float)
        model_mae, model_rmse, baseline_mae, baseline_rmse = _metric_pair(
            target, model._blended_predictions(X, horizon), baseline
        )
        mae_improvement = 100.0 * (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0 else 0.0
        rmse_improvement = 100.0 * (baseline_rmse - model_rmse) / baseline_rmse if baseline_rmse > 0 else 0.0
        mae_improvements[horizon] = mae_improvement
        rmse_improvements[horizon] = rmse_improvement
        print(
            f"  +{horizon}h: ML MAE={model_mae:.2f} nT, RMSE={model_rmse:.2f} nT; "
            f"persistence MAE={baseline_mae:.2f} nT, RMSE={baseline_rmse:.2f} nT; "
            f"MAE improvement={mae_improvement:+.1f}%, RMSE improvement={rmse_improvement:+.1f}%; "
            f"precision={evaluation.precision:.3f}, recall={evaluation.recall:.3f}, F1={evaluation.f1:.3f}"
        )
    return mae_improvements, rmse_improvements, evaluations


def _evaluate_backtest_stability(
    model: GeomagneticForecaster,
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    residual: pd.Series,
    *,
    folds: int,
    amplitude_window_samples: int,
) -> dict[int, dict[str, float]]:
    """Evaluate the frozen model on multiple contiguous unseen test windows."""
    if folds < 2:
        raise ValueError("backtest folds must be at least 2")
    if len(features) < folds * 500:
        raise ValueError("not enough samples for requested backtest folds")

    boundaries = np.linspace(0, len(features), folds + 1, dtype=int)
    by_horizon: dict[int, list[tuple[float, float]]] = {h: [] for h in model.config.horizons_hours}
    print(f"\nWalk-forward stability backtest ({folds} contiguous unseen folds):")

    for fold in range(folds):
        start, end = int(boundaries[fold]), int(boundaries[fold + 1])
        fold_features = features.iloc[start:end]
        fold_targets = {h: y.iloc[start:end] for h, y in targets.items()}
        baseline = _persistence_amplitude(
            residual, fold_features.index, amplitude_window_samples=amplitude_window_samples
        )
        X = model._prepare_X(fold_features, model.feature_columns)
        for horizon in model.config.horizons_hours:
            target = fold_targets[horizon].to_numpy(dtype=float)
            prediction = model._blended_predictions(X, horizon)
            model_mae, model_rmse, base_mae, base_rmse = _metric_pair(target, prediction, baseline)
            mae_imp = 100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else 0.0
            rmse_imp = 100.0 * (base_rmse - model_rmse) / base_rmse if base_rmse > 0 else 0.0
            by_horizon[horizon].append((mae_imp, rmse_imp))
            print(
                f"  fold {fold + 1}/{folds} +{horizon}h: MAE {mae_imp:+.1f}%, RMSE {rmse_imp:+.1f}% "
                f"({fold_features.index.min().date()} -> {fold_features.index.max().date()})"
            )

    stability: dict[int, dict[str, float]] = {}
    for horizon, values in by_horizon.items():
        mae = np.asarray([v[0] for v in values], dtype=float)
        rmse = np.asarray([v[1] for v in values], dtype=float)
        stability[horizon] = {
            "mean_mae_improvement_percent": float(mae.mean()),
            "median_mae_improvement_percent": float(np.median(mae)),
            "min_mae_improvement_percent": float(mae.min()),
            "positive_mae_fold_fraction": float(np.mean(mae > 0.0)),
            "mean_rmse_improvement_percent": float(rmse.mean()),
            "median_rmse_improvement_percent": float(np.median(rmse)),
            "min_rmse_improvement_percent": float(rmse.min()),
        }
        print(
            f"  +{horizon}h stability: mean MAE={stability[horizon]['mean_mae_improvement_percent']:+.1f}%, "
            f"median MAE={stability[horizon]['median_mae_improvement_percent']:+.1f}%, "
            f"positive folds={stability[horizon]['positive_mae_fold_fraction']:.0%}; "
            f"mean RMSE={stability[horizon]['mean_rmse_improvement_percent']:+.1f}%"
        )
    return stability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--column", default="x_nt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--backtest-folds", type=int, default=4)
    parser.add_argument(
        "--model-loss",
        choices=("auto", "absolute_error", "huber", "squared_error"),
        default="auto",
        help="validation-selected regression loss; auto compares robust and squared losses",
    )
    parser.add_argument(
        "--save-candidate",
        action="store_true",
        help="save a non-passing research artifact under models/artifacts/candidates; never production-approved",
    )
    args = parser.parse_args()
    if args.days < 90:
        parser.error("use at least 90 historical days for production training")
    if args.backtest_folds < 2:
        parser.error("--backtest-folds must be at least 2")

    mag, kp, dst, analysis_start = fetch_inputs(args.observatory, args.start_date, args.days, args.warmup_days)
    if args.column not in mag.columns:
        parser.error(f"magnetometer column {args.column!r} is unavailable; available columns: {list(mag.columns)}")

    result = md.run_analysis(
        mag[args.column].to_numpy(), 60.0,
        label=f"{args.observatory} ML training {args.start_date}+{args.days}d",
        start_time=mag.index.min().to_pydatetime(),
        analysis_start_time=analysis_start.to_pydatetime(),
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
    valid_rows = features["persistence_amplitude_nt"].notna()
    features = features.loc[valid_rows]
    targets = {h: target.loc[features.index] for h, target in targets.items()}

    max_target_reach_hours = max(config.horizons_hours) + config.amplitude_window_min / 60.0
    purge_samples = int(round(max_target_reach_hours * 3600.0 / 60.0))
    (
        train_features, train_targets,
        validation_features, validation_targets,
        test_features, test_targets,
    ) = _purged_three_way_split(features, targets, purge_samples=purge_samples)
    print(
        f"Training samples: {len(train_features)}; validation: {len(validation_features)}; "
        f"test: {len(test_features)}; purge: {purge_samples} samples ({max_target_reach_hours:.1f}h)"
    )

    validation_baseline = _persistence_amplitude(
        residual, validation_features.index,
        amplitude_window_samples=int(round(config.amplitude_window_min * 60.0 / 60.0)),
    )
    validation_model, selection = _select_validation_model(
        config, train_features, train_targets, validation_features, validation_targets,
        validation_baseline, args.model_loss,
    )
    validation_weights = dict(validation_model.blend_weights)
    validation_improvements, validation_rmse, _ = _print_metrics(
        "Chronological validation (selected model)",
        validation_model, validation_features, validation_targets, validation_baseline,
    )
    print("  Selected model:", selection)
    print("  Validation blend weights:", ", ".join(f"+{h}h={w:.2f}" for h, w in validation_weights.items()))

    final_train_features = pd.concat([train_features, validation_features])
    final_train_targets = {h: pd.concat([train_targets[h], validation_targets[h]]) for h in targets}
    final_config = replace(config, **{
        "regression_loss": str(selection["regression_loss"]),
        "learning_rate": float(selection["learning_rate"]),
        "max_leaf_nodes": int(selection["max_leaf_nodes"]),
        "min_samples_leaf": int(selection["min_samples_leaf"]),
        "l2_regularization": float(selection["l2_regularization"]),
    })
    final_model = GeomagneticForecaster(final_config).fit(final_train_features, final_train_targets)
    final_model.blend_weights = dict(validation_weights)
    final_model.training_metadata["blend_weights"] = {str(k): float(v) for k, v in validation_weights.items()}
    final_model.training_metadata["validation_model_selection"] = selection

    test_baseline = _persistence_amplitude(
        residual, test_features.index,
        amplitude_window_samples=int(round(config.amplitude_window_min * 60.0 / 60.0)),
    )
    test_improvements, test_rmse, test_evaluations = _print_metrics(
        "Final chronological test", final_model, test_features, test_targets, test_baseline
    )

    stability = _evaluate_backtest_stability(
        final_model, test_features, test_targets, residual,
        folds=args.backtest_folds,
        amplitude_window_samples=int(round(config.amplitude_window_min * 60.0 / 60.0)),
    )

    failed: list[int] = []
    for horizon in final_config.horizons_hours:
        s = stability[horizon]
        if validation_improvements[horizon] < -2.0 or validation_rmse[horizon] < -2.0:
            failed.append(horizon)
            continue
        if test_improvements[horizon] <= 0.0 or test_rmse[horizon] < -2.0:
            failed.append(horizon)
            continue
        if (
            s["mean_mae_improvement_percent"] <= 0.0
            or s["median_mae_improvement_percent"] <= 0.0
            or s["positive_mae_fold_fraction"] < 0.75
            or s["min_mae_improvement_percent"] < -5.0
            or s["mean_rmse_improvement_percent"] < -2.0
            or s["min_rmse_improvement_percent"] < -5.0
        ):
            failed.append(horizon)

    failed = sorted(set(failed))
    production_passed = not failed

    final_model.training_metadata.update({
        "production_gate": "passed" if production_passed else "failed",
        "test_mae_improvement_percent": {str(h): float(v) for h, v in test_improvements.items()},
        "test_rmse_improvement_percent": {str(h): float(v) for h, v in test_rmse.items()},
        "validation_mae_improvement_percent": {str(h): float(v) for h, v in validation_improvements.items()},
        "validation_rmse_improvement_percent": {str(h): float(v) for h, v in validation_rmse.items()},
        "walk_forward_stability": {str(h): values for h, values in stability.items()},
        "backtest_folds": int(args.backtest_folds),
        "purge_hours": max_target_reach_hours,
        "data_start": analysis_start.isoformat(),
        "data_end": index[-1].isoformat(),
        "index_availability": {"kp": kp is not None, "dst": dst is not None},
        "production_gate_failures": [int(h) for h in failed],
        "model_type": "hist_gradient_boosting_delta_plus_persistence_blend",
        "gate_policy": {
            "validation_max_mae_regression_percent": 2.0,
            "validation_max_rmse_regression_percent": 2.0,
            "test_min_mae_improvement_percent": 0.0,
            "test_max_rmse_regression_percent": 2.0,
            "minimum_positive_fold_fraction": 0.75,
            "maximum_fold_mae_regression_percent": 5.0,
            "maximum_fold_rmse_regression_percent": 5.0,
        },
    })

    if not production_passed:
        print(
            "\nPRODUCTION GATE FAILED: performance was not stable enough across "
            f"unseen chronological windows at horizons {failed}. No production artifact was saved.",
            file=sys.stderr,
        )
        if args.save_candidate:
            candidate = Path(args.output or f"models/artifacts/candidates/{args.observatory.lower()}_forecaster.pkl")
            candidate.parent.mkdir(parents=True, exist_ok=True)
            save_model(final_model, candidate)
            print(f"Saved research candidate (NOT production-approved): {candidate}")
        return 3

    output = Path(args.output or f"models/artifacts/{args.observatory.lower()}_forecaster.pkl")
    save_model(final_model, output)
    print(f"\nSaved production artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
