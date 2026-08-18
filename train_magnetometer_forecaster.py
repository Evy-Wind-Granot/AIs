#!/usr/bin/env python3
"""Train, validate and gate the production geomagnetic forecaster.

The protocol is strictly chronological. It uses causal QDC residual features,
purged train/validation/test partitions, a persistence baseline, validation-only
blend calibration, and a fail-closed production gate. Candidate artifacts can
be saved for research, but the live pipeline only accepts a passed artifact.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
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
        print("Dst index unavailable for the requested training window; continuing with missingness features.", file=sys.stderr)
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
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a test set")
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


def _persistence_amplitude(residual: pd.Series, target_index: pd.DatetimeIndex, *, amplitude_window_samples: int) -> np.ndarray:
    """Predict future amplitude by persistence of the current causal window."""
    current = residual.rolling(amplitude_window_samples, min_periods=amplitude_window_samples).max() - residual.rolling(
        amplitude_window_samples, min_periods=amplitude_window_samples
    ).min()
    return current.reindex(target_index).to_numpy(dtype=float)


def _print_metrics(
    name: str,
    model: GeomagneticForecaster,
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    baseline: np.ndarray,
) -> tuple[dict[int, float], dict[int, object]]:
    """Print ML-vs-persistence metrics and return MAE improvements/evaluations."""
    evaluations = model.evaluate(features, targets)
    improvements: dict[int, float] = {}
    print(f"\n{name}:")
    for horizon, evaluation in evaluations.items():
        target = targets[horizon].to_numpy(dtype=float)
        valid = np.isfinite(target) & np.isfinite(baseline)
        if valid.sum() < 100:
            raise RuntimeError(f"not enough valid baseline samples for +{horizon}h")
        baseline_mae = float(mean_absolute_error(target[valid], baseline[valid]))
        baseline_rmse = float(np.sqrt(mean_squared_error(target[valid], baseline[valid])))
        improvement = 100.0 * (baseline_mae - evaluation.mae_nt) / baseline_mae if baseline_mae > 0 else 0.0
        improvements[horizon] = improvement
        print(
            f"  +{horizon}h: ML MAE={evaluation.mae_nt:.2f} nT, RMSE={evaluation.rmse_nt:.2f} nT; "
            f"persistence MAE={baseline_mae:.2f} nT, RMSE={baseline_rmse:.2f} nT; "
            f"MAE improvement={improvement:+.1f}%; precision={evaluation.precision:.3f}, "
            f"recall={evaluation.recall:.3f}, F1={evaluation.f1:.3f}"
        )
    return improvements, evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--column", default="x_nt")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--save-candidate",
        action="store_true",
        help="save a non-passing research artifact under models/artifacts/candidates; never production-approved",
    )
    args = parser.parse_args()
    if args.days < 90:
        parser.error("use at least 90 historical days for production training")

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
    train_features, train_targets, validation_features, validation_targets, test_features, test_targets = _purged_three_way_split(
        features, targets, purge_samples=purge_samples
    )
    print(
        f"Training samples: {len(train_features)}; validation: {len(validation_features)}; "
        f"test: {len(test_features)}; purge: {purge_samples} samples ({max_target_reach_hours:.1f}h)"
    )

    validation_model = GeomagneticForecaster(config).fit(train_features, train_targets)
    validation_weights = validation_model.calibrate_blend(validation_features, validation_targets)
    validation_baseline = _persistence_amplitude(
        residual, validation_features.index,
        amplitude_window_samples=int(round(config.amplitude_window_min * 60.0 / 60.0)),
    )
    validation_improvements, _ = _print_metrics(
        "Chronological validation", validation_model, validation_features, validation_targets, validation_baseline
    )
    print("  Validation blend weights:", ", ".join(f"+{h}h={w:.2f}" for h, w in validation_weights.items()))

    final_train_features = pd.concat([train_features, validation_features])
    final_train_targets = {h: pd.concat([train_targets[h], validation_targets[h]]) for h in targets}
    final_model = GeomagneticForecaster(config).fit(final_train_features, final_train_targets)
    final_model.blend_weights = dict(validation_weights)
    final_model.training_metadata["blend_weights"] = {str(k): float(v) for k, v in validation_weights.items()}

    test_baseline = _persistence_amplitude(
        residual, test_features.index,
        amplitude_window_samples=int(round(config.amplitude_window_min * 60.0 / 60.0)),
    )
    test_improvements, test_evaluations = _print_metrics(
        "Final chronological test", final_model, test_features, test_targets, test_baseline
    )

    # Fail closed: every production horizon must beat persistence on unseen data,
    # and validation cannot show a material regression. Candidate artifacts are
    # explicitly marked non-production and are never consumed by live inference.
    failed_test = [h for h, value in test_improvements.items() if value <= 0.0]
    failed_validation = [h for h, value in validation_improvements.items() if value < -2.0]
    failed = sorted(set(failed_test + failed_validation))
    production_passed = not failed

    final_model.training_metadata.update({
        "production_gate": "passed" if production_passed else "failed",
        "test_mae_improvement_percent": {str(h): float(v) for h, v in test_improvements.items()},
        "validation_mae_improvement_percent": {str(h): float(v) for h, v in validation_improvements.items()},
        "purge_hours": max_target_reach_hours,
        "data_start": analysis_start.isoformat(),
        "data_end": index[-1].isoformat(),
        "index_availability": {"kp": kp is not None, "dst": dst is not None},
        "production_gate_failures": [int(h) for h in failed],
        "model_type": "hist_gradient_boosting_delta_plus_persistence_blend",
    })

    if not production_passed:
        print(
            "\nPRODUCTION GATE FAILED: unseen ML performance did not satisfy the "
            f"persistence safety gate at horizons {failed}. No production artifact was saved.",
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
