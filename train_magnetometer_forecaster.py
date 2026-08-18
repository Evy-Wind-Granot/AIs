#!/usr/bin/env python3
"""Train and validate the short-horizon geomagnetic forecaster on real data.

The training protocol is deliberately operational rather than a random ML
split: chronological train/validation/test partitions, a purge gap covering
the complete future-target reach, persistence baselines, and a production
 gate that refuses to publish an artifact which cannot beat persistence.

Example:
    python train_magnetometer_forecaster.py --observatory VIC \
        --start-date 2024-01-01 --days 180
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
from magnetometer.acquisition import (
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
)
from magnetometer.parsing import parse_iaga2002_to_dataframe
from models.forecaster import (
    ForecastConfig,
    GeomagneticForecaster,
    build_training_data,
    save_model,
)


def fetch_inputs(
    observatory: str, start_date: str, days: int, warmup_days: int
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, pd.Timestamp]:
    """Fetch a historical magnetometer window plus Kp and Dst."""
    start = pd.to_datetime(start_date, utc=True)
    fetch_start = (start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end = (start + pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    with ThreadPoolExecutor(max_workers=3) as pool:
        mag_future = pool.submit(
            fetch_intermagnet_iaga2002,
            observatory,
            fetch_start,
            days + warmup_days,
        )
        kp_future = pool.submit(fetch_kp_gfz, fetch_start, end)
        mag = parse_iaga2002_to_dataframe(mag_future.result())
        if mag is None or mag.empty:
            raise RuntimeError(f"No magnetometer data returned for {observatory}")

        months = sorted({(ts.year, ts.month) for ts in mag.index})
        dst_futures = [
            pool.submit(fetch_dst_kyoto, year, month)
            for year, month in months
        ]
        dst_parts = [future.result() for future in dst_futures]

    dst_parts = [
        part for part in dst_parts if part is not None and not part.empty
    ]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    return mag, kp_future.result(), dst, start


def _purged_three_way_split(
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    *,
    purge_samples: int,
    train_fraction: float = 0.65,
    validation_fraction: float = 0.15,
) -> tuple[
    pd.DataFrame,
    dict[int, pd.Series],
    pd.DataFrame,
    dict[int, pd.Series],
    pd.DataFrame,
    dict[int, pd.Series],
]:
    """Split chronologically with target-aware gaps between every partition."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a test set")

    n = len(features)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    validation_start = train_end + purge_samples
    test_start = validation_end + purge_samples

    if validation_start >= validation_end or test_start >= n:
        raise ValueError("dataset is too small for the requested purged split")

    train_x = features.iloc[:train_end]
    validation_x = features.iloc[validation_start:validation_end]
    test_x = features.iloc[test_start:]

    def cut(start: int, end: int | None) -> dict[int, pd.Series]:
        return {h: target.iloc[start:end] for h, target in targets.items()}

    return (
        train_x,
        cut(0, train_end),
        validation_x,
        cut(validation_start, validation_end),
        test_x,
        cut(test_start, None),
    )


def _persistence_amplitude(
    residual: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    amplitude_window_samples: int,
) -> np.ndarray:
    """Predict future amplitude by persistence of the current causal window."""
    current = (
        residual.rolling(
            amplitude_window_samples,
            min_periods=amplitude_window_samples,
        ).max()
        - residual.rolling(
            amplitude_window_samples,
            min_periods=amplitude_window_samples,
        ).min()
    )
    return current.reindex(target_index).to_numpy(dtype=float)


def _print_metrics(
    name: str,
    model: GeomagneticForecaster,
    features: pd.DataFrame,
    targets: dict[int, pd.Series],
    baseline: np.ndarray,
) -> dict[int, float]:
    """Print model-vs-baseline metrics and return MAE improvements."""
    metrics = model.evaluate(features, targets)
    improvements: dict[int, float] = {}
    print(f"\n{name}:")
    for horizon, evaluation in metrics.items():
        target = targets[horizon].to_numpy(dtype=float)
        valid = np.isfinite(target) & np.isfinite(baseline)
        if valid.sum() < 100:
            raise RuntimeError(
                f"not enough valid baseline samples for +{horizon}h"
            )
        baseline_mae = float(mean_absolute_error(target[valid], baseline[valid]))
        baseline_rmse = float(
            np.sqrt(mean_squared_error(target[valid], baseline[valid]))
        )
        improvement = (
            100.0 * (baseline_mae - evaluation.mae_nt) / baseline_mae
            if baseline_mae > 0
            else 0.0
        )
        improvements[horizon] = improvement
        print(
            f"  +{horizon}h: ML MAE={evaluation.mae_nt:.2f} nT, "
            f"RMSE={evaluation.rmse_nt:.2f} nT; "
            f"persistence MAE={baseline_mae:.2f} nT, "
            f"RMSE={baseline_rmse:.2f} nT; "
            f"MAE improvement={improvement:+.1f}%; "
            f"precision={evaluation.precision:.3f}, "
            f"recall={evaluation.recall:.3f}, F1={evaluation.f1:.3f}"
        )
    return improvements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--column", default="x_nt")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-nonbeating",
        action="store_true",
        help="publish a candidate artifact even when ML does not beat persistence",
    )
    args = parser.parse_args()

    if args.days < 90:
        parser.error("use at least 90 historical days for production training")

    mag, kp, dst, analysis_start = fetch_inputs(
        args.observatory, args.start_date, args.days, args.warmup_days
    )

    if args.column not in mag.columns:
        parser.error(
            f"magnetometer column {args.column!r} is unavailable; "
            f"available columns: {list(mag.columns)}"
        )

    result = md.run_analysis(
        mag[args.column].to_numpy(),
        60.0,
        label=f"{args.observatory} ML training {args.start_date}+{args.days}d",
        start_time=mag.index.min().to_pydatetime(),
        analysis_start_time=analysis_start.to_pydatetime(),
        dst_series=dst,
        kp_series=kp,
        observatory=args.observatory,
    )
    if result["status"] != "ok":
        print(
            "Training aborted: deterministic quality gate returned "
            f"{result['status']}",
            file=sys.stderr,
        )
        return 2

    index = pd.date_range(
        analysis_start,
        periods=len(result["residual"]),
        freq="min",
        tz="UTC",
    )
    residual = pd.Series(
        np.asarray(result["residual"], dtype=float), index=index
    )
    kp_aligned = kp.reindex(index, method="ffill") if kp is not None else None
    dst_aligned = dst.reindex(index, method="ffill") if dst is not None else None

    config = ForecastConfig(
        minor_storm_nt=md.FLAG_THRESHOLD_MINOR_STORM_NT,
        major_storm_nt=md.FLAG_THRESHOLD_MAJOR_STORM_NT,
        severe_storm_nt=md.FLAG_THRESHOLD_SEVERE_STORM_NT,
    )

    features, targets = build_training_data(
        residual, kp_aligned, dst_aligned, config=config
    )
    valid_rows = features.notna().any(axis=1)
    features = features.loc[valid_rows]
    targets = {h: target.loc[features.index] for h, target in targets.items()}

    max_target_reach_hours = (
        max(config.horizons_hours) + config.amplitude_window_min / 60.0
    )
    purge_samples = int(round(max_target_reach_hours * 3600.0 / 60.0))

    (
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        test_features,
        test_targets,
    ) = _purged_three_way_split(
        features,
        targets,
        purge_samples=purge_samples,
    )

    print(
        f"Training samples: {len(train_features)}; "
        f"validation: {len(validation_features)}; "
        f"test: {len(test_features)}; "
        f"purge: {purge_samples} samples ({max_target_reach_hours:.1f}h)"
    )

    validation_model = GeomagneticForecaster(config).fit(
        train_features, train_targets
    )
    amplitude_window_samples = int(
        round(config.amplitude_window_min * 60.0 / 60.0)
    )
    validation_baseline = _persistence_amplitude(
        residual,
        validation_features.index,
        amplitude_window_samples=amplitude_window_samples,
    )
    _print_metrics(
        "Chronological validation",
        validation_model,
        validation_features,
        validation_targets,
        validation_baseline,
    )

    # Fit only on observations strictly before the final test period. The
    # target-aware purge is retained by starting the test after test_start.
    pre_test_end = test_features.index[0]
    final_train_features = features.loc[features.index < pre_test_end]
    final_train_targets = {
        h: target.loc[final_train_features.index] for h, target in targets.items()
    }
    final_model = GeomagneticForecaster(config).fit(
        final_train_features, final_train_targets
    )

    test_baseline = _persistence_amplitude(
        residual,
        test_features.index,
        amplitude_window_samples=amplitude_window_samples,
    )
    improvements = _print_metrics(
        "Final chronological test",
        final_model,
        test_features,
        test_targets,
        test_baseline,
    )

    failed = [h for h, value in improvements.items() if value <= 0.0]
    if failed and not args.allow_nonbeating:
        print(
            "\nPRODUCTION GATE FAILED: ML did not beat the persistence "
            f"baseline at horizons {failed}. No production artifact was saved.",
            file=sys.stderr,
        )
        return 3

    output = Path(
        args.output
        or f"models/artifacts/{args.observatory.lower()}_forecaster.pkl"
    )
    final_model.training_metadata.update(
        {
            "production_gate": "passed" if not failed else "overridden",
            "test_mae_improvement_percent": {
                str(h): float(v) for h, v in improvements.items()
            },
            "purge_hours": max_target_reach_hours,
            "data_start": analysis_start.isoformat(),
            "data_end": index[-1].isoformat(),
            "index_availability": {
                "kp": kp is not None,
                "dst": dst is not None,
            },
        }
    )
    save_model(final_model, output)
    print(f"\nSaved production artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
