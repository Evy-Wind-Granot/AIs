#!/usr/bin/env python3
"""Train and evaluate the short-horizon geomagnetic forecaster on real data.

Example:
    python train_magnetometer_forecaster.py --observatory VIC \
        --start-date 2024-05-08 --days 90

The final 20% is held out chronologically.  After evaluation, a fresh model is
fitted on the full period and serialized for live inference.
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


def fetch_inputs(observatory: str, start_date: str, days: int, warmup_days: int):
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
    dst_parts = [part for part in dst_parts if part is not None and not part.empty]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    return mag, kp_future.result(), dst, start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-05-08")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--column", default="x_nt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.days < 30:
        parser.error("use at least 30 historical days for a meaningful model")

    mag, kp, dst, analysis_start = fetch_inputs(
        args.observatory, args.start_date, args.days, args.warmup_days
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

    split = int(len(features) * 0.8)
    train_features = features.iloc[:split]
    test_features = features.iloc[split:]
    train_targets = {h: target.iloc[:split] for h, target in targets.items()}
    test_targets = {h: target.iloc[split:] for h, target in targets.items()}

    holdout_model = GeomagneticForecaster(config).fit(
        train_features, train_targets
    )
    metrics = holdout_model.evaluate(test_features, test_targets)
    current_amplitude = (
        residual.rolling(180, min_periods=180).max()
        - residual.rolling(180, min_periods=180).min()
    )
    baseline = current_amplitude.reindex(test_features.index).to_numpy()

    print("\nChronological holdout metrics:")
    for horizon, evaluation in metrics.items():
        target = test_targets[horizon].to_numpy(dtype=float)
        valid = np.isfinite(target) & np.isfinite(baseline)
        baseline_mae = float(mean_absolute_error(target[valid], baseline[valid]))
        baseline_rmse = float(
            np.sqrt(mean_squared_error(target[valid], baseline[valid]))
        )
        improvement = (
            100.0 * (baseline_mae - evaluation.mae_nt) / baseline_mae
            if baseline_mae > 0
            else 0.0
        )
        print(
            f"  +{horizon}h: ML MAE={evaluation.mae_nt:.2f} nT, "
            f"RMSE={evaluation.rmse_nt:.2f} nT; "
            f"persistence MAE={baseline_mae:.2f} nT, "
            f"RMSE={baseline_rmse:.2f} nT; "
            f"MAE improvement={improvement:+.1f}%; "
            f"precision={evaluation.precision:.3f}, "
            f"recall={evaluation.recall:.3f}, F1={evaluation.f1:.3f}"
        )

    output = Path(
        args.output
        or f"models/artifacts/{args.observatory.lower()}_forecaster.pkl"
    )
    final_model = GeomagneticForecaster(config).fit(features, targets)
    save_model(final_model, output)
    print(f"\nSaved production artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
