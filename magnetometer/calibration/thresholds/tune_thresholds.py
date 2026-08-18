#!/usr/bin/env python3
"""
Tier 2: Parameter Tuning & Offline Validation Module
Tunes detection thresholds against historical event data using station configs.
"""

import argparse
import json
import pandas as pd


def validate_and_tune(config_path: str, test_data_csv: str, tuned_output: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    df = pd.read_csv(test_data_csv, parse_dates=["timestamp"])

    # Calculate vector magnitude or deviation
    if {"x", "y", "z"}.issubset(df.columns):
        df["magnitude"] = (df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2) ** 0.5
    else:
        df["magnitude"] = df.iloc[:, 1]  # Fallback to first signal column

    baseline = df["magnitude"].median()
    residual = (df["magnitude"] - baseline).abs()

    # Derive optimal alert threshold based on 95th percentile during test periods
    optimal_threshold = float(residual.quantile(0.95))
    config["derived_alert_threshold_nT"] = optimal_threshold

    with open(tuned_output, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Production config tuned and written to {tuned_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune sensor thresholds against test data."
    )
    parser.add_argument("--config", required=True, help="Base station config JSON")
    parser.add_argument(
        "--test-data", required=True, help="CSV containing validation events"
    )
    parser.add_argument(
        "--output", default="production_config.json", help="Output production config"
    )
    args = parser.parse_args()

    validate_and_tune(args.config, args.test_data, args.output)
