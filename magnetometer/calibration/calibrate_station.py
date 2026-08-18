#!/usr/bin/env python3
"""
Tier 1: Station Calibration Module
Generates station-specific YAML/JSON threshold configs from quiet historical data.
"""
import argparse
import json
import numpy as np
import pandas as pd


def calibrate(input_csv: str, station_id: str, output_config: str):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    
    # Calculate baseline statistics for raw components (e.g., X, Y, Z or H, D, Z in nT)
    stats = {}
    for col in ["x", "y", "z"]:
        if col in df.columns:
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            stats[col] = {
                "baseline_mean_nT": mean_val,
                "noise_floor_std_nT": std_val,
                "threshold_3sigma_nT": mean_val + (3 * std_val)
            }

    config = {
        "station_id": station_id,
        "sample_rate_hz": 1.0,
        "components": stats,
        "qdc_window_hours": 24,
        "anomaly_threshold_multiplier": 3.0
    }

    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Station calibration saved to {output_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate local magnetometer noise floor.")
    parser.add_argument("--input", required=True, help="Path to quiet-day CSV data")
    parser.add_argument("--station", required=True, help="Station identifier (e.g., SENSOR_01)")
    parser.add_argument("--output", default="sensor_config.json", help="Output config path")
    args = parser.parse_args()
    
    calibrate(args.input, args.station, args.output)