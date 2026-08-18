#!/usr/bin/env python3
"""Run the deterministic magnetometer pipeline with optional ML forecasting.

This wrapper composes the existing deterministic QDC/Harmonic implementation.
Deterministic current classification remains authoritative; the ML model is a
forward-looking advisory layer.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hybrid_inference import build_aligned_forecast_frame, hybrid_status_payload  # noqa: E402
from magnetometer_demo import (  # noqa: E402
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    handle_gaps,
    parse_iaga2002_to_dataframe,
    run_analysis,
)
from models.forecaster import GeomagneticForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magnetometer_hybrid")


def _fetch_dst(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """Fetch Dst or return None when no usable reference data is available."""
    parts: list[pd.Series] = []
    months = pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M")
    for period in months:
        value = fetch_dst_kyoto(int(period.year), int(period.month))
        if value is not None and not value.empty:
            parts.append(value)
    if not parts:
        return None
    result = pd.concat(parts).sort_index()
    result.index = pd.to_datetime(result.index, utc=True)
    return result.astype(float)


def run_hybrid(
    observatory: str,
    start_date: str,
    days: int,
    model_path: str | None,
    column: str = "f_nt",
) -> Dict[str, Any]:
    if days <= 0:
        raise ValueError("days must be positive")

    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=start_date,
        duration_days=days,
        samples_per_day="Minute",
    )
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty:
        raise RuntimeError("INTERMAGNET returned no samples.")
    if column not in df.columns:
        raise ValueError(f"Requested magnetometer column {column!r} is unavailable.")

    series = handle_gaps(df[column], max_gap_samples=3)
    valid_samples = int(series.notna().sum())
    if valid_samples == 0:
        raise RuntimeError(
            f"INTERMAGNET returned {len(series)} timestamps but zero valid samples in {column!r}; "
            "refusing to run QDC/ML inference on an all-missing window."
        )

    completeness = valid_samples / max(1, len(series))
    if completeness < 0.50:
        raise RuntimeError(
            f"Insufficient {column} data for live inference: {valid_samples}/{len(series)} "
            f"({completeness:.1%}) valid samples."
        )

    cadence = df.index.to_series().diff().dropna().dt.total_seconds().median()
    cadence_s = float(cadence) if np.isfinite(cadence) and cadence > 0 else 60.0

    start_dt = pd.to_datetime(df.index.min(), utc=True)
    end_dt = pd.to_datetime(df.index.max(), utc=True)
    kp = fetch_kp_gfz(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    dst = _fetch_dst(start_dt, end_dt)

    analysis = run_analysis(
        series.to_numpy(dtype=float),
        cadence_s,
        label=f"INTERMAGNET {observatory}",
        start_time=start_dt.to_pydatetime(),
        dst_series=dst,
        kp_series=kp,
    )
    flags = analysis["flags"]
    deterministic_tier = str(flags[-1]) if len(flags) else "unknown"

    payload: Dict[str, Any] = {
        "realtime": {
            "tier": deterministic_tier,
            "observatory": observatory,
            "window_start": df.index[0].isoformat(),
            "window_end": df.index[-1].isoformat(),
            "data_completeness": completeness,
        },
        "forecast": None,
        "hybrid": {"enabled": False, "error": None},
    }

    if model_path:
        try:
            model = GeomagneticForecaster.load_model(model_path)
            frame = build_aligned_forecast_frame(
                analysis["residual"],
                df.index,
                kp_series=kp,
                dst_series=dst,
            )
            hybrid = hybrid_status_payload(
                frame,
                deterministic_tier=deterministic_tier,
                forecaster=model,
                cadence_s=cadence_s,
            )
            payload.update({
                "forecast": hybrid["forecast"],
                "hybrid": {**hybrid["hybrid"], "enabled": True, "error": None},
            })
        except Exception as exc:
            # ML is advisory. A model failure must never suppress current
            # deterministic status or turn a healthy monitor into a hard fault.
            logger.exception("ML forecasting failed; returning deterministic status only.")
            payload["hybrid"] = {"enabled": False, "error": str(exc), "degraded": True}

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid deterministic + ML magnetometer monitor")
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", default="2024-03-15")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--column", default="f_nt")
    ap.add_argument("--model-path", default=None, help="Path prefix for .joblib/.json forecaster artifact")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        idx = pd.date_range("2025-01-01", periods=1200, freq="min", tz="UTC")
        base = 50000.0 + 20.0 * np.sin(np.arange(len(idx)) / 100.0)
        base[800:950] += 80.0
        df = pd.DataFrame({"f_nt": base}, index=idx)
        analysis = run_analysis(df["f_nt"].to_numpy(), 60.0, label="synthetic", start_time=idx[0].to_pydatetime())
        print(json.dumps({"deterministic_final_tier": str(analysis["flags"][-1]), "samples": len(idx)}, indent=2))
        return

    result = run_hybrid(args.observatory.upper(), args.start_date, args.days, args.model_path, args.column)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
