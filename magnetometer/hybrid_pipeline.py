#!/usr/bin/env python3
"""Hybrid deterministic + certified ML magnetometer monitor."""
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
from magnetometer_demo import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz, handle_gaps, parse_iaga2002_to_dataframe, run_analysis  # noqa: E402
from models.forecaster import GeomagneticForecaster  # noqa: E402
from state import merge_forecast_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magnetometer_hybrid")


def _fetch_dst(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    parts: list[pd.Series] = []
    for period in pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"):
        value = fetch_dst_kyoto(int(period.year), int(period.month))
        if value is not None and not value.empty:
            parts.append(value)
    if not parts:
        return None
    result = pd.concat(parts).sort_index()
    result.index = pd.to_datetime(result.index, utc=True)
    return result[~result.index.duplicated(keep="last")].astype(float)


def resolve_model_path(model_path: str | None, observatory: str) -> str | None:
    if not model_path:
        return None
    return model_path.replace("{observatory}", observatory.upper())


def run_hybrid(observatory: str, start_date: str, days: int, model_path: str | None, column: str = "f_nt", state_path: str | None = None) -> Dict[str, Any]:
    if days <= 0:
        raise ValueError("days must be positive")
    raw = fetch_intermagnet_iaga2002(observatory=observatory, start_date=start_date, duration_days=days, samples_per_day="Minute")
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty:
        raise RuntimeError("INTERMAGNET returned no samples.")
    if column not in df.columns:
        raise ValueError(f"Requested magnetometer column {column!r} is unavailable.")
    series = handle_gaps(df[column], max_gap_samples=3)
    valid_samples = int(series.notna().sum())
    if valid_samples == 0:
        raise RuntimeError(f"INTERMAGNET returned {len(series)} timestamps but zero valid samples in {column!r}.")
    completeness = valid_samples / max(1, len(series))
    if completeness < 0.50:
        raise RuntimeError(f"Insufficient {column} data: {valid_samples}/{len(series)} ({completeness:.1%}) valid samples.")

    cadence = df.index.to_series().diff().dropna().dt.total_seconds().median()
    cadence_s = float(cadence) if np.isfinite(cadence) and cadence > 0 else 60.0
    start_dt = pd.to_datetime(df.index.min(), utc=True)
    end_dt = pd.to_datetime(df.index.max(), utc=True)
    kp = fetch_kp_gfz(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    dst = _fetch_dst(start_dt, end_dt)
    analysis = run_analysis(series.to_numpy(dtype=float), cadence_s, label=f"INTERMAGNET {observatory}", start_time=start_dt.to_pydatetime(), dst_series=dst, kp_series=kp)
    deterministic_tier = str(analysis["flags"][-1]) if len(analysis["flags"]) else "unknown"

    payload: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "realtime": {"tier": deterministic_tier, "observatory": observatory, "window_start": df.index[0].isoformat(), "window_end": df.index[-1].isoformat(), "data_completeness": completeness, "valid_samples": valid_samples},
        "forecast": None,
        "hybrid": {"enabled": False, "error": None, "degraded": False},
    }
    resolved = resolve_model_path(model_path, observatory)
    if resolved:
        try:
            model = GeomagneticForecaster.load_model(resolved)
            frame = build_aligned_forecast_frame(analysis["residual"], df.index, kp_series=kp, dst_series=dst)
            hybrid = hybrid_status_payload(frame, deterministic_tier=deterministic_tier, forecaster=model, cadence_s=cadence_s)
            payload.update({"generated_at": hybrid["generated_at"], "forecast": hybrid["forecast"], "hybrid": {**hybrid["hybrid"], "enabled": True, "error": None, "degraded": False}, "model": hybrid["model"]})
        except Exception as exc:
            logger.exception("ML forecasting failed; returning deterministic status only.")
            payload["hybrid"] = {"enabled": False, "error": str(exc), "degraded": True}
    if state_path:
        merge_forecast_state(state_path, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid deterministic + certified ML magnetometer monitor")
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", default="2024-03-15")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--column", default="f_nt")
    ap.add_argument("--model-path", default="magnetometer/data/models/{observatory}/magnetometer_forecaster")
    ap.add_argument("--state-path", default=".magnetometer_state.json")
    ap.add_argument("--disable-ml", action="store_true")
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
    result = run_hybrid(args.observatory.upper(), args.start_date, args.days, None if args.disable_ml else args.model_path, args.column, args.state_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
