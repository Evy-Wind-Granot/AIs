#!/usr/bin/env python3
"""Run the modular magnetometer pipeline once against real data."""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from magnetometer import pipeline
from magnetometer.acquisition import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz
from magnetometer.parsing import parse_iaga2002_to_dataframe


def fetch_window(observatory: str, start_date: str, days: int, warmup_days: float):
    start_dt = pd.to_datetime(start_date, utc=True)
    fetch_start = (start_dt - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    total_days = int(days + warmup_days)
    end_guess = (start_dt + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    with ThreadPoolExecutor(max_workers=3) as pool:
        mag_future = pool.submit(fetch_intermagnet_iaga2002, observatory, fetch_start, total_days)
        kp_future = pool.submit(fetch_kp_gfz, fetch_start, end_guess)
        mag_text = mag_future.result()
        frame = parse_iaga2002_to_dataframe(mag_text)
        if frame.empty:
            raise RuntimeError(f"No magnetometer data returned for {observatory}")
        dst_parts = list(pool.map(fetch_dst_kyoto, sorted({(ts.year, ts.month) for ts in frame.index})))
    try:
        kp = kp_future.result()
    except Exception as exc:
        print(f"(Kp unavailable: {exc})", file=sys.stderr)
        kp = None
    dst_parts = [part for part in dst_parts if part is not None and not part.empty]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    return frame, start_dt, kp, dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observatory", default="VIC")
    parser.add_argument("--start-date", default="2024-05-08")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--warmup-days", type=float, default=3)
    parser.add_argument("--column", default="x_nt")
    parser.add_argument("--quiet-log", action="store_true")
    args = parser.parse_args()

    # The modular pipeline owns logging; configure it through its CLI helper.
    from magnetometer.pipeline.cli import setup_logging
    setup_logging(level=logging.WARNING if args.quiet_log else logging.INFO)

    frame, start_dt, kp, dst = fetch_window(args.observatory, args.start_date, args.days, args.warmup_days)
    if args.column not in frame.columns:
        raise RuntimeError(f"Column {args.column!r} is not present in {list(frame.columns)}")

    result = pipeline.run_analysis(
        frame[args.column].to_numpy(),
        60.0,
        label=f"{args.observatory} {args.start_date}",
        start_time=pd.to_datetime(frame.index.min(), utc=True).to_pydatetime(),
        analysis_start_time=start_dt.to_pydatetime(),
        dst_series=dst,
        kp_series=kp,
        observatory=args.observatory,
    )
    if result["status"] != "ok":
        print(f"\nRun did not pass the data-quality gate: status={result['status']}")
        return 1

    print(f"\n=== {args.observatory} {args.start_date} (+{args.days}d) ===")
    print("\nValidation metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key:28s} {value:.4f}" if isinstance(value, float) else f"  {key:28s} {value}")
    print("\nActivity flag breakdown:")
    for level, count in result["flag_counts"].items():
        print(f"  {level:12s} {count}")
    health = result.get("health", {})
    print(f"\nHealth: {'OK' if health.get('healthy') else 'issues found'}")
    for check, ok in (health.get("checks") or {}).items():
        print(f"  {check:24s} {'ok' if ok else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
