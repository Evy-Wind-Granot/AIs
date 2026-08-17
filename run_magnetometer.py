#!/usr/bin/env python3
"""Run the magnetometer pipeline once against real data and print the stats.

No config file, no --live loop, no JSON/Prometheus output, no state file
juggling. Just: fetch a window of real INTERMAGNET data (+ Kp/Dst for
cross-checking), run the classifier, print the numbers.

    python3 run_magnetometer.py --observatory VIC --start-date 2024-05-08 --days 5

Performance notes vs the old CLI invocations:
  - Kp and Dst are fetched concurrently with each other (ThreadPoolExecutor)
    instead of sequentially, cutting index-fetch wall time roughly in half.
  - The magnetometer series and the indices are fetched concurrently too.
  - HTTP_CACHE_ENABLED stays on (magnetometer_demo's default), so re-running
    the same window is instant instead of re-hitting INTERMAGNET/GFZ/Kyoto.
  - This never touches --live, so it never trips over clock skew between
    the local machine and the station feed, or the "freshness" checks that
    only make sense for a live monitor.
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import magnetometer_demo as md


def fetch_window(observatory: str, start_date: str, days: int, warmup_days: float):
    """Fetch magnetometer data + Kp/Dst concurrently and return aligned inputs."""
    start_dt = pd.to_datetime(start_date, utc=True)
    fetch_start = (start_dt - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    total_days = int(days + warmup_days)

    with ThreadPoolExecutor(max_workers=3) as pool:
        mag_future = pool.submit(
            md.fetch_intermagnet_iaga2002,
            observatory=observatory,
            start_date=fetch_start,
            duration_days=total_days,
        )
        # Kp needs the fetch window, which we already know without waiting on
        # the magnetometer download, so kick it off in parallel.
        end_guess = (start_dt + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        kp_future = pool.submit(md.fetch_kp_gfz, fetch_start, end_guess)

        mag_text = mag_future.result()
        df = md.parse_iaga2002_to_dataframe(mag_text)
        if df is None or df.empty:
            raise RuntimeError(f"No magnetometer data returned for {observatory}")

        months = sorted({(t.year, t.month) for t in df.index})
        dst_parts = list(pool.map(lambda ym: md.fetch_dst_kyoto(*ym), months))

    kp = None
    try:
        kp = kp_future.result()
    except Exception as e:
        print(f"  (Kp unavailable: {e})", file=sys.stderr)

    dst_parts = [p for p in dst_parts if p is not None]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None

    return df, start_dt, kp, dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--start-date", default="2024-05-08")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--warmup-days", type=float, default=3)
    ap.add_argument("--column", default="x_nt")
    ap.add_argument("--quiet-log", action="store_true", help="Suppress pipeline INFO logs")
    args = ap.parse_args()

    md.setup_logging(level=logging.WARNING if args.quiet_log else logging.INFO)

    df, start_dt, kp, dst = fetch_window(
        args.observatory, args.start_date, args.days, args.warmup_days
    )

    result = md.run_analysis(
        df[args.column].to_numpy(),
        60,
        label=f"{args.observatory} {args.start_date}",
        start_time=pd.to_datetime(df.index.min()).to_pydatetime(),
        analysis_start_time=start_dt.to_pydatetime(),
        dst_series=dst,
        kp_series=kp,
        observatory=args.observatory,
    )

    if result["status"] != "ok":
        print(f"\nRun did not pass the data-quality gate: status={result['status']}")
        print("Try a different window, or drop --warmup-days if data is sparse.")
        return 1

    print(f"\n=== {args.observatory} {args.start_date} (+{args.days}d) ===")
    print("\nValidation metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")

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
