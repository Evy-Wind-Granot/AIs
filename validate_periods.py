#!/usr/bin/env python3
"""Score the production classifier over several storm/quiet periods.

Each period is fetched with the same warmup + cross-check settings the CLI
uses, so the numbers reported here are exactly what `magnetometer_demo.py`
would print for that window. Held-out periods are the ones the thresholds
were *not* tuned on.

    python3 validate_periods.py [--centered]
"""

import argparse
import logging
import sys

import pandas as pd

import magnetometer_demo as md

PERIODS = [
    # (label, start_date, days, kind, tuned_on)
    ("2024-05-08 Gannon storm", "2024-05-08", 5, "storm", True),
    ("2024-03-24 storm", "2024-03-24", 3, "storm", True),
    ("2023-04-23 storm", "2023-04-23", 3, "storm", True),
    ("2024-01-01 quiet", "2024-01-01", 5, "quiet", True),
    ("2024-08-11 storm", "2024-08-11", 3, "storm", False),
    ("2021-11-03 storm", "2021-11-03", 3, "storm", False),
    ("2024-10-10 storm", "2024-10-10", 3, "storm", False),
    ("2019-06-01 quiet", "2019-06-01", 5, "quiet", False),
]

WARMUP_DAYS = 3


def score(start_date: str, days: int, observatory: str = "VIC") -> dict:
    start_dt = pd.to_datetime(start_date, utc=True)
    fetch_start = (start_dt - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    total_days = days + WARMUP_DAYS
    df = md.parse_iaga2002_to_dataframe(
        md.fetch_intermagnet_iaga2002(
            observatory=observatory,
            start_date=fetch_start,
            duration_days=total_days,
        )
    )
    if df is None or df.empty:
        raise RuntimeError("no magnetometer data")

    kp = md.fetch_kp_gfz(
        pd.to_datetime(df.index.min()).strftime("%Y-%m-%d"),
        pd.to_datetime(df.index.max()).strftime("%Y-%m-%d"),
    )
    months = sorted({(t.year, t.month) for t in df.index})
    dst_parts = [md.fetch_dst_kyoto(y, m) for y, m in months]
    dst_parts = [p for p in dst_parts if p is not None]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None

    res = md.run_analysis(
        df["x_nt"].to_numpy(),
        60,
        label=f"{observatory} {start_date}",
        start_time=pd.to_datetime(df.index.min()).to_pydatetime(),
        analysis_start_time=start_dt.to_pydatetime(),
        dst_series=dst,
        kp_series=kp,
    )
    if res["status"] != "ok":
        raise RuntimeError(f"analysis status={res['status']}")
    return res["metrics"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centered", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--unsettled", type=float, default=None)
    ap.add_argument("--active", type=float, default=None)
    ap.add_argument("--minor", type=float, default=None)
    ap.add_argument("--major", type=float, default=None)
    ap.add_argument("--severe", type=float, default=None)
    ap.add_argument("--mode", default=None)
    args = ap.parse_args()

    md.setup_logging(level=logging.ERROR)
    if args.centered:
        md.FLAG_AMPLITUDE_CENTERED = True
    if args.legacy:
        for name, value in md.LEGACY_FLAG_SETTINGS.items():
            setattr(md, name, value)
    for attr, value in (
        ("FLAG_AMPLITUDE_WINDOW_MIN", args.window),
        ("FLAG_THRESHOLD_UNSETTLED_NT", args.unsettled),
        ("FLAG_THRESHOLD_ACTIVE_NT", args.active),
        ("FLAG_THRESHOLD_MINOR_STORM_NT", args.minor),
        ("FLAG_THRESHOLD_MAJOR_STORM_NT", args.major),
        ("FLAG_THRESHOLD_SEVERE_STORM_NT", args.severe),
        ("FLAG_AMPLITUDE_MODE", args.mode),
    ):
        if value is not None:
            setattr(md, attr, value)

    cols = [
        "storm_detection_rate",
        "false_alarm_rate",
        "missed_global_event_rate",
        "under_reacting_rate",
        "unconfirmed_storm_rate",
        "mean_abs_level_error",
    ]
    header = f"{'period':26s} {'set':6s} " + " ".join(f"{c[:9]:>9s}" for c in cols)
    print(header)
    print("-" * len(header))
    rows = []
    for label, start, days, kind, tuned in PERIODS:
        try:
            m = score(start, days)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"{label:26s} {'train' if tuned else 'held':6s} FAILED: {e}")
            continue
        vals = [m.get(c, float("nan")) for c in cols]
        print(
            f"{label:26s} {'train' if tuned else 'held':6s} "
            + " ".join(f"{v:9.4f}" for v in vals)
        )
        rows.append((label, kind, tuned, dict(zip(cols, vals))))

    storms = [r for r in rows if r[1] == "storm"]
    if storms:
        sdr = [r[3]["storm_detection_rate"] for r in storms]
        print(
            f"\nstorm sdr: mean={sum(sdr)/len(sdr):.4f} worst={min(sdr):.4f} "
            f"(n={len(sdr)})"
        )
    quiets = [r for r in rows if r[1] == "quiet"]
    if quiets:
        far = [r[3]["false_alarm_rate"] for r in quiets]
        print(f"quiet far: mean={sum(far)/len(far):.4f} worst={max(far):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
