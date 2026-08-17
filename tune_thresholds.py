#!/usr/bin/env python3
"""Diagnose and tune the activity classification against Kp/Dst.

Fits the baseline once per period, then evaluates metrics for many candidate
classification settings (thresholds + optional residual smoothing + anomaly
handling), since none of those affect the baseline fit.
"""

import argparse
import itertools
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import magnetometer_demo as md

md.setup_logging(level=40)

LEVELS = md.MetricsEngine.LOCAL_LEVELS


def load_period(observatory: str, start: str, days: int, warmup: float, column: str):
    """Return (residual, kp_aligned, dst_aligned) over the analysis window."""
    baseline_start = (
        pd.to_datetime(start, utc=True) - pd.Timedelta(days=warmup)
    ).strftime("%Y-%m-%d")
    total_days = int(days + warmup)
    text = md.fetch_intermagnet_iaga2002(
        observatory=observatory, start_date=baseline_start, duration_days=total_days
    )
    df = md.parse_iaga2002_to_dataframe(text)
    x = md.handle_gaps(df[column], max_gap_samples=md.MAX_GAP_SAMPLES).values

    start_dt = pd.to_datetime(df.index.min())
    end_dt = pd.to_datetime(df.index.max())
    kp = None
    try:
        kp = md.fetch_kp_gfz(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    except Exception as e:  # pragma: no cover - network
        print(f"  Kp unavailable: {e}")
    dst = None
    months = sorted(
        {(d.year, d.month) for d in pd.date_range(start_dt, end_dt, freq="D")}
    )
    parts = [p for p in (md.fetch_dst_kyoto(y, m) for y, m in months) if p is not None]
    if parts:
        dst = pd.concat(parts).sort_index()

    result = md.run_analysis(
        x,
        60,
        label=f"{observatory} {start}",
        start_time=start_dt.to_pydatetime(),
        analysis_start_time=(start_dt + pd.Timedelta(days=warmup)).to_pydatetime(),
        kp_series=kp,
        dst_series=dst,
    )
    residual = result["residual"]
    n_full = len(x)
    index = pd.date_range(start=start_dt, periods=n_full, freq="60s")
    offset = n_full - len(residual)
    idx_a = index[offset:]

    def align(series):
        if series is None:
            return np.full(len(residual), np.nan)
        return series.reindex(
            idx_a, method="ffill", tolerance=pd.Timedelta("3h")
        ).values

    return residual, align(kp), align(dst), result["metrics"]


def disturbance_amplitude(residual: np.ndarray, window: int, mode: str) -> np.ndarray:
    """Amplitude of disturbance over a rolling window, in nT.

    Kp is a 3-hourly *range* index and Dst an hourly average, so comparing a
    single minute's residual against them is the main source of both misses and
    false alarms. Modes: instantaneous |residual| (window 1), max |residual|,
    peak-to-peak range, or a robust percentile. A ``_causal`` suffix uses a
    trailing window (no lookahead), suitable for live monitoring.
    """
    if window <= 1:
        return np.abs(residual)
    causal = mode.endswith("_causal")
    base = mode[: -len("_causal")] if causal else mode
    s = pd.Series(residual)
    roll = s.rolling(window, center=not causal, min_periods=1)
    if base == "max":
        amp = roll.max().abs().combine(roll.min().abs(), max)
    elif base == "range":
        amp = roll.max() - roll.min()
    elif base == "p95":
        amp = s.abs().rolling(window, center=not causal, min_periods=1).quantile(0.95)
    else:
        raise ValueError(f"unknown mode {mode}")
    out = amp.to_numpy(dtype=float)
    out[~np.isfinite(residual)] = np.nan
    return out


def classify(a: np.ndarray, cfg: dict) -> np.ndarray:
    """Local activity levels 0..4 (NaN where invalid) for precomputed amplitude."""
    levels = np.full(len(a), np.nan)
    finite = np.isfinite(a)
    levels[finite] = 0.0
    for level, key in (
        (1.0, "unsettled"),
        (2.0, "active"),
        (3.0, "minor"),
        (4.0, "major"),
    ):
        levels[finite & (a >= cfg[key])] = level
    return levels


def evaluate(levels: np.ndarray, kp: np.ndarray, dst: np.ndarray) -> dict:
    g = md.MetricsEngine._global_levels(kp, dst)
    has_g = np.isfinite(g)
    has_l = np.isfinite(levels)
    both = has_g & has_l

    truth_storm = has_g & (g >= 3)
    pred_storm = has_l & (levels >= 3)
    tp = int(np.sum(pred_storm & truth_storm))
    fn = int(np.sum(~pred_storm & truth_storm))
    fp = int(np.sum(pred_storm & has_g & (g < 3)))
    tn = int(np.sum(~pred_storm & has_g & (g < 3)))
    missed = int(np.sum(has_g & (g >= 3) & has_l & (levels < 2)))
    under = int(np.sum(has_g & (g >= 2) & has_l & (levels < 2)))
    n_active = int(np.sum(has_g & (g >= 2)))

    return {
        "storm_detection_rate": tp / (tp + fn) if tp + fn else np.nan,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
        "missed_global_event_rate": missed / max(1, int(np.sum(truth_storm))),
        "under_reacting_rate": under / n_active if n_active else np.nan,
        "unconfirmed_storm_rate": fp / max(1, int(np.sum(pred_storm))),
        "mean_abs_level_error": (
            float(np.mean(np.abs(levels[both] - g[both]))) if np.any(both) else np.nan
        ),
    }


def prefilter(residual: np.ndarray, med: int) -> np.ndarray:
    """Median-filter the residual to drop single-sample artifacts."""
    if med <= 1:
        return residual
    return (
        pd.Series(residual).rolling(med, center=True, min_periods=1).median().to_numpy()
    )


TRAIN = [
    ("storm", "2024-05-08", 5),
    ("storm", "2023-04-22", 4),
    ("storm", "2024-03-23", 4),
    ("storm", "2024-10-09", 4),
    ("quiet", "2024-04-01", 7),
    ("quiet", "2024-06-10", 5),
]
HOLDOUT = [
    ("storm", "2024-08-11", 4),
    ("storm", "2021-11-03", 4),
    ("quiet", "2024-02-05", 5),
    ("quiet", "2023-12-01", 5),
]


def aggregate(evals):
    """Mean + worst case of each metric across periods."""
    storm = [e for kind, e in evals if kind == "storm"]
    allp = [e for _, e in evals]

    def m(key, subset):
        vals = [e[key] for e in subset if np.isfinite(e[key])]
        return (
            (float(np.mean(vals)), float(np.max(vals)), float(np.min(vals)))
            if vals
            else (np.nan,) * 3
        )

    sdr_mean, _, sdr_worst = m("storm_detection_rate", storm)
    far_mean, far_worst, _ = m("false_alarm_rate", allp)
    missed_mean, missed_worst, _ = m("missed_global_event_rate", storm)
    under_mean, under_worst, _ = m("under_reacting_rate", allp)
    mae_mean, mae_worst, _ = m("mean_abs_level_error", allp)
    unc_mean, _, _ = m("unconfirmed_storm_rate", storm)
    return {
        "sdr_mean": sdr_mean,
        "sdr_worst": sdr_worst,
        "far_mean": far_mean,
        "far_worst": far_worst,
        "missed_mean": missed_mean,
        "missed_worst": missed_worst,
        "under_mean": under_mean,
        "under_worst": under_worst,
        "mae_mean": mae_mean,
        "mae_worst": mae_worst,
        "unconfirmed_mean": unc_mean,
    }


def robust_score(a: dict) -> float:
    """Lower is better. Penalizes the worst period, not just the average."""
    return (
        10.0 * max(0.0, 0.90 - a["sdr_mean"])
        + 5.0 * max(0.0, 0.90 - a["sdr_worst"])
        + 1.0 * a["far_mean"]
        + 0.5 * a["far_worst"]
        + 1.0 * a["missed_mean"]
        + 0.5 * a["missed_worst"]
        + 1.0 * a["under_mean"]
        + 0.5 * a["mae_mean"]
        + 0.25 * a["mae_worst"]
    )


DEFAULT_CFG = {
    "med": 1,
    "smooth": 1,
    "mode": "max",
    "unsettled": md.FLAG_THRESHOLD_UNSETTLED_NT,
    "active": md.FLAG_THRESHOLD_ACTIVE_NT,
    "minor": md.FLAG_THRESHOLD_MINOR_STORM_NT,
    "major": md.FLAG_THRESHOLD_MAJOR_STORM_NT,
}


def score_cfg(cfg, periods, amp_cache):
    evals = []
    for kind, key in periods:
        amp = amp_cache[(cfg["med"], cfg["smooth"], cfg["mode"], key)]
        evals.append((kind, evaluate(classify(amp, cfg), *_INDEX[key])))
    return aggregate(evals), evals


_INDEX = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observatory", default="VIC")
    ap.add_argument("--column", default="x_nt")
    ap.add_argument("--warmup-days", type=float, default=3)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--output", default="tuning_sweep.csv")
    args = ap.parse_args()

    residuals = {}
    for kind, start, days in TRAIN + HOLDOUT:
        print(f"Loading {kind} {start} (+{days}d)...")
        res, kp, dst, default_metrics = load_period(
            args.observatory, start, days, args.warmup_days, args.column
        )
        residuals[start] = res
        _INDEX[start] = (kp, dst)
        if start == "2024-05-08":
            print("\nDefault pipeline metrics on 2024-05-08 (reference):")
            for k, v in default_metrics.items():
                print(f"  {k:28s} {v}")

    train_periods = [(kind, start) for kind, start, _ in TRAIN]
    hold_periods = [(kind, start) for kind, start, _ in HOLDOUT]

    grid_med = [1, 5]
    grid_smooth = [181, 241, 301, 361]
    grid_mode = ["range", "range_causal"]
    grid_unsettled = [15, 18, 20, 25]
    grid_active = [30, 35, 40, 50]
    grid_minor = [100, 110, 120, 130, 140, 150]
    grid_major = [300, 350, 400, 450]

    amp_cache = {}
    for med, smooth, mode in itertools.product(grid_med, grid_smooth, grid_mode):
        for start, res in residuals.items():
            amp_cache[(med, smooth, mode, start)] = disturbance_amplitude(
                prefilter(res, med), smooth, mode
            )
    for start, res in residuals.items():
        amp_cache[(1, 1, "max", start)] = disturbance_amplitude(res, 1, "max")

    rows = []
    for med, smooth, mode, u, ac, mi, ma in itertools.product(
        grid_med,
        grid_smooth,
        grid_mode,
        grid_unsettled,
        grid_active,
        grid_minor,
        grid_major,
    ):
        if not (u < ac < mi < ma):
            continue
        cfg = {
            "med": med,
            "smooth": smooth,
            "mode": mode,
            "unsettled": u,
            "active": ac,
            "minor": mi,
            "major": ma,
        }
        agg, _ = score_cfg(cfg, train_periods, amp_cache)
        rows.append({**cfg, **agg, "score": robust_score(agg)})

    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(
        f"\nEvaluated {len(df)} configs on {len(train_periods)} training periods -> {args.output}"
    )

    cols = [
        "med",
        "smooth",
        "mode",
        "unsettled",
        "active",
        "minor",
        "major",
        "sdr_mean",
        "sdr_worst",
        "far_mean",
        "far_worst",
        "missed_mean",
        "under_mean",
        "mae_mean",
        "score",
    ]
    fmt = lambda v: f"{v:.4f}"
    print("\nTOP TRAINING CONFIGS:")
    print(df[cols].head(args.top_n).to_string(index=False, float_format=fmt))

    print("\n=== HELD-OUT SCORES ===")
    candidates = [DEFAULT_CFG] + df.head(5).to_dict("records")
    for cfg in candidates:
        agg_h, evals_h = score_cfg(cfg, hold_periods, amp_cache)
        agg_t, evals_t = score_cfg(cfg, train_periods, amp_cache)
        tag = (
            f"med={cfg['med']} smooth={cfg['smooth']} mode={cfg['mode']} "
            f"thr={cfg['unsettled']}/{cfg['active']}/{cfg['minor']}/{cfg['major']}"
        )
        print(f"\n-- {tag}")
        for label, evals in (("train", evals_t), ("holdout", evals_h)):
            per = pd.DataFrame(
                [
                    {"set": label, "period": f"{kind} {p}", **e}
                    for (kind, p), (_, e) in zip(
                        train_periods if label == "train" else hold_periods, evals
                    )
                ]
            )
            print(per.to_string(index=False, float_format=fmt))
        print(
            f"   train score={robust_score(agg_t):.4f}  holdout score={robust_score(agg_h):.4f}"
        )


if __name__ == "__main__":
    main()
