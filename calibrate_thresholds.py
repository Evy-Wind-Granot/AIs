#!/usr/bin/env python3
"""Calibrate the activity thresholds for one observatory and emit its config.

The production defaults in the magnetometer demo module were fitted at VIC. Local
disturbance amplitude scales with geomagnetic latitude and station noise, so
the same nT thresholds mean different things at CNB, ABK or HON — running the
pipeline elsewhere without recalibration is the single largest source of wrong
labels. This command produces a per-station thresholds file:

    python calibrate_observatory.py --observatory CNB \
        --storm 2024-05-08:5 --storm 2023-04-23:3 \
        --quiet 2024-01-01:5 \
        --out cnb_thresholds.yaml

Method
------
1. Quiet periods fix the two lower tiers: ``unsettled`` is the 99th percentile
   of the rolling disturbance amplitude on quiet data (so a quiet week is
   labelled quiet by construction), ``active`` a configurable multiple above.
2. Storm periods fix the three storm tiers by sweeping candidates and scoring
   them against Kp/Dst with the pipeline's own MetricsEngine — the same metric
   definitions the operational run reports, so a calibration number is
   comparable with a production number.
3. The winning configuration is scored again on any ``--holdout`` periods, run
   through ``validate_settings`` and only then written out, with the achieved
   metrics recorded in the file as comments.

Everything is computed with the pipeline's own ``disturbance_amplitude`` under
the requested window/mode, so the calibration cannot drift from production
behaviour. Causal (trailing) windows are used unless ``--centered`` is given.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from magnetometer.demos import magnetometer_demo as md  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("calibration")
md.setup_logging(level=logging.ERROR)

LEVEL_OF_TIER = {"unsettled": 1.0, "active": 2.0, "minor": 3.0, "major": 4.0}


class Period:
    """One labelled window of data, prepared for repeated scoring."""

    def __init__(self, kind: str, start: str, days: int):
        self.kind = kind
        self.start = start
        self.days = days
        self.residual: Optional[np.ndarray] = None
        self.kp: Optional[np.ndarray] = None
        self.dst: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f"{self.kind} {self.start} (+{self.days}d)"


def parse_period(spec: str, kind: str) -> Period:
    """Parse ``YYYY-MM-DD:DAYS``."""
    date, _, days = spec.partition(":")
    try:
        pd.to_datetime(date)
        n_days = int(days) if days else 3
    except Exception:
        raise SystemExit(f"Bad period {spec!r}; expected YYYY-MM-DD:DAYS")
    if n_days <= 0:
        raise SystemExit(f"Bad period {spec!r}; DAYS must be positive")
    return Period(kind, date, n_days)


def load_period(
    period: Period, observatory: str, warmup_days: float, column: str, cadence_s: int
) -> None:
    """Fetch the window and cache its residual and aligned global indices."""
    logger.info(f"Loading {period} for {observatory}...")
    baseline_start = (
        pd.to_datetime(period.start, utc=True) - pd.Timedelta(days=warmup_days)
    ).strftime("%Y-%m-%d")
    total_days = int(period.days + warmup_days)

    text = md.fetch_intermagnet_iaga2002(
        observatory=observatory, start_date=baseline_start, duration_days=total_days
    )
    df = md.parse_iaga2002_to_dataframe(text)
    if column not in df.columns:
        raise SystemExit(f"Column {column!r} not served for {observatory}")
    x = md.handle_gaps(df[column], max_gap_samples=md.MAX_GAP_SAMPLES).values

    start_dt = pd.to_datetime(df.index.min())
    end_dt = pd.to_datetime(df.index.max())

    kp = None
    try:
        kp = md.fetch_kp_gfz(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    except Exception as e:
        logger.warning(f"  Kp unavailable: {e}")
    months = sorted(
        {(d.year, d.month) for d in pd.date_range(start_dt, end_dt, freq="D")}
    )
    dst_parts = [
        p for p in (md.fetch_dst_kyoto(y, m) for y, m in months) if p is not None
    ]
    dst = pd.concat(dst_parts).sort_index() if dst_parts else None
    if kp is None and dst is None:
        raise SystemExit(
            f"Neither Kp nor Dst is available for {period}; cannot calibrate "
            f"against it. Choose another period."
        )

    result = md.run_analysis(
        x,
        cadence_s,
        label=f"{observatory} {period.start}",
        start_time=start_dt.to_pydatetime(),
        analysis_start_time=(start_dt + pd.Timedelta(days=warmup_days)).to_pydatetime(),
        kp_series=kp,
        dst_series=dst,
    )
    if result["status"] != "ok":
        raise SystemExit(
            f"{period} failed the data-quality gate "
            f"(coverage={result['coverage']:.1%}); pick a cleaner window."
        )

    residual = result["residual"]
    index = pd.date_range(
        start=start_dt, periods=len(x), freq=pd.Timedelta(seconds=cadence_s)
    )[len(x) - len(residual) :]

    def align(series: Optional[pd.Series], tolerance: str) -> np.ndarray:
        if series is None:
            return np.full(len(residual), np.nan)
        return series.reindex(
            index, method="ffill", tolerance=pd.Timedelta(tolerance)
        ).values.astype(float)

    period.residual = residual
    period.kp = align(kp, "3h")
    period.dst = align(dst, "1h")
    logger.info(
        f"  {len(residual)} samples, "
        f"{np.isfinite(period.kp).mean():.0%} Kp / "
        f"{np.isfinite(period.dst).mean():.0%} Dst coverage"
    )


def amplitude(period: Period, cadence_s: int) -> np.ndarray:
    """Disturbance amplitude under the currently configured window/mode."""
    assert period.residual is not None, "load_period() must run first"
    return md.disturbance_amplitude(period.residual, cadence_s)


def local_levels(amp: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Activity level 0..4 per sample; NaN where the amplitude is unusable."""
    levels = np.full(len(amp), np.nan)
    finite = np.isfinite(amp)
    levels[finite] = 0.0
    for tier, level in LEVEL_OF_TIER.items():
        levels[finite & (amp >= thresholds[tier])] = level
    return levels


def score(levels: np.ndarray, period: Period) -> Dict[str, float]:
    """Recall / false alarm / miss / under-reaction / MAE for one period.

    Deliberately expressed in terms of MetricsEngine's own global levels so
    calibration and production agree on what a storm is.
    """
    assert period.kp is not None and period.dst is not None
    g = md.MetricsEngine._global_levels(period.kp, period.dst)
    has_g, has_l = np.isfinite(g), np.isfinite(levels)
    both = has_g & has_l

    truth_storm = has_g & (g >= 3)
    pred_storm = has_l & (levels >= 3)
    tp = int(np.sum(pred_storm & truth_storm))
    fn = int(np.sum(~pred_storm & truth_storm))
    fp = int(np.sum(pred_storm & has_g & (g < 3)))
    tn = int(np.sum(~pred_storm & has_g & (g < 3)))
    n_storm = int(np.sum(truth_storm))
    n_active = int(np.sum(has_g & (g >= 2)))

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else np.nan

    return {
        "storm_detection_rate": ratio(tp, tp + fn),
        "false_alarm_rate": ratio(fp, fp + tn),
        "missed_global_event_rate": ratio(
            int(np.sum(truth_storm & has_l & (levels < 2))), n_storm
        ),
        "under_reacting_rate": ratio(
            int(np.sum(has_g & (g >= 2) & has_l & (levels < 2))), n_active
        ),
        "unconfirmed_storm_rate": ratio(fp, int(np.sum(pred_storm))),
        "mean_abs_level_error": (
            float(np.mean(np.abs(levels[both] - g[both]))) if np.any(both) else np.nan
        ),
    }


def aggregate(scores: List[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
    """Mean and worst case per metric, keeping storm and quiet roles apart."""
    storm = [s for kind, s in scores if kind == "storm"]
    every = [s for _, s in scores]

    def stat(
        key: str, subset: List[Dict[str, float]], worst: str
    ) -> Tuple[float, float]:
        vals = [s[key] for s in subset if np.isfinite(s[key])]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(min(vals) if worst == "min" else max(vals))

    sdr_mean, sdr_worst = stat("storm_detection_rate", storm, "min")
    far_mean, far_worst = stat("false_alarm_rate", every, "max")
    missed_mean, missed_worst = stat("missed_global_event_rate", storm, "max")
    under_mean, under_worst = stat("under_reacting_rate", every, "max")
    mae_mean, mae_worst = stat("mean_abs_level_error", every, "max")
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
    }


def objective(agg: Dict[str, float], target_recall: float) -> float:
    """Lower is better; the worst period is penalised, not just the mean.

    A configuration that averages the target recall by excelling on one storm
    and failing another is worse operationally than one that is uniformly close
    to it, hence the explicit worst-case terms.
    """

    def miss(value: float) -> float:
        return max(0.0, target_recall - value) if np.isfinite(value) else 1.0

    def val(value: float) -> float:
        return value if np.isfinite(value) else 1.0

    return (
        10.0 * miss(agg["sdr_mean"])
        + 5.0 * miss(agg["sdr_worst"])
        + 1.0 * val(agg["far_mean"])
        + 0.5 * val(agg["far_worst"])
        + 1.0 * val(agg["missed_mean"])
        + 1.0 * val(agg["under_mean"])
        + 0.5 * val(agg["mae_mean"])
        + 0.25 * val(agg["mae_worst"])
    )


def quiet_tiers(
    periods: List[Period], cadence_s: int, active_ratio: float, percentile: float
) -> Tuple[float, float]:
    """Derive unsettled/active from the amplitude distribution on quiet data."""
    quiet_amps = [amplitude(p, cadence_s) for p in periods if p.kind == "quiet"]
    if not quiet_amps:
        logger.warning(
            "No quiet period supplied; keeping the built-in lower tiers "
            "(these are VIC-derived and may not fit this station)."
        )
        return md.FLAG_THRESHOLD_UNSETTLED_NT, md.FLAG_THRESHOLD_ACTIVE_NT
    pooled = np.concatenate([a[np.isfinite(a)] for a in quiet_amps])
    unsettled = float(np.percentile(pooled, percentile))
    unsettled = max(1.0, round(unsettled, 1))
    active = round(unsettled * active_ratio, 1)
    logger.info(
        f"Quiet-derived tiers: unsettled={unsettled:g} nT "
        f"(p{percentile:g} of quiet amplitude), active={active:g} nT"
    )
    return unsettled, active


def sweep(
    periods: List[Period],
    cadence_s: int,
    unsettled: float,
    active: float,
    minor_grid: List[float],
    major_grid: List[float],
    severe_grid: List[float],
    target_recall: float,
) -> pd.DataFrame:
    """Score every ordered (minor, major, severe) candidate."""
    amps = [(p.kind, p, amplitude(p, cadence_s)) for p in periods]
    rows = []
    for minor, major, severe in itertools.product(minor_grid, major_grid, severe_grid):
        if not active < minor < major < severe:
            continue
        thresholds = {
            "unsettled": unsettled,
            "active": active,
            "minor": minor,
            "major": major,
        }
        scores = [
            (kind, score(local_levels(amp, thresholds), period))
            for kind, period, amp in amps
        ]
        agg = aggregate(scores)
        rows.append(
            {
                "minor": minor,
                "major": major,
                "severe": severe,
                **agg,
                "objective": objective(agg, target_recall),
            }
        )
    if not rows:
        raise SystemExit(
            "No candidate satisfies active < minor < major < severe; widen the grids."
        )
    return pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)


def render_config(
    observatory: str,
    thresholds: Dict[str, float],
    window_min: float,
    mode: str,
    centered: bool,
    train: List[Period],
    holdout: List[Period],
    train_agg: Dict[str, float],
    holdout_agg: Optional[Dict[str, float]],
) -> str:
    """Render a config file fragment, with the evidence behind it in comments."""

    def block(
        title: str, periods: List[Period], agg: Optional[Dict[str, float]]
    ) -> List[str]:
        if not periods or agg is None:
            return []
        lines = [f"#   {title}: " + ", ".join(str(p) for p in periods)]
        lines += [
            f"#     storm recall  mean {agg['sdr_mean']:.4f}  worst {agg['sdr_worst']:.4f}",
            f"#     false alarms  mean {agg['far_mean']:.4f}  worst {agg['far_worst']:.4f}",
            f"#     missed        mean {agg['missed_mean']:.4f}  "
            f"under-reacting mean {agg['under_mean']:.4f}",
            f"#     level MAE     mean {agg['mae_mean']:.4f}  worst {agg['mae_worst']:.4f}",
        ]
        return lines

    header = [
        f"# Activity thresholds calibrated for {observatory}.",
        f"# Generated by calibrate_observatory.py "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"(pipeline {md.__version__}).",
        "#",
        "# Thresholds are rolling disturbance *amplitudes* in nT, not per-sample",
        "# residuals: do not mix them with pre-2.0 threshold values.",
        "#",
        "# Measured performance of exactly these settings:",
    ]
    header += block("training periods", train, train_agg)
    header += block("held-out periods", holdout, holdout_agg)
    if not holdout:
        header.append(
            "#   No held-out periods were supplied, so these numbers are "
            "in-sample; re-run with --holdout before trusting them."
        )
    if centered:
        header.append(
            "#   WARNING: calibrated with a centered window, which uses future "
            "samples. Valid for re-analysis only, not live monitoring."
        )

    body = [
        "",
        "thresholds:",
        f"  amplitude_window_min: {window_min:g}",
        f"  amplitude_mode: {mode}",
        f"  amplitude_centered: {str(centered).lower()}",
        "",
        f"  unsettled: {thresholds['unsettled']:g}",
        f"  active: {thresholds['active']:g}",
        f"  minor_storm: {thresholds['minor']:g}",
        f"  major_storm: {thresholds['major']:g}",
        f"  severe_storm: {thresholds['severe']:g}",
        f"  anomaly_jump: {md.FLAG_THRESHOLD_ANOMALY_JUMP_NT:g}",
        "",
        "observatory:",
        f"  default: {observatory}",
        "",
    ]
    return "\n".join(header + body)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate activity thresholds for one observatory."
    )
    ap.add_argument("--observatory", required=True)
    ap.add_argument("--column", default="x_nt")
    ap.add_argument("--cadence-s", type=int, default=60)
    ap.add_argument("--warmup-days", type=float, default=3.0)
    ap.add_argument(
        "--storm",
        action="append",
        default=[],
        metavar="DATE:DAYS",
        help="Storm period to calibrate on (repeatable).",
    )
    ap.add_argument(
        "--quiet",
        action="append",
        default=[],
        metavar="DATE:DAYS",
        help="Quiet period; fixes the unsettled/active tiers.",
    )
    ap.add_argument(
        "--holdout",
        action="append",
        default=[],
        metavar="KIND:DATE:DAYS",
        help="Period scored but never fitted, e.g. storm:2024-08-11:3.",
    )
    ap.add_argument("--window-min", type=float, default=md.FLAG_AMPLITUDE_WINDOW_MIN)
    ap.add_argument(
        "--mode",
        choices=["range", "hybrid", "max", "instant"],
        default=md.FLAG_AMPLITUDE_MODE,
    )
    ap.add_argument(
        "--centered",
        action="store_true",
        help="Retrospective only: lets labels use future samples.",
    )
    ap.add_argument("--quiet-percentile", type=float, default=99.0)
    ap.add_argument("--active-ratio", type=float, default=1.5)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument(
        "--minor-grid", type=float, nargs="+", default=[60, 80, 100, 120, 140, 160]
    )
    ap.add_argument(
        "--major-grid", type=float, nargs="+", default=[250, 300, 350, 400, 450]
    )
    ap.add_argument(
        "--severe-grid", type=float, nargs="+", default=[600, 700, 800, 900]
    )
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument(
        "--sweep-csv", default=None, help="Also write every scored candidate here."
    )
    ap.add_argument("--out", required=True, help="Config file to write.")
    args = ap.parse_args()

    if not args.storm:
        raise SystemExit("At least one --storm period is required.")

    train = [parse_period(s, "storm") for s in args.storm]
    train += [parse_period(s, "quiet") for s in args.quiet]
    holdout = []
    for spec in args.holdout:
        kind, _, rest = spec.partition(":")
        if kind not in ("storm", "quiet") or not rest:
            raise SystemExit(f"Bad --holdout {spec!r}; expected storm|quiet:DATE:DAYS")
        holdout.append(parse_period(rest, kind))

    # Calibrate under exactly the amplitude settings the station will run with.
    md.FLAG_AMPLITUDE_WINDOW_MIN = args.window_min
    md.FLAG_AMPLITUDE_MODE = args.mode
    md.FLAG_AMPLITUDE_CENTERED = args.centered

    for period in train + holdout:
        load_period(
            period, args.observatory, args.warmup_days, args.column, args.cadence_s
        )

    unsettled, active = quiet_tiers(
        train, args.cadence_s, args.active_ratio, args.quiet_percentile
    )

    logger.info("Sweeping storm tiers...")
    df = sweep(
        train,
        args.cadence_s,
        unsettled,
        active,
        args.minor_grid,
        args.major_grid,
        args.severe_grid,
        args.target_recall,
    )
    if args.sweep_csv:
        df.to_csv(args.sweep_csv, index=False)
        logger.info(f"Wrote {len(df)} scored candidates to {args.sweep_csv}")

    cols = [
        "minor",
        "major",
        "severe",
        "sdr_mean",
        "sdr_worst",
        "far_mean",
        "missed_mean",
        "under_mean",
        "mae_mean",
        "objective",
    ]
    print("\nTOP CANDIDATES (training periods)")
    print(
        df[cols]
        .head(args.top_n)
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    best = df.iloc[0]
    thresholds = {
        "unsettled": unsettled,
        "active": active,
        "minor": float(best["minor"]),
        "major": float(best["major"]),
        "severe": float(best["severe"]),
    }
    train_agg = {
        k: float(best[k])
        for k in (
            "sdr_mean",
            "sdr_worst",
            "far_mean",
            "far_worst",
            "missed_mean",
            "missed_worst",
            "under_mean",
            "under_worst",
            "mae_mean",
            "mae_worst",
        )
    }

    holdout_agg = None
    if holdout:
        holdout_scores = [
            (p.kind, score(local_levels(amplitude(p, args.cadence_s), thresholds), p))
            for p in holdout
        ]
        holdout_agg = aggregate(holdout_scores)
        print("\nHELD-OUT PERFORMANCE OF THE CHOSEN THRESHOLDS")
        print(
            pd.DataFrame(
                [{"period": str(p), **s} for p, (_, s) in zip(holdout, holdout_scores)]
            ).to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
        if (
            np.isfinite(holdout_agg["sdr_mean"])
            and holdout_agg["sdr_mean"] < args.target_recall
        ):
            logger.warning(
                f"Held-out recall {holdout_agg['sdr_mean']:.3f} is below the "
                f"{args.target_recall:.2f} target: this station needs more "
                f"calibration periods, or the target is not achievable here."
            )

    # Reject a file the pipeline itself would refuse to load.
    md.validate_settings(
        {
            "FLAG_THRESHOLD_UNSETTLED_NT": thresholds["unsettled"],
            "FLAG_THRESHOLD_ACTIVE_NT": thresholds["active"],
            "FLAG_THRESHOLD_MINOR_STORM_NT": thresholds["minor"],
            "FLAG_THRESHOLD_MAJOR_STORM_NT": thresholds["major"],
            "FLAG_THRESHOLD_SEVERE_STORM_NT": thresholds["severe"],
            "FLAG_AMPLITUDE_MODE": args.mode,
            "FLAG_AMPLITUDE_WINDOW_MIN": args.window_min,
            "FLAG_AMPLITUDE_CENTERED": args.centered,
        }
    )

    text = render_config(
        args.observatory,
        thresholds,
        args.window_min,
        args.mode,
        args.centered,
        train,
        holdout,
        train_agg,
        holdout_agg,
    )
    Path(args.out).write_text(text)
    logger.info(f"Wrote {args.observatory} thresholds to {args.out}")
    print(
        f"\nUse it with:  python -m magnetometer.demos.magnetometer_demo --config {args.out} "
        f"--fetch-real-data --observatory {args.observatory} --cross-check-indices"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
