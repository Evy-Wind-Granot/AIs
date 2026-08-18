"""Command-line interface for the modular magnetometer pipeline."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import acquisition
from ..baseline import handle_gaps
from ..parsing import parse_iaga2002_to_dataframe
from ..state import PipelineState
from . import config, settings
from .analysis import run_analysis, write_json_output

LOGGER = logging.getLogger("magnetometer_pipeline")


def setup_logging(level: int = logging.INFO, fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(logging.Formatter('{"level":"%(levelname)s","message":"%(message)s"}'))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.handlers = [handler]
    LOGGER.setLevel(level)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Magnetometer Pipeline v2")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic offline smoke test")
    parser.add_argument("--config")
    parser.add_argument("--fetch-real-data", action="store_true")
    parser.add_argument("--observatory")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start-date")
    parser.add_argument("--warmup-days", type=float, default=3)
    parser.add_argument("--cadence-s", type=int)
    parser.add_argument("--column")
    parser.add_argument("--cross-check-indices", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--log-format", choices=("text", "json"), default="text")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-file", default=settings.STATE_FILE)
    parser.add_argument("--no-state-save", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-ttl-hours", type=float)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--flag-window-min", type=float)
    parser.add_argument("--flag-mode", choices=settings.VALID_AMPLITUDE_MODES)
    parser.add_argument("--flag-centered", action="store_true")
    parser.add_argument("--legacy-flags", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--loop-interval-min", type=float)
    parser.add_argument("--max-latency-min", type=float)
    parser.add_argument("--fail-on-unhealthy", action="store_true")
    parser.add_argument("--alert-webhook")
    parser.add_argument("--version", action="version", version=f"%(prog)s {settings.__version__}")
    return parser


def _synthetic_run(args) -> dict:
    rng = np.random.default_rng(42)
    n = 24 * 60
    t = np.arange(n) / 60.0
    x = 100.0 + 8.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 2.0, n)
    x[700:760] += np.linspace(0, 180, 60)
    result = run_analysis(x, 60.0, label="synthetic", start_time=datetime.now(timezone.utc), observatory="SYNTHETIC")
    print(f"self-test status={result['status']} max_level={result.get('max_local_level')}")
    return result


def _load_cli_overrides(args) -> None:
    if args.legacy_flags:
        for name, value in settings.LEGACY_FLAG_SETTINGS.items():
            setattr(settings, name, value)
    if args.flag_window_min is not None:
        settings.FLAG_AMPLITUDE_WINDOW_MIN = args.flag_window_min
    if args.flag_mode is not None:
        settings.FLAG_AMPLITUDE_MODE = args.flag_mode
    if args.flag_centered:
        settings.FLAG_AMPLITUDE_CENTERED = True
    if args.state_file:
        settings.STATE_FILE = args.state_file
    if args.no_state_save:
        settings.STATE_AUTO_SAVE = False
    if args.cache_dir is not None:
        settings.HTTP_CACHE_DIR = args.cache_dir
    if args.cache_ttl_hours is not None:
        settings.HTTP_CACHE_TTL_HOURS = args.cache_ttl_hours
    if args.no_cache:
        settings.HTTP_CACHE_ENABLED = False
    if args.max_latency_min is not None:
        settings.MAX_DATA_LATENCY_MIN = args.max_latency_min
    if args.alert_webhook is not None:
        settings.ALERT_WEBHOOK_URL = args.alert_webhook
    config.validate_settings({name: getattr(settings, name) for name in settings.SETTING_TYPES})


def main(argv: list[str] | None = None) -> dict:
    args = _parser().parse_args(argv)
    setup_logging(logging.INFO, args.log_format)
    if args.config:
        try:
            config.load_config(args.config)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    _load_cli_overrides(args)

    if args.self_test:
        return _synthetic_run(args)
    if not args.fetch_real_data:
        raise RuntimeError("must supply --fetch-real-data or --self-test")

    observatory = args.observatory or settings.DEFAULT_OBSERVATORY
    cadence = args.cadence_s or settings.DEFAULT_CADENCE_S
    column = args.column or settings.DEFAULT_COLUMN

    if args.live:
        start_dt = (pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=args.days - 1))
    elif args.start_date:
        start_dt = pd.to_datetime(args.start_date, utc=True)
    else:
        start_dt = pd.Timestamp("2024-01-01", tz="UTC")
    fetch_start = start_dt - pd.Timedelta(days=args.warmup_days)
    total_days = int(args.days + args.warmup_days)

    text = acquisition.fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=fetch_start.strftime("%Y-%m-%d"),
        duration_days=total_days,
    )
    frame = parse_iaga2002_to_dataframe(text)
    if frame.empty or column not in frame:
        raise RuntimeError(f"No usable {column} samples returned for {observatory}")

    if args.live:
        now = pd.Timestamp.now(tz="UTC")
        frame = frame[frame.index <= now]
    if frame.empty:
        raise RuntimeError("No elapsed live samples available")

    kp = dst = None
    if args.cross_check_indices:
        try:
            kp = acquisition.fetch_kp_gfz(frame.index.min().strftime("%Y-%m-%d"), frame.index.max().strftime("%Y-%m-%d"))
        except Exception as exc:
            LOGGER.warning("Kp cross-validation unavailable: %s", exc)
        parts = []
        for year, month in sorted({(ts.year, ts.month) for ts in frame.index}):
            try:
                value = acquisition.fetch_dst_kyoto(year, month)
                if value is not None:
                    parts.append(value)
            except Exception as exc:
                LOGGER.warning("Dst cross-validation unavailable for %04d-%02d: %s", year, month, exc)
        if parts:
            dst = pd.concat(parts).sort_index()

    clean = handle_gaps(frame[column].astype(float), max_gap_samples=settings.MAX_GAP_SAMPLES)
    newest = pd.to_datetime(frame.index.max(), utc=True)
    latency = max(0.0, (pd.Timestamp.now(tz="UTC") - newest).total_seconds() / 60.0)
    expected = max(1.0, (total_days * 86400.0) / cadence)
    requested_coverage = min(1.0, float(np.isfinite(frame[column]).sum()) / expected)

    state = PipelineState(args.state_file, load=True)
    if state.observatory and state.observatory != observatory:
        state = PipelineState(args.state_file, load=False)

    result = run_analysis(
        clean.to_numpy(), cadence,
        label=f"INTERMAGNET {observatory}",
        start_time=pd.to_datetime(clean.index.min(), utc=True).to_pydatetime(),
        analysis_start_time=start_dt.to_pydatetime(),
        dst_series=dst,
        kp_series=kp,
        state=state,
        dry_run=args.dry_run,
        live=args.live,
        data_latency_min=latency,
        requested_coverage=requested_coverage,
        observatory=observatory,
    )

    if args.output_json:
        write_json_output(result, args.output_json)
    if settings.STATE_AUTO_SAVE:
        state.save(observatory)

    print(f"{observatory}: status={result['status']} max_level={result.get('max_local_level')} coverage={result.get('coverage', 0):.1%}")
    return result


def run_cli(argv: list[str] | None = None) -> int:
    try:
        main(argv)
        return settings.EXIT_OK
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        message = str(exc)
        if "config" in message.lower():
            return settings.EXIT_CONFIG_INVALID
        if "fetch-real-data" in message:
            return settings.EXIT_USAGE
        return settings.EXIT_UPSTREAM_UNAVAILABLE
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        return settings.EXIT_USAGE if int(exc.code or 0) == 2 else int(exc.code or 0)
    except Exception:
        LOGGER.exception("Unhandled magnetometer error")
        return settings.EXIT_INTERNAL


def run_loop(interval_min: float, argv: list[str] | None = None) -> int:
    while True:
        started = time.monotonic()
        code = run_cli(argv)
        if code not in (settings.EXIT_OK,) + settings.RETRYABLE_EXIT_CODES:
            return code
        if code != settings.EXIT_OK:
            LOGGER.warning("Pipeline cycle failed with exit code %d; retrying", code)
        try:
            time.sleep(max(0.0, interval_min * 60 - (time.monotonic() - started)))
        except KeyboardInterrupt:
            return settings.EXIT_OK


def main_entry(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--loop-interval-min" in args:
        idx = args.index("--loop-interval-min")
        if idx + 1 < len(args):
            try:
                interval = float(args[idx + 1])
                if interval > 0:
                    return run_loop(interval, args)
            except ValueError:
                pass
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main_entry())
