#!/usr/bin/env python3
"""
Magnetometer Pipeline — Production-Ready Edition v2.

Processes magnetometer time-series using robust harmonic quiet-day curves (QDC),
residual analysis, 5-tier geomagnetic activity classification, and global index
cross-validation (Kp and Dst).

Production features:
  - YAML configuration (reload without code changes)
  - State persistence (warm restarts, crash recovery)
  - JSON output mode (machine-readable downstream consumption)
  - Structured logging (JSON Lines for log aggregation)
  - Dry-run mode (validate data availability without full compute)
  - Provenance metadata (config version, thresholds, git hash)
  - Symmetric implausibility guards (±3000 nT)
  - Data-quality gates (coverage + median-fill checks)
  - Optional webhook alerting on storm detection / gate failure

Usage:
  # Standard run with defaults
  python magnetometer_demo.py --fetch-real-data --observatory VIC --days 5 --start-date 2024-05-08

  # With config file
  python magnetometer_demo.py --config config.yaml --fetch-real-data

  # JSON output for downstream automation
  python magnetometer_demo.py --fetch-real-data --output-json results.json --log-format json

  # Dry run — validate data availability only
  python magnetometer_demo.py --fetch-real-data --dry-run

  # Resume from saved state (skip warmup re-fetch)
  python magnetometer_demo.py --fetch-real-data --state-file .magnetometer_state.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy import linalg
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Optional: YAML config support (PyYAML is common; fallback to JSON if missing)
# ---------------------------------------------------------------------------
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
    warnings.warn(
        "PyYAML not installed; --config requires YAML. Install with: pip install pyyaml"
    )

# ---------------------------------------------------------------------------
# Optional: JSON Lines logging
# ---------------------------------------------------------------------------
try:
    # python-json-logger >= 3.1 moved the formatter; the old path still works
    # but emits a DeprecationWarning, which is noise in an operational log.
    from pythonjsonlogger.json import JsonFormatter

    _HAS_JSON_LOGGER = True
except ImportError:
    try:
        from pythonjsonlogger.jsonlogger import JsonFormatter

        _HAS_JSON_LOGGER = True
    except ImportError:
        _HAS_JSON_LOGGER = False

# ---------------------------------------------------------------------------
# Version / Provenance
# ---------------------------------------------------------------------------
__version__ = "2.0.0"


# Exit-code contract (documented in RUNBOOK.md; scripts and schedulers rely on
# these, so they must not be renumbered).
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_QUALITY_GATE = 2
EXIT_UPSTREAM_UNAVAILABLE = 3
EXIT_CONFIG_INVALID = 4
EXIT_STALE_DATA = 5
EXIT_INTERNAL = 70


class PipelineError(RuntimeError):
    """Fatal, *expected* pipeline failure carrying the process exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_INTERNAL):
        super().__init__(message)
        self.exit_code = exit_code


# Correlates every log record, alert and JSON result from one invocation.
RUN_ID = uuid.uuid4().hex[:12]


def _git_hash() -> str:
    """Best-effort git commit hash for provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Default Constants (overridden by config file / CLI)
# ---------------------------------------------------------------------------
INTERMAGNET_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
KP_GFZ_URL = "https://kp.gfz-potsdam.de/app/json/"
DEFAULT_OBSERVATORY = "VIC"
DEFAULT_CADENCE_S = 60
DEFAULT_COLUMN = "x_nt"
DEFAULT_SAMPLES_PER_DAY = "Minute"
USER_AGENT = f"MagnetometerProductionPipeline/{__version__}"

# Quality gates
MIN_ANALYSIS_COVERAGE = 0.90
MAX_MEDIAN_FILL_FRACTION = 0.20
INPUT_NAN_WARNING_THRESHOLD = 0.01

# Activity classification.
#
# Tiers are assigned from the disturbance *amplitude* — the peak-to-peak range
# of the residual over a rolling window — rather than a single sample's
# residual. Kp is a 3-hourly range index and Dst an hourly average, so scoring
# one minute's deviation against them is what produced most of the misses (a
# storm's residual passes through zero many times per hour) and most of the
# false alarms (an isolated excursion during a quiet day). Set the window to 0
# to recover the legacy instantaneous behaviour.
#
# The window is trailing (causal) by default: a live monitor cannot see future
# samples, and a centered window would leak them into the label. Retrospective
# re-analysis can set FLAG_AMPLITUDE_CENTERED = True, which scores slightly
# better but is only valid offline.
#
# Thresholds below are VIC-calibrated for this window (see OPTIMIZATIONS.md).
FLAG_AMPLITUDE_WINDOW_MIN = 180.0
FLAG_AMPLITUDE_MODE = "range"  # "range" | "hybrid" | "max" | "instant"
FLAG_AMPLITUDE_CENTERED = False
FLAG_THRESHOLD_UNSETTLED_NT = 20.0
FLAG_THRESHOLD_ACTIVE_NT = 30.0
FLAG_THRESHOLD_MINOR_STORM_NT = 100.0
FLAG_THRESHOLD_MAJOR_STORM_NT = 400.0
FLAG_THRESHOLD_SEVERE_STORM_NT = 800.0
FLAG_THRESHOLD_ANOMALY_JUMP_NT = 100.0

# Legacy (pre-calibration) instantaneous thresholds, kept for --legacy-flags.
LEGACY_FLAG_SETTINGS = {
    "FLAG_AMPLITUDE_WINDOW_MIN": 0.0,
    "FLAG_AMPLITUDE_MODE": "instant",
    "FLAG_AMPLITUDE_CENTERED": False,
    "FLAG_THRESHOLD_UNSETTLED_NT": 15.0,
    "FLAG_THRESHOLD_ACTIVE_NT": 35.0,
    "FLAG_THRESHOLD_MINOR_STORM_NT": 40.0,
    "FLAG_THRESHOLD_MAJOR_STORM_NT": 140.0,
    "FLAG_THRESHOLD_SEVERE_STORM_NT": 300.0,
}
MAX_PLAUSIBLE_RESIDUAL_NT = 3000.0
MIN_PLAUSIBLE_RESIDUAL_NT = -3000.0

# Baseline fitting
BASELINE_N_ITER = 4
BASELINE_OUTLIER_THRESHOLD_NT = 30.0
BASELINE_WINDOW_HOURS = 24
BASELINE_STEP_HOURS = 12
MAX_GAP_SAMPLES = 3
STORM_FRACTION_THRESHOLD = 0.05

# State persistence
STATE_FILE = ".magnetometer_state.json"
STATE_AUTO_SAVE = True
STATE_MAX_AGE_HOURS = 168

# Output
OUTPUT_INCLUDE_ARRAYS = False
OUTPUT_INCLUDE_PROVENANCE = True

# HTTP response cache (repeat runs / parameter sweeps hit disk instead of network)
HTTP_CACHE_ENABLED = True
HTTP_CACHE_DIR = ".magnetometer_cache"
HTTP_CACHE_TTL_HOURS = 24.0

# Reuse the fitted quiet-day baseline when only classification thresholds change
BASELINE_CACHE_ENABLED = True
BASELINE_CACHE_SIZE = 4

# Alerting
ALERT_WEBHOOK_URL: Optional[str] = None
ALERT_WEBHOOK_MIN_LEVEL = "minor_storm"
ALERT_WEBHOOK_TIMEOUT_S = 10.0
# Bearer token for the webhook, read from the environment so it never lives in
# the config file or the process arguments.
ALERT_TOKEN_ENV = "MAG_ALERT_TOKEN"

# Health / freshness (matters for live monitoring, not for historical re-runs)
MAX_DATA_LATENCY_MIN = 90.0  # age of the newest finite sample
# Timestamps this far ahead of the local clock are treated as clock skew, not
# freshness, because a station cannot legitimately report the future.
CLOCK_SKEW_TOLERANCE_MIN = 5.0
MIN_REQUESTED_COVERAGE = 0.80  # fraction of the requested window actually returned
MAX_BASELINE_DRIFT_NT = 50.0  # |mean baseline shift| vs the last good fit
EXPECTED_QUIET_RMS_MAX_NT = 30.0  # station noise ceiling; above this, suspect sensor

STORM_LEVEL_ORDER = ("minor_storm", "major_storm", "severe_storm")

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("magnetometer_pipeline")


class _ContextFilter(logging.Filter):
    """Stamps run_id/observatory on every record so logs from concurrent
    observatory runs can be separated downstream."""

    def __init__(self) -> None:
        super().__init__()
        self.observatory = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = RUN_ID
        record.observatory = self.observatory
        return True


LOG_CONTEXT = _ContextFilter()


def setup_logging(level: int = logging.INFO, fmt: str = "text") -> None:
    """Configure root logger. fmt='text' | 'json'"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter: logging.Formatter
    if fmt == "json" and _HAS_JSON_LOGGER:
        formatter = JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s "
            "%(run_id)s %(observatory)s",
            rename_fields={"levelname": "level", "asctime": "ts"},
        )
    else:
        if fmt == "json":
            warnings.warn(
                "python-json-logger not installed; falling back to text format"
            )
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.filters = [LOG_CONTEXT]
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML/JSON configuration, validate it, and apply to module constants.

    A silently-ignored typo in an operational config is how a monitoring system
    ends up running with different thresholds than its operators believe, so
    unknown keys are logged and an inconsistent config is fatal
    (`EXIT_CONFIG_INVALID`) rather than merely warned about.
    """
    p = Path(path)
    if not p.exists():
        raise PipelineError(f"Config file not found: {path}", EXIT_CONFIG_INVALID)

    text = p.read_text()
    try:
        if _HAS_YAML:
            cfg = yaml.safe_load(text)
        else:
            cfg = json.loads(text)
    except Exception as e:
        raise PipelineError(
            f"Config file {path} is not valid: {e}", EXIT_CONFIG_INVALID
        )

    if not isinstance(cfg, dict):
        raise PipelineError(
            f"Config file {path} must contain a mapping at the top level",
            EXIT_CONFIG_INVALID,
        )

    overrides = _flatten_config(cfg)
    _warn_unknown_config_keys(cfg)
    validate_settings({**{k: globals()[k] for k in _SETTING_TYPES}, **overrides})

    # Apply to module globals so existing code paths keep working
    globals().update(overrides)
    logger.debug(f"Applied {len(overrides)} setting(s) from {path}")
    return cfg


# Every setting a config file may override, with the type it must have.
_SETTING_TYPES: Dict[str, Tuple[type, ...]] = {
    "MIN_ANALYSIS_COVERAGE": (int, float),
    "MAX_MEDIAN_FILL_FRACTION": (int, float),
    "INPUT_NAN_WARNING_THRESHOLD": (int, float),
    "FLAG_THRESHOLD_UNSETTLED_NT": (int, float),
    "FLAG_THRESHOLD_ACTIVE_NT": (int, float),
    "FLAG_THRESHOLD_MINOR_STORM_NT": (int, float),
    "FLAG_THRESHOLD_MAJOR_STORM_NT": (int, float),
    "FLAG_THRESHOLD_SEVERE_STORM_NT": (int, float),
    "FLAG_THRESHOLD_ANOMALY_JUMP_NT": (int, float),
    "FLAG_AMPLITUDE_WINDOW_MIN": (int, float),
    "FLAG_AMPLITUDE_MODE": (str,),
    "FLAG_AMPLITUDE_CENTERED": (bool,),
    "MAX_PLAUSIBLE_RESIDUAL_NT": (int, float),
    "MIN_PLAUSIBLE_RESIDUAL_NT": (int, float),
    "BASELINE_N_ITER": (int,),
    "BASELINE_OUTLIER_THRESHOLD_NT": (int, float),
    "BASELINE_WINDOW_HOURS": (int, float),
    "BASELINE_STEP_HOURS": (int, float),
    "MAX_GAP_SAMPLES": (int,),
    "STORM_FRACTION_THRESHOLD": (int, float),
    "STATE_FILE": (str,),
    "STATE_AUTO_SAVE": (bool,),
    "STATE_MAX_AGE_HOURS": (int, float),
    "HTTP_CACHE_ENABLED": (bool,),
    "HTTP_CACHE_DIR": (str,),
    "HTTP_CACHE_TTL_HOURS": (int, float),
    "BASELINE_CACHE_ENABLED": (bool,),
    "OUTPUT_INCLUDE_ARRAYS": (bool,),
    "OUTPUT_INCLUDE_PROVENANCE": (bool,),
    "ALERT_WEBHOOK_URL": (str, type(None)),
    "ALERT_WEBHOOK_MIN_LEVEL": (str,),
    "MAX_DATA_LATENCY_MIN": (int, float),
    "MIN_REQUESTED_COVERAGE": (int, float),
    "MAX_BASELINE_DRIFT_NT": (int, float),
    "EXPECTED_QUIET_RMS_MAX_NT": (int, float),
    "DEFAULT_OBSERVATORY": (str,),
    "DEFAULT_CADENCE_S": (int,),
    "DEFAULT_COLUMN": (str,),
}

_VALID_AMPLITUDE_MODES = ("range", "hybrid", "max", "instant")


def validate_settings(settings: Dict[str, Any]) -> None:
    """Raise PipelineError unless the resolved settings are self-consistent."""
    problems: List[str] = []

    for name, types in _SETTING_TYPES.items():
        if name not in settings:
            continue
        val = settings[name]
        # bool is an int subclass; a bool where a number belongs is a mistake.
        if bool not in types and isinstance(val, bool):
            problems.append(f"{name} must be numeric, got boolean {val!r}")
        elif not isinstance(val, types):
            names = "/".join(t.__name__ for t in types)
            problems.append(f"{name} must be {names}, got {type(val).__name__}")

    def num(name: str) -> Optional[float]:
        val = settings.get(name)
        return (
            float(val)
            if isinstance(val, (int, float)) and not isinstance(val, bool)
            else None
        )

    tiers = [
        ("FLAG_THRESHOLD_UNSETTLED_NT", num("FLAG_THRESHOLD_UNSETTLED_NT")),
        ("FLAG_THRESHOLD_ACTIVE_NT", num("FLAG_THRESHOLD_ACTIVE_NT")),
        ("FLAG_THRESHOLD_MINOR_STORM_NT", num("FLAG_THRESHOLD_MINOR_STORM_NT")),
        ("FLAG_THRESHOLD_MAJOR_STORM_NT", num("FLAG_THRESHOLD_MAJOR_STORM_NT")),
        ("FLAG_THRESHOLD_SEVERE_STORM_NT", num("FLAG_THRESHOLD_SEVERE_STORM_NT")),
    ]
    if all(v is not None for _, v in tiers):
        for (lo_name, lo), (hi_name, hi) in zip(tiers, tiers[1:]):
            assert lo is not None and hi is not None  # guarded above
            if not lo < hi:
                problems.append(
                    f"thresholds must increase: {lo_name}={lo:g} is not below "
                    f"{hi_name}={hi:g} (tiers would be unreachable)"
                )
        if tiers[0][1] is not None and tiers[0][1] <= 0:
            problems.append("FLAG_THRESHOLD_UNSETTLED_NT must be positive")

    mode = settings.get("FLAG_AMPLITUDE_MODE")
    if isinstance(mode, str) and mode not in _VALID_AMPLITUDE_MODES:
        problems.append(
            f"FLAG_AMPLITUDE_MODE must be one of {_VALID_AMPLITUDE_MODES}, got {mode!r}"
        )

    for name in (
        "MIN_ANALYSIS_COVERAGE",
        "MAX_MEDIAN_FILL_FRACTION",
        "MIN_REQUESTED_COVERAGE",
        "INPUT_NAN_WARNING_THRESHOLD",
    ):
        val = num(name)
        if val is not None and not 0.0 <= val <= 1.0:
            problems.append(f"{name} must be a fraction in [0, 1], got {val:g}")

    lo, hi = num("MIN_PLAUSIBLE_RESIDUAL_NT"), num("MAX_PLAUSIBLE_RESIDUAL_NT")
    if lo is not None and hi is not None and lo >= hi:
        problems.append(
            f"MIN_PLAUSIBLE_RESIDUAL_NT={lo:g} must be below "
            f"MAX_PLAUSIBLE_RESIDUAL_NT={hi:g}"
        )

    level = settings.get("ALERT_WEBHOOK_MIN_LEVEL")
    if isinstance(level, str) and level not in STORM_LEVEL_ORDER:
        problems.append(
            f"alerting.webhook_min_level must be one of {STORM_LEVEL_ORDER}, "
            f"got {level!r}"
        )

    url = settings.get("ALERT_WEBHOOK_URL")
    if isinstance(url, str) and url and not url.startswith(("http://", "https://")):
        problems.append("alerting.webhook_url must be an http(s) URL")

    window = num("BASELINE_WINDOW_HOURS")
    step = num("BASELINE_STEP_HOURS")
    if window is not None and step is not None and not 0 < step <= window:
        problems.append(
            f"baseline.step_hours={step:g} must be positive and no larger than "
            f"baseline.window_hours={window:g}"
        )

    if problems:
        raise PipelineError(
            "Invalid configuration:\n  - " + "\n  - ".join(problems),
            EXIT_CONFIG_INVALID,
        )


def _warn_unknown_config_keys(cfg: Dict[str, Any]) -> None:
    """Log config keys the pipeline does not understand (typo protection)."""
    known_sections: Dict[str, set] = {}
    for dotted in _CONFIG_MAPPING:
        section, _, leaf = dotted.partition(".")
        known_sections.setdefault(section, set()).add(leaf)

    for section, body in cfg.items():
        if section not in known_sections:
            logger.warning(f"Ignoring unknown config section '{section}'")
            continue
        if isinstance(body, dict):
            for leaf in body:
                if leaf not in known_sections[section]:
                    logger.warning(f"Ignoring unknown config key '{section}.{leaf}'")


# Config key -> module constant. Also drives unknown-key detection.
_CONFIG_MAPPING = {
    "observatory.default": "DEFAULT_OBSERVATORY",
    "observatory.cadence_seconds": "DEFAULT_CADENCE_S",
    "observatory.column": "DEFAULT_COLUMN",
    "quality_gates.min_analysis_coverage": "MIN_ANALYSIS_COVERAGE",
    "quality_gates.max_median_fill_fraction": "MAX_MEDIAN_FILL_FRACTION",
    "quality_gates.input_nan_warning_threshold": "INPUT_NAN_WARNING_THRESHOLD",
    "thresholds.unsettled": "FLAG_THRESHOLD_UNSETTLED_NT",
    "thresholds.active": "FLAG_THRESHOLD_ACTIVE_NT",
    "thresholds.minor_storm": "FLAG_THRESHOLD_MINOR_STORM_NT",
    "thresholds.major_storm": "FLAG_THRESHOLD_MAJOR_STORM_NT",
    "thresholds.severe_storm": "FLAG_THRESHOLD_SEVERE_STORM_NT",
    "thresholds.anomaly_jump": "FLAG_THRESHOLD_ANOMALY_JUMP_NT",
    "thresholds.amplitude_window_min": "FLAG_AMPLITUDE_WINDOW_MIN",
    "thresholds.amplitude_mode": "FLAG_AMPLITUDE_MODE",
    "thresholds.amplitude_centered": "FLAG_AMPLITUDE_CENTERED",
    "thresholds.max_plausible_residual": "MAX_PLAUSIBLE_RESIDUAL_NT",
    "thresholds.min_plausible_residual": "MIN_PLAUSIBLE_RESIDUAL_NT",
    "baseline.n_iterations": "BASELINE_N_ITER",
    "baseline.outlier_threshold_nt": "BASELINE_OUTLIER_THRESHOLD_NT",
    "baseline.window_hours": "BASELINE_WINDOW_HOURS",
    "baseline.step_hours": "BASELINE_STEP_HOURS",
    "baseline.max_gap_samples": "MAX_GAP_SAMPLES",
    "baseline.storm_fraction_threshold": "STORM_FRACTION_THRESHOLD",
    "state.file": "STATE_FILE",
    "state.auto_save": "STATE_AUTO_SAVE",
    "state.max_age_hours": "STATE_MAX_AGE_HOURS",
    "cache.enabled": "HTTP_CACHE_ENABLED",
    "cache.dir": "HTTP_CACHE_DIR",
    "cache.ttl_hours": "HTTP_CACHE_TTL_HOURS",
    "cache.reuse_baseline": "BASELINE_CACHE_ENABLED",
    "output.include_arrays": "OUTPUT_INCLUDE_ARRAYS",
    "output.include_provenance": "OUTPUT_INCLUDE_PROVENANCE",
    "alerting.webhook_url": "ALERT_WEBHOOK_URL",
    "alerting.webhook_min_level": "ALERT_WEBHOOK_MIN_LEVEL",
    "health.max_data_latency_min": "MAX_DATA_LATENCY_MIN",
    "health.min_requested_coverage": "MIN_REQUESTED_COVERAGE",
    "health.max_baseline_drift_nt": "MAX_BASELINE_DRIFT_NT",
    "health.expected_quiet_rms_max_nt": "EXPECTED_QUIET_RMS_MAX_NT",
}


def _flatten_config(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dict into module-level constant names."""
    flat: Dict[str, Any] = {}
    for key, const_name in _CONFIG_MAPPING.items():
        parts = key.split(".")
        val: Any = cfg
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                val = None
                break
        if val is not None:
            flat[const_name] = val
    return flat


# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------
class PipelineState:
    """Serializable state for warm restarts and crash recovery."""

    def __init__(self, path: str = STATE_FILE, load: bool = True):
        self.path = Path(path)
        self.last_good_coeffs: Optional[np.ndarray] = None
        self.seed_coeffs: Optional[np.ndarray] = None
        self.seed_storm_frac: float = 1.0
        self.timestamp: Optional[str] = None
        self.observatory: Optional[str] = None
        if load:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.last_good_coeffs = (
                np.array(data["last_good_coeffs"], dtype=float)
                if data.get("last_good_coeffs")
                else None
            )
            self.seed_coeffs = (
                np.array(data["seed_coeffs"], dtype=float)
                if data.get("seed_coeffs")
                else None
            )
            for name in ("last_good_coeffs", "seed_coeffs"):
                arr = getattr(self, name)
                if arr is not None and not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name} contains non-finite values")
            self.seed_storm_frac = float(data.get("seed_storm_frac", 1.0))
            self.timestamp = data.get("timestamp")
            self.observatory = data.get("observatory")
            logger.info(f"Loaded state from {self.path} (saved {self.timestamp})")
        except Exception as e:
            # A corrupt or truncated state file must degrade to a cold start,
            # never abort the run or seed the baseline with garbage.
            logger.warning(f"Discarding unusable state at {self.path}: {e}")
            self.clear()

    def clear(self) -> None:
        """Forget everything loaded from disk (cold start)."""
        self.last_good_coeffs = None
        self.seed_coeffs = None
        self.seed_storm_frac = 1.0
        self.timestamp = None
        self.observatory = None

    def save(self, observatory: str) -> None:
        data = {
            "last_good_coeffs": (
                self.last_good_coeffs.tolist()
                if self.last_good_coeffs is not None
                else None
            ),
            "seed_coeffs": (
                self.seed_coeffs.tolist() if self.seed_coeffs is not None else None
            ),
            "seed_storm_frac": float(self.seed_storm_frac),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observatory": observatory,
            "version": __version__,
        }
        try:
            # Write-then-rename: a crash mid-write leaves the previous good
            # state intact instead of a truncated file the next run must throw
            # away.
            parent = self.path.parent if str(self.path.parent) else Path(".")
            parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.path)
            logger.info(f"Saved state to {self.path}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def is_fresh(self, max_age_hours: Optional[float] = None) -> bool:
        # Resolved at call time: a config file may raise/lower the limit after
        # this class was defined.
        if max_age_hours is None:
            max_age_hours = STATE_MAX_AGE_HOURS
        if self.timestamp is None:
            return False
        try:
            saved = datetime.fromisoformat(self.timestamp)
            age = datetime.now(timezone.utc) - saved
            return age < timedelta(hours=max_age_hours)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------
def create_resilient_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        # POST is retried too: the only POST here is the alert webhook, whose
        # payload carries run_id so a receiver can deduplicate.
        allowed_methods=frozenset({"GET", "HEAD", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_CLIENT = create_resilient_session()

# In-process response cache; shared with the on-disk cache below.
_HTTP_MEMO: Dict[str, Tuple[int, str]] = {}
_HTTP_MEMO_LOCK = threading.Lock()
_DST_UNAVAILABLE: set = set()


def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    payload = json.dumps([url, sorted((params or {}).items(), key=str)], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _cache_read(key: str) -> Optional[Tuple[int, str]]:
    with _HTTP_MEMO_LOCK:
        hit = _HTTP_MEMO.get(key)
    if hit is not None:
        return hit
    if not HTTP_CACHE_ENABLED:
        return None
    path = Path(HTTP_CACHE_DIR) / f"{key}.json"
    if not path.exists():
        return None
    try:
        age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
        if age_h > HTTP_CACHE_TTL_HOURS:
            return None
        with open(path) as f:
            data = json.load(f)
        entry = (int(data["status"]), data["text"])
    except Exception as e:
        logger.debug(f"Ignoring unreadable cache entry {path}: {e}")
        return None
    with _HTTP_MEMO_LOCK:
        _HTTP_MEMO[key] = entry
    return entry


def _cache_write(key: str, status: int, text: str) -> None:
    with _HTTP_MEMO_LOCK:
        _HTTP_MEMO[key] = (status, text)
    if not HTTP_CACHE_ENABLED:
        return
    try:
        cache_dir = Path(HTTP_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f"{key}.json.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"status": status, "text": text, "url_key": key}, f)
        tmp.replace(cache_dir / f"{key}.json")
    except Exception as e:
        logger.debug(f"Could not write HTTP cache entry: {e}")


def _cache_has(url: str, params: Optional[Dict[str, Any]] = None) -> bool:
    """True when a GET for this url/params would be served from cache."""
    return _cache_read(_cache_key(url, params)) is not None


def http_get_text(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    cacheable: bool = True,
) -> Tuple[int, str]:
    """GET with a memory+disk response cache. Returns (status_code, text).

    Only successful responses are cached, and only when the caller marks the
    request as cacheable (historical windows: the upstream bytes never change).
    """
    key = _cache_key(url, params)
    if cacheable:
        hit = _cache_read(key)
        if hit is not None:
            logger.debug(f"Cache hit for {url}")
            return hit

    resp = HTTP_CLIENT.get(url, params=params, timeout=timeout)
    if cacheable and resp.status_code == 200:
        _cache_write(key, resp.status_code, resp.text)
    return resp.status_code, resp.text


def _window_is_historical(end_date: Optional[Any], min_age_days: float = 2.0) -> bool:
    """True when a requested window is old enough that upstream data is final."""
    if end_date is None:
        return False
    try:
        end = pd.to_datetime(end_date, utc=True)
    except Exception:
        return False
    return end < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=min_age_days)


# ---------------------------------------------------------------------------
# INTERMAGNET I/O
# ---------------------------------------------------------------------------
def fetch_intermagnet_iaga2002(
    observatory: str = DEFAULT_OBSERVATORY,
    start_date: Optional[str] = None,
    duration_days: int = 7,
    samples_per_day: str = DEFAULT_SAMPLES_PER_DAY,
) -> str:
    if start_date is None:
        start_date = "2024-01-01"

    params = {
        "Request": "GetData",
        "observatoryIagaCode": observatory,
        "samplesPerDay": samples_per_day,
        "dataStartDate": start_date,
        "dataDuration": duration_days,
        "format": "iaga2002",
        "orientation": "XYZF",
    }
    logger.info(
        f"Fetching INTERMAGNET data for {observatory} from {start_date} ({duration_days} days)..."
    )
    end_date = pd.to_datetime(start_date, utc=True) + pd.Timedelta(days=duration_days)
    status, text = http_get_text(
        INTERMAGNET_BASE,
        params=params,
        timeout=60,
        cacheable=_window_is_historical(end_date),
    )
    if status >= 400:
        raise requests.HTTPError(
            f"INTERMAGNET returned HTTP {status} for {observatory}"
        )
    return text


def parse_iaga2002_to_dataframe(text: str) -> pd.DataFrame:
    """Parse an IAGA-2002 payload into a UTC-indexed DataFrame.

    The data block is parsed in bulk (single whitespace-delimited read plus
    vectorized datetime/numeric conversion) rather than row by row, which is
    what dominates runtime on multi-day minute-cadence fetches.
    """
    if text.strip().startswith(("<", "<!DOCTYPE", "<html")):
        raise ValueError("INTERMAGNET returned HTML instead of IAGA-2002 data.")

    lines = text.splitlines()
    data_lines = []
    col_names = None

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        if line_s.startswith("DATE"):
            col_names = line_s.replace("|", "").split()
        elif line_s[0].isdigit():
            data_lines.append(line_s)

    if not col_names or len(col_names) < 7:
        raise ValueError("Could not parse IAGA-2002 headers.")

    def find_col(key: str) -> Optional[int]:
        for i, name in enumerate(col_names):
            if name.upper().endswith(key.upper()) and len(name) == 4:
                return i
        return None

    n_cols = len(col_names)
    # Truncated/short records are dropped, matching per-row parsing semantics.
    data_lines = [line for line in data_lines if line.count(" ") + 1 >= n_cols]
    empty = pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ("x_nt", "y_nt", "z_nt", "f_nt")},
        index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
    )
    if not data_lines:
        return empty

    raw = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        dtype=str,
        names=range(n_cols),
        index_col=False,
        engine="c",
    )
    raw = raw[raw[n_cols - 1].notna()]
    if raw.empty:
        return empty

    stamps = pd.to_datetime(raw[0] + " " + raw[1], utc=True, errors="coerce")
    if stamps.isna().any():
        n_bad = int(stamps.isna().sum())
        logger.warning(
            f"Dropped {n_bad} IAGA-2002 records with unparseable timestamps."
        )
        keep = stamps.notna().to_numpy()
        raw, stamps = raw[keep], stamps[keep]
        if raw.empty:
            return empty

    def column_values(idx: Optional[int]) -> np.ndarray:
        if idx is None or idx >= n_cols:
            return np.full(len(raw), np.nan)
        vals = pd.to_numeric(raw[idx], errors="coerce").to_numpy(dtype=float)
        # IAGA-2002 uses 99999.0 / 88888.0 sentinels for missing values.
        vals[np.abs(vals) >= 99999] = np.nan
        return vals

    df = pd.DataFrame(
        {
            "x_nt": column_values(find_col("X")),
            "y_nt": column_values(find_col("Y")),
            "z_nt": column_values(find_col("Z")),
            "f_nt": column_values(find_col("F")),
        },
        index=pd.DatetimeIndex(stamps.to_numpy(), tz="UTC", name="datetime"),
    )
    return df.sort_index()


# ---------------------------------------------------------------------------
# Global Index Fetchers
# ---------------------------------------------------------------------------
def fetch_kp_gfz(start_date: str, end_date: str) -> pd.Series:
    url = f"{KP_GFZ_URL}?start={start_date}T00:00:00Z&end={end_date}T23:59:59Z&index=Kp"
    cacheable = _window_is_historical(end_date)
    if not (cacheable and _cache_has(url)):
        logger.info("Fetching Kp index from GFZ Potsdam...")
    status, text = http_get_text(url, timeout=30, cacheable=cacheable)
    if status >= 400:
        raise requests.HTTPError(f"Kp service returned HTTP {status}")
    data = json.loads(text)

    series = pd.Series(
        np.asarray(data["Kp"], dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(data["datetime"], utc=True)),
        name="kp",
    )
    series.index.name = "datetime"
    return series.sort_index()


def fetch_dst_kyoto(year: int, month: int) -> Optional[pd.Series]:
    """Fetches Dst index from Kyoto WDC with graceful error handling."""
    yy, mm = year % 100, month
    with _HTTP_MEMO_LOCK:
        if (year, month) in _DST_UNAVAILABLE:
            return None
    urls_to_try = [
        f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for",
        f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year:04d}{mm:02d}/"
        f"dst{yy:02d}{mm:02d}.for",
        f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for",
    ]

    cacheable = _window_is_historical(
        pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.DateOffset(months=1)
    )

    def try_url(url: str) -> Optional[str]:
        try:
            status, body = http_get_text(url, timeout=15, cacheable=cacheable)
        except requests.RequestException:
            return None
        if status == 200 and "Not Found" not in body and "<html" not in body.lower():
            return body
        return None

    # The three archives (final/provisional/realtime) are probed concurrently;
    # only one exists for a given month, so probing serially pays for every miss.
    with ThreadPoolExecutor(max_workers=len(urls_to_try)) as pool:
        candidates = list(pool.map(try_url, urls_to_try))

    text = next((body for body in candidates if body is not None), None)

    if text is None:
        # Remember the miss so a second lookup for the same month in this run
        # neither re-probes the archives nor repeats the warning.
        with _HTTP_MEMO_LOCK:
            _DST_UNAVAILABLE.add((year, month))
        logger.warning(
            f"Dst index unavailable for {year:04d}-{mm:02d} from Kyoto WDC "
            f"(server down or restricted). Skipping Dst."
        )
        return None

    rows = []
    for line in text.splitlines():
        if len(line) >= 116 and (
            line[:3].strip().isdigit() or line[3:5].strip().isdigit()
        ):
            try:
                day = int(line[8:10].strip())
                hourly_part = line[20:116]
                for hour in range(24):
                    val_str = hourly_part[hour * 4 : (hour + 1) * 4].strip()
                    if val_str and val_str != "9999":
                        val = int(val_str)
                        dt = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
                        rows.append({"datetime": dt, "dst": val})
            except (ValueError, IndexError):
                continue

    if not rows:
        return None

    return pd.DataFrame(rows).set_index("datetime")["dst"].sort_index()


# ---------------------------------------------------------------------------
# Processing Engine
# ---------------------------------------------------------------------------
def handle_gaps(series: pd.Series, max_gap_samples: int = 3) -> pd.Series:
    if series.empty:
        return series
    deltas = series.index.to_series().diff().dropna()
    freq_s = max(1, int(deltas.median().total_seconds()))
    freq = pd.Timedelta(seconds=freq_s)

    regular_index = pd.date_range(
        start=series.index.min(), end=series.index.max(), freq=freq, tz="UTC"
    )
    series = series.reindex(regular_index)
    return series.interpolate(method="linear", limit=max_gap_samples)


@lru_cache(maxsize=8)
def _hanning(m: int) -> np.ndarray:
    w = np.hanning(m)
    w.flags.writeable = False
    return w


def build_design_matrix(t_hours: np.ndarray) -> np.ndarray:
    """Pure harmonic design matrix — no trend terms."""
    t = np.asarray(t_hours, dtype=float)
    cols = [
        np.ones_like(t),
        np.sin(2 * np.pi * t / 24),
        np.cos(2 * np.pi * t / 24),
        np.sin(2 * np.pi * t / 12),
        np.cos(2 * np.pi * t / 12),
        np.sin(2 * np.pi * t / 8),
        np.cos(2 * np.pi * t / 8),
        np.sin(2 * np.pi * t / 6),
        np.cos(2 * np.pi * t / 6),
    ]
    return np.column_stack(cols)


def robust_harmonic_baseline(
    x: np.ndarray,
    cadence_s: float,
    n_iter: int = 4,
    outlier_threshold_nt: float = 30.0,
    t_hours: Optional[np.ndarray] = None,
    design_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if t_hours is None and design_matrix is None:
        t_hours = np.arange(n) * cadence_s / 3600.0

    if design_matrix is not None:
        A = design_matrix
    else:
        assert t_hours is not None  # set above when no design matrix is supplied
        A = build_design_matrix(t_hours)
    valid = np.isfinite(x)
    w = np.ones(n)
    w[~valid] = 0.0

    coeffs = np.zeros(A.shape[1])
    for _ in range(n_iter):
        if valid.sum() < A.shape[1]:
            break
        Aw = A[valid] * w[valid, np.newaxis]
        xw = x[valid] * w[valid]
        coeffs, *_ = linalg.lstsq(Aw, xw, check_finite=False)[:2]
        pred = A @ coeffs
        resid = x - pred
        mad = np.median(np.abs(resid[valid] - np.median(resid[valid])))
        sigma = 1.4826 * mad + 1e-12

        w = np.ones(n)
        w[~valid] = 0.0
        w[np.abs(resid) > outlier_threshold_nt] = 0.1
        w[np.abs(resid) > 3 * sigma] = 0.01

    return A @ coeffs, coeffs


# Integer tier codes used internally by flag_activity; index into _FLAG_LABELS.
(
    _FLAG_INVALID,
    _FLAG_QUIET,
    _FLAG_UNSETTLED,
    _FLAG_ACTIVE,
    _FLAG_MINOR,
    _FLAG_MAJOR,
    _FLAG_SEVERE,
    _FLAG_ANOMALY,
) = range(8)
_FLAG_LABELS = np.array(
    [
        "invalid",
        "quiet",
        "unsettled",
        "active",
        "minor_storm",
        "major_storm",
        "severe_storm",
        "anomaly",
    ],
    dtype=object,
)


def disturbance_amplitude(residual: np.ndarray, cadence_s: float) -> np.ndarray:
    """
    Disturbance amplitude per sample (nT), used to assign activity tiers.

    "range" takes the peak-to-peak spread of the residual over the window,
    "max" the largest excursion from baseline, "instant" the sample's own
    |residual| (legacy behaviour, also used when the window is disabled).
    The window trails each sample unless FLAG_AMPLITUDE_CENTERED is set, so a
    label never depends on data that a live run would not have yet.
    NaN residuals stay NaN so the caller keeps flagging them invalid.
    """
    residual = np.asarray(residual, dtype=float)
    r = np.abs(residual)
    window = int(round(FLAG_AMPLITUDE_WINDOW_MIN * 60.0 / max(cadence_s, 1e-9)))
    if FLAG_AMPLITUDE_MODE == "instant" or window <= 1 or len(residual) == 0:
        return r

    window = min(window, len(residual))
    roll = pd.Series(residual).rolling(
        window, center=bool(FLAG_AMPLITUDE_CENTERED), min_periods=1
    )
    if FLAG_AMPLITUDE_MODE == "range":
        amp = (roll.max() - roll.min()).to_numpy()
    elif FLAG_AMPLITUDE_MODE == "hybrid":
        # Peak-to-peak misses a *sustained* depression (a Dst recovery phase
        # sits far from the quiet baseline while fluctuating very little), so
        # also score the window's mean offset from baseline.
        amp = np.maximum(
            (roll.max() - roll.min()).to_numpy(),
            2.0 * np.abs(roll.mean().to_numpy()),
        )
    elif FLAG_AMPLITUDE_MODE == "max":
        amp = np.maximum(roll.max().abs().to_numpy(), roll.min().abs().to_numpy())
    else:
        raise ValueError(f"Unknown FLAG_AMPLITUDE_MODE: {FLAG_AMPLITUDE_MODE}")

    amp[~np.isfinite(residual)] = np.nan
    return amp


def flag_activity(residual: np.ndarray, cadence_s: float = 60.0) -> np.ndarray:
    """
    Classify each residual sample into an activity tier.

    NaN/Inf residuals are explicitly flagged "invalid" rather than falling
    through to the default — comparisons like `NaN > 15` are silently False,
    so an unguarded default of "quiet" would misclassify missing/garbage
    data as a clean quiet baseline and corrupt every downstream metric.
    """
    residual = np.asarray(residual, dtype=float)
    r = np.abs(residual)
    finite = np.isfinite(residual)
    amplitude = disturbance_amplitude(residual, cadence_s)

    # Tiers are assigned as integer codes and mapped to labels once at the end;
    # writing into an object array per tier is what made this expensive.
    codes = np.full(len(residual), _FLAG_INVALID, dtype=np.int8)
    codes[finite] = _FLAG_QUIET

    for code, threshold in (
        (_FLAG_UNSETTLED, FLAG_THRESHOLD_UNSETTLED_NT),
        (_FLAG_ACTIVE, FLAG_THRESHOLD_ACTIVE_NT),
        (_FLAG_MINOR, FLAG_THRESHOLD_MINOR_STORM_NT),
        (_FLAG_MAJOR, FLAG_THRESHOLD_MAJOR_STORM_NT),
        (_FLAG_SEVERE, FLAG_THRESHOLD_SEVERE_STORM_NT),
    ):
        codes[finite & (amplitude > threshold)] = code

    # Sensor artifacts are *spikes*: the field jumps and snaps straight back
    # within one sample. Geophysical disturbance cannot do that, whereas a
    # storm onset is a sustained step, so requiring the reversal keeps real
    # onsets in their storm tier instead of relabelling them "anomaly".
    diff = np.diff(residual, prepend=residual[0])
    rebound = np.roll(diff, -1)
    rebound[-1] = 0.0
    big = np.abs(diff) > FLAG_THRESHOLD_ANOMALY_JUMP_NT
    spike = (
        finite
        & np.isfinite(diff)
        & np.isfinite(rebound)
        & big
        & (np.abs(rebound) > FLAG_THRESHOLD_ANOMALY_JUMP_NT)
        & (np.sign(rebound) != np.sign(diff))
    )
    codes[spike] = _FLAG_ANOMALY

    # Physically-implausible magnitude: don't let a sensor fault masquerade
    # as a detected severe storm. This check is LAST so it overrides any
    # storm or anomaly classification — impossible data is always invalid.
    implausible = finite & (
        (r > MAX_PLAUSIBLE_RESIDUAL_NT) | (residual < MIN_PLAUSIBLE_RESIDUAL_NT)
    )
    if np.any(implausible):
        codes[implausible] = _FLAG_INVALID

    return _FLAG_LABELS[codes]


def cross_validate_flags(
    local_flags: np.ndarray,
    dst_vals: np.ndarray,
    kp_vals: np.ndarray,
) -> np.ndarray:
    local_flags = np.asarray(local_flags, dtype=object)
    dst_vals = np.asarray(dst_vals, dtype=float)
    kp_vals = np.asarray(kp_vals, dtype=float)

    with np.errstate(invalid="ignore"):
        main_phase = (dst_vals < -50) | (kp_vals >= 6)
        active = (dst_vals < -30) | (kp_vals >= 4)
    calm = ~active & ~main_phase

    quiet = local_flags == "quiet"
    big_storm = (local_flags == "major_storm") | (local_flags == "severe_storm")

    validation = np.full(len(local_flags), "ok", dtype=object)
    validation[quiet & main_phase] = "missed_global_event"
    validation[quiet & ~main_phase & active] = "under_reacting"
    validation[big_storm & calm] = "unconfirmed_storm"

    return validation


# ---------------------------------------------------------------------------
# Metrics Engine
# ---------------------------------------------------------------------------
class MetricsEngine:
    """
    Compute key validation metrics comparing local pipeline output against
    global geomagnetic indices (Kp, Dst).
    """

    LOCAL_LEVELS = {
        "quiet": 0,
        "unsettled": 1,
        "active": 2,
        "minor_storm": 3,
        "major_storm": 4,
        "severe_storm": 4,
    }

    def __init__(
        self,
        quiet_kp: float = 2.0,
        quiet_dst: float = -10.0,
        storm_kp: float = 7.0,
        storm_dst: float = -100.0,
    ):
        self.quiet_kp = quiet_kp
        self.quiet_dst = quiet_dst
        self.storm_kp = storm_kp
        self.storm_dst = storm_dst

    @staticmethod
    def _global_level(kp: float, dst: float) -> float:
        kp_level = np.nan
        if pd.notna(kp):
            if kp <= 2:
                kp_level = 0
            elif kp <= 4:
                kp_level = 1
            elif kp < 6:
                kp_level = 2
            elif kp < 8:
                kp_level = 3
            else:
                kp_level = 4

        dst_level = np.nan
        if pd.notna(dst):
            if dst >= -10:
                dst_level = 0
            elif dst >= -30:
                dst_level = 1
            elif dst >= -50:
                dst_level = 2
            elif dst >= -100:
                dst_level = 3
            else:
                dst_level = 4

        if pd.notna(kp_level) and pd.notna(dst_level):
            return max(kp_level, dst_level)
        if pd.notna(kp_level):
            return kp_level
        return dst_level

    @staticmethod
    def _global_levels(kp_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
        """Vectorized equivalent of _global_level over whole arrays."""
        kp = np.asarray(kp_vals, dtype=float)
        dst = np.asarray(dst_vals, dtype=float)

        with np.errstate(invalid="ignore"):
            kp_level = np.select(
                [kp <= 2, kp <= 4, kp < 6, kp < 8], [0.0, 1.0, 2.0, 3.0], default=4.0
            )
            dst_level = np.select(
                [dst >= -10, dst >= -30, dst >= -50, dst >= -100],
                [0.0, 1.0, 2.0, 3.0],
                default=4.0,
            )
        kp_level[~np.isfinite(kp)] = np.nan
        dst_level[~np.isfinite(dst)] = np.nan

        # fmax propagates only when both are NaN, matching the scalar fallbacks.
        return np.fmax(kp_level, dst_level)

    def compute(
        self,
        residual: np.ndarray,
        flags: np.ndarray,
        validation: np.ndarray,
        kp_vals: np.ndarray,
        dst_vals: np.ndarray,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        n = len(flags)

        flags = np.asarray(flags, dtype=object)
        local_levels = np.full(n, np.nan)
        for label, level in self.LOCAL_LEVELS.items():
            local_levels[flags == label] = level

        global_levels = self._global_levels(kp_vals, dst_vals)

        has_global = np.isfinite(global_levels)
        has_local = np.isfinite(local_levels)
        both = has_global & has_local

        # --- 1. Quiet-period RMS (NaN-safe) ---
        quiet_mask = (flags == "quiet") & has_global & (global_levels == 0)
        finite_quiet = quiet_mask & np.isfinite(residual)
        if np.any(finite_quiet):
            metrics["quiet_rms_nt"] = float(
                np.sqrt(np.nanmean(residual[quiet_mask] ** 2))
            )
        else:
            metrics["quiet_rms_nt"] = np.nan

        # --- 2. Storm Detection Rate (Recall) ---
        truth_storm = has_global & (global_levels >= 3)
        pred_storm = has_local & (local_levels >= 3)

        tp = int(np.sum(pred_storm & truth_storm))
        fn = int(np.sum(~pred_storm & truth_storm))
        metrics["storm_detection_rate"] = (
            float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        )

        # --- 3. False Alarm Rate ---
        fp = int(np.sum(pred_storm & has_global & (global_levels < 3)))
        tn = int(np.sum(~pred_storm & has_global & (global_levels < 3)))
        metrics["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan

        # --- 4. Missed Global Event Rate ---
        missed = int(
            np.sum(has_global & (global_levels >= 3) & has_local & (local_levels < 2))
        )
        total_global_storms = int(np.sum(truth_storm))
        metrics["missed_global_event_rate"] = (
            float(missed / total_global_storms) if total_global_storms > 0 else np.nan
        )

        # --- 5. Under-reacting Rate ---
        under = int(
            np.sum(has_global & (global_levels >= 2) & has_local & (local_levels < 2))
        )
        total_global_active = int(np.sum(has_global & (global_levels >= 2)))
        metrics["under_reacting_rate"] = (
            float(under / total_global_active) if total_global_active > 0 else np.nan
        )

        # --- 6. Unconfirmed Storm Rate ---
        local_storm_count = int(np.sum(pred_storm))
        unconfirmed = int(np.sum(pred_storm & has_global & (global_levels < 3)))
        metrics["unconfirmed_storm_rate"] = (
            float(unconfirmed / local_storm_count) if local_storm_count > 0 else np.nan
        )

        # --- 7. Mean Absolute Level Error ---
        if np.any(both):
            metrics["mean_abs_level_error"] = float(
                np.mean(np.abs(local_levels[both] - global_levels[both]))
            )
        else:
            metrics["mean_abs_level_error"] = np.nan

        # --- 8. Validation Yield (% ok) ---
        if len(validation) > 0:
            metrics["validation_yield_ok"] = float(
                np.sum(validation == "ok") / len(validation)
            )
        else:
            metrics["validation_yield_ok"] = np.nan

        # --- 9. Coverage ---
        metrics["samples_with_global_data"] = int(np.sum(has_global))
        metrics["total_samples"] = n

        return metrics


# ---------------------------------------------------------------------------
# Processing Engine (continued)
# ---------------------------------------------------------------------------
_BaselineFit = Tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float
]
_BASELINE_CACHE: "OrderedDict[str, _BaselineFit]" = OrderedDict()


def _baseline_cache_key(
    x: np.ndarray,
    cadence_s: float,
    kp_aligned: np.ndarray,
    dst_aligned: np.ndarray,
    last_good_coeffs: Optional[np.ndarray],
    seed_coeffs: Optional[np.ndarray],
    seed_storm_frac: float,
) -> str:
    h = hashlib.blake2b(digest_size=16)
    for arr in (x, kp_aligned, dst_aligned, last_good_coeffs, seed_coeffs):
        h.update(
            b"|" if arr is None else np.ascontiguousarray(arr, dtype=float).tobytes()
        )
    h.update(
        repr(
            (
                float(cadence_s),
                float(seed_storm_frac),
                BASELINE_N_ITER,
                BASELINE_OUTLIER_THRESHOLD_NT,
                BASELINE_WINDOW_HOURS,
                BASELINE_STEP_HOURS,
                STORM_FRACTION_THRESHOLD,
            )
        ).encode()
    )
    return h.hexdigest()


def _fit_baseline(
    x: np.ndarray,
    cadence_s: float,
    kp_aligned: np.ndarray,
    dst_aligned: np.ndarray,
    last_good_coeffs: Optional[np.ndarray],
    seed_coeffs: Optional[np.ndarray],
    seed_storm_frac: float,
) -> _BaselineFit:
    """Fit the sliding-window harmonic QDC baseline.

    The result depends only on the input series, cadence, global indices and
    the baseline settings — not on the activity thresholds — so repeated calls
    with the same inputs (threshold sweeps) are served from a small cache.
    """
    cache_key = None
    if BASELINE_CACHE_ENABLED:
        cache_key = _baseline_cache_key(
            x,
            cadence_s,
            kp_aligned,
            dst_aligned,
            last_good_coeffs,
            seed_coeffs,
            seed_storm_frac,
        )
        cached = _BASELINE_CACHE.get(cache_key)
        if cached is not None:
            _BASELINE_CACHE.move_to_end(cache_key)
            logger.debug("Reusing cached quiet-day baseline fit.")
            baseline, uncomputed, good, seed, frac = cached
            return baseline.copy(), uncomputed, good, seed, frac

    n = len(x)
    baseline = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)

    window_samples = int(BASELINE_WINDOW_HOURS * 3600 / cadence_s)
    step_samples = int(BASELINE_STEP_HOURS * 3600 / cadence_s)
    t_global = np.arange(n) * cadence_s / 3600.0
    # One design matrix for the whole series; windows are row slices of it.
    A_global = build_design_matrix(t_global)

    for start in range(0, max(1, n - step_samples), step_samples):
        end = min(start + window_samples, n)
        if end - start < step_samples // 2:
            break

        segment = x[start:end]
        t_seg = t_global[start:end]
        A_seg = A_global[start:end]

        if np.isfinite(segment).sum() < (end - start) * 0.5:
            continue

        # Check global indices for this window
        global_storm = False
        kp_win = kp_aligned[start:end]
        dst_win = dst_aligned[start:end]
        if np.any(np.isfinite(kp_win)) or np.any(np.isfinite(dst_win)):
            kp_max = np.nanmax(kp_win) if np.any(np.isfinite(kp_win)) else np.nan
            dst_min = np.nanmin(dst_win) if np.any(np.isfinite(dst_win)) else np.nan
            if (np.isfinite(kp_max) and kp_max >= 5) or (
                np.isfinite(dst_min) and dst_min < -30
            ):
                global_storm = True

        seg_base, coeffs = robust_harmonic_baseline(
            segment,
            cadence_s,
            n_iter=BASELINE_N_ITER,
            outlier_threshold_nt=BASELINE_OUTLIER_THRESHOLD_NT,
            t_hours=t_seg,
            design_matrix=A_seg,
        )

        seg_res = segment - seg_base
        storm_frac = np.sum(np.abs(seg_res) > 50) / len(segment)

        # Track seed baseline from the calmest window seen so far
        if storm_frac < seed_storm_frac:
            seed_storm_frac = storm_frac
            seed_coeffs = coeffs

        window_is_stormy = (storm_frac > STORM_FRACTION_THRESHOLD) or global_storm
        can_extrapolate = last_good_coeffs is not None

        if window_is_stormy and can_extrapolate:
            seg_base = A_seg @ last_good_coeffs
            if global_storm:
                logger.info(
                    f"Global storm window [{start}:{end}] "
                    f"(Kp_max={kp_max:.1f}, Dst_min={dst_min:.1f}). "
                    f"Extrapolating quiet baseline."
                )
            else:
                logger.info(
                    f"Local storm window [{start}:{end}]. Extrapolating quiet baseline."
                )
        elif not window_is_stormy:
            last_good_coeffs = coeffs
        else:
            # Stormy and no prior good baseline: try seed, else provisional
            if seed_coeffs is not None and seed_coeffs is not coeffs:
                seg_base = A_seg @ seed_coeffs
                logger.warning(
                    f"Storm at [{start}:{end}] with no prior quiet baseline. "
                    f"Using seed baseline from calmest window (storm_frac={seed_storm_frac:.3f})."
                )
            else:
                logger.warning(
                    f"Storm detected at [{start}:{end}] but no prior quiet baseline. "
                    f"Using provisional fit (will not be saved)."
                )

        w_win = _hanning(end - start)
        baseline[start:end] += seg_base * w_win
        weights[start:end] += w_win

    weights_mask = weights > 0
    if np.any(weights_mask):
        baseline[weights_mask] /= weights[weights_mask]

    # Edge handling: interpolate small edge gaps instead of falling back to global median
    uncomputed = ~weights_mask
    if np.any(uncomputed):
        if np.any(np.isfinite(x)):
            computed_idx = np.where(weights_mask)[0]
            gap_idx = np.where(uncomputed)[0]
            if len(computed_idx) >= 2:
                # Linear interpolation between the nearest computed samples,
                # clamped to the end values outside the computed span.
                baseline[gap_idx] = np.interp(
                    gap_idx, computed_idx, baseline[computed_idx]
                )
            else:
                baseline[uncomputed] = np.nanmedian(x)
            n_filled = len(gap_idx)
            if n_filled > 2:
                logger.info(
                    f"{n_filled} edge samples interpolated from neighboring baselines."
                )
        else:
            logger.error(
                "No valid data anywhere in the input. Baseline cannot be computed."
            )
            baseline[uncomputed] = 0.0

    if cache_key is not None:
        _BASELINE_CACHE[cache_key] = (
            baseline.copy(),
            uncomputed,
            last_good_coeffs,
            seed_coeffs,
            seed_storm_frac,
        )
        while len(_BASELINE_CACHE) > max(1, BASELINE_CACHE_SIZE):
            _BASELINE_CACHE.popitem(last=False)

    return baseline, uncomputed, last_good_coeffs, seed_coeffs, seed_storm_frac


def _flag_counts(flags: Optional[np.ndarray]) -> Dict[str, int]:
    """Flag histogram, JSON-safe, for output and alerting."""
    if flags is None or len(flags) == 0:
        return {}
    labels, counts = np.unique(np.asarray(flags, dtype=object), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def _max_local_level(flag_counts: Dict[str, int]) -> Optional[str]:
    """Most severe activity tier present, or None when nothing was classified."""
    order = (
        "quiet",
        "unsettled",
        "active",
        "minor_storm",
        "major_storm",
        "severe_storm",
    )
    present = [level for level in order if flag_counts.get(level, 0) > 0]
    return present[-1] if present else None


def assess_health(
    coverage: float,
    median_fill_frac: float,
    quiet_rms_nt: Optional[float],
    data_latency_min: Optional[float],
    requested_coverage: Optional[float],
    baseline_drift_nt: Optional[float],
    live: bool,
) -> Dict[str, Any]:
    """Machine-readable health verdict for the run.

    Every check is `True` when it passes. A check that cannot be evaluated (no
    reference state yet, no quiet samples, a historical window with no latency
    requirement) is omitted rather than reported either way, so `healthy` never
    claims a verdict the data does not support; a monitoring rule can require
    all present checks to be true.
    """
    checks: Dict[str, bool] = {
        "coverage": bool(coverage >= MIN_ANALYSIS_COVERAGE),
        "baseline_fit": bool(median_fill_frac <= MAX_MEDIAN_FILL_FRACTION),
    }
    if live:
        # A negative latency means the newest timestamp is ahead of our clock, so
        # freshness is unverifiable rather than excellent: fail it instead.
        checks["data_freshness"] = bool(
            data_latency_min is not None
            and -CLOCK_SKEW_TOLERANCE_MIN <= data_latency_min <= MAX_DATA_LATENCY_MIN
        )
    if requested_coverage is not None:
        checks["requested_window_returned"] = bool(
            requested_coverage >= MIN_REQUESTED_COVERAGE
        )
    if baseline_drift_nt is not None and np.isfinite(baseline_drift_nt):
        checks["baseline_stability"] = bool(
            abs(baseline_drift_nt) <= MAX_BASELINE_DRIFT_NT
        )
    if quiet_rms_nt is not None and np.isfinite(quiet_rms_nt):
        checks["station_noise"] = bool(quiet_rms_nt <= EXPECTED_QUIET_RMS_MAX_NT)

    health = {
        "healthy": all(checks.values()),
        "checks": checks,
        "data_latency_min": data_latency_min,
        "requested_coverage": requested_coverage,
        "baseline_drift_nt": baseline_drift_nt,
        "quiet_rms_nt": quiet_rms_nt,
    }
    for name, ok in checks.items():
        if not ok:
            logger.warning(f"Health check failed: {name}")
    return health


def _baseline_drift_nt(
    baseline: np.ndarray,
    coeffs: Optional[np.ndarray],
    reference_coeffs: Optional[np.ndarray],
) -> Optional[float]:
    """Shift in the fitted quiet-day level vs the previous good fit, in nT.

    Element 0 of the harmonic coefficient vector is the DC offset, so the
    difference of the two offsets is the constant part of the baseline change —
    a sensor step, a pier change or a failed fit shows up here before it shows
    up in the classification metrics.
    """
    if coeffs is None or reference_coeffs is None:
        return None
    if len(coeffs) == 0 or len(reference_coeffs) == 0:
        return None
    try:
        return float(coeffs[0] - reference_coeffs[0])
    except Exception:
        return None


def run_analysis(
    x: np.ndarray,
    cadence_s: float,
    label: str = "",
    start_time: Optional[datetime] = None,
    analysis_start_time: Optional[datetime] = None,
    dst_series: Optional[pd.Series] = None,
    kp_series: Optional[pd.Series] = None,
    state: Optional[PipelineState] = None,
    dry_run: bool = False,
    live: bool = False,
    data_latency_min: Optional[float] = None,
    requested_coverage: Optional[float] = None,
    observatory: str = "-",
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    logger.info(
        f"Running Analysis: {label} ({n} samples, {n * cadence_s / 3600:.1f} hours)"
    )
    if n == 0:
        raise PipelineError(
            "No samples to analyse — upstream returned an empty series.",
            EXIT_UPSTREAM_UNAVAILABLE,
        )

    # Defensive NaN check on input
    input_nan_frac = np.sum(~np.isfinite(x)) / n
    if input_nan_frac > INPUT_NAN_WARNING_THRESHOLD:
        logger.warning(
            f"Input data has {input_nan_frac*100:.1f}% NaN/Inf values. "
            f"Gap filling may be insufficient."
        )

    # Pre-align global indices to the full time grid
    kp_aligned = np.full(n, np.nan)
    dst_aligned = np.full(n, np.nan)
    if dst_series is not None or kp_series is not None:
        index = pd.date_range(
            start=start_time, periods=n, freq=pd.Timedelta(seconds=cadence_s)
        )
        if kp_series is not None:
            kp_aligned = kp_series.reindex(
                index, method="ffill", tolerance=pd.Timedelta("3h")
            ).values
        if dst_series is not None:
            # Dst is hourly; a 3 h tolerance would carry a value two hours past
            # its validity and blur storm onsets in the cross-check.
            dst_aligned = dst_series.reindex(
                index, method="ffill", tolerance=pd.Timedelta("1h")
            ).values

    # Dry run: stop here after data validation
    if dry_run:
        logger.info("Dry run complete — data validated, skipping computation.")
        return {
            "status": "dry_run",
            "coverage": 1.0 - input_nan_frac,
            "median_fill_frac": 0.0,
            "baseline": None,
            "residual": None,
            "flags": None,
            "validation": None,
            "metrics": {},
            "flag_counts": {},
            "validation_source": "none",
            "health": assess_health(
                1.0 - input_nan_frac,
                0.0,
                None,
                data_latency_min,
                requested_coverage,
                None,
                live,
            ),
        }

    # Restore state if fresh
    last_good_coeffs = None
    seed_coeffs = None
    seed_storm_frac = 1.0
    reference_coeffs = state.last_good_coeffs if state is not None else None
    if state is not None and state.is_fresh():
        last_good_coeffs = state.last_good_coeffs
        seed_coeffs = state.seed_coeffs
        seed_storm_frac = state.seed_storm_frac
        logger.info(
            f"Restored state from {state.path} (seed_storm_frac={seed_storm_frac:.3f})"
        )

    baseline, uncomputed, last_good_coeffs, seed_coeffs, seed_storm_frac = (
        _fit_baseline(
            x,
            cadence_s,
            kp_aligned,
            dst_aligned,
            last_good_coeffs,
            seed_coeffs,
            seed_storm_frac,
        )
    )

    residual = x - baseline
    flags = flag_activity(residual, cadence_s)

    # Robust analysis period slicing using the actual data index
    if analysis_start_time is not None and start_time is not None:
        full_index = pd.date_range(
            start=start_time, periods=n, freq=pd.Timedelta(seconds=cadence_s)
        )
        valid_positions = np.where(full_index >= analysis_start_time)[0]
        analysis_start_idx = int(valid_positions[0]) if len(valid_positions) > 0 else n
        analysis_start_idx = max(0, min(analysis_start_idx, n))
    else:
        analysis_start_idx = 0

    # Slice to analysis period for reporting
    baseline_a = baseline[analysis_start_idx:]
    residual_a = residual[analysis_start_idx:]
    flags_a = flags[analysis_start_idx:]
    validation_a = None
    kp_a = kp_aligned[analysis_start_idx:]
    dst_a = dst_aligned[analysis_start_idx:]
    median_filled_a = uncomputed[analysis_start_idx:]

    n_a = max(1, len(residual_a))

    # Defensive NaN check on analysis residual
    analysis_nan_frac = np.sum(~np.isfinite(residual_a)) / n_a
    if analysis_nan_frac > 0:
        logger.warning(
            f"Analysis period has {analysis_nan_frac*100:.1f}% non-finite residuals. "
            f"Check for data gaps or baseline coverage holes."
        )

    median_fill_frac = float(np.sum(median_filled_a) / n_a)
    if median_fill_frac > 0:
        logger.info(
            f"Analysis period baseline was interpolated for {median_fill_frac*100:.1f}% of samples."
        )

    # --- Data-quality gate --------------------------------------------
    coverage = 1.0 - analysis_nan_frac
    if coverage < MIN_ANALYSIS_COVERAGE or median_fill_frac > MAX_MEDIAN_FILL_FRACTION:
        logger.error(
            f"Data quality gate failed: analysis-period finite coverage={coverage:.1%} "
            f"(min {MIN_ANALYSIS_COVERAGE:.0%}), interpolated fraction={median_fill_frac:.1%} "
            f"(max {MAX_MEDIAN_FILL_FRACTION:.0%}). Refusing to report metrics for this run."
        )
        result = {
            "status": "insufficient_data",
            "coverage": coverage,
            "median_fill_frac": median_fill_frac,
            "baseline": baseline_a,
            "residual": residual_a,
            "flags": flags_a,
            "validation": None,
            "metrics": {},
            "flag_counts": _flag_counts(flags_a),
            "validation_source": "none",
            "health": assess_health(
                coverage,
                median_fill_frac,
                None,
                data_latency_min,
                requested_coverage,
                _baseline_drift_nt(baseline, last_good_coeffs, reference_coeffs),
                live,
            ),
        }
        _maybe_alert(result, observatory)
        _maybe_save_state(state, last_good_coeffs, seed_coeffs, seed_storm_frac)
        return result

    metrics: Dict[str, Any] = {}
    validation_source = "none"
    if dst_series is not None or kp_series is not None:
        # kp_aligned / dst_aligned already hold the ffill-aligned indices.
        validation_a = cross_validate_flags(flags_a, dst_a, kp_a)

        metrics_engine = MetricsEngine()
        metrics = metrics_engine.compute(
            residual_a,
            flags_a,
            validation_a,
            kp_a,
            dst_a,
        )

        has_dst = dst_series is not None and not dst_series.empty
        has_kp = kp_series is not None and not kp_series.empty
        if has_dst and has_kp:
            validation_source = "Kp+Dst"
        elif has_kp:
            validation_source = "Kp-only"
        elif has_dst:
            validation_source = "Dst-only"

        logger.info("--- Validation Metrics (Analysis Period) ---")
        for key, val in metrics.items():
            if isinstance(val, float):
                if np.isfinite(val):
                    logger.info(f"  {key:30s}: {val:.4f}")
                else:
                    logger.info(f"  {key:30s}: N/A")
            else:
                logger.info(f"  {key:30s}: {val}")

    else:
        validation_a = np.full(len(flags_a), "no_index_data", dtype=object)

    logger.info("Activity Flag Breakdown (Analysis Period):")
    for u, c in zip(*np.unique(flags_a, return_counts=True)):
        logger.info(f"  {u:12s}: {c}")

    quiet_mask = flags_a == "quiet"
    if np.any(quiet_mask):
        quiet_resid = residual_a[quiet_mask]
        if np.any(np.isfinite(quiet_resid)):
            logger.info(f"Quiet Period Residual RMS: {np.nanstd(quiet_resid):.2f} nT")
        else:
            logger.info("Quiet Period Residual RMS: N/A (no finite quiet samples)")

    if np.any(np.isfinite(residual_a)):
        logger.info(f"Overall Residual RMS: {np.nanstd(residual_a):.2f} nT")
        logger.info(
            f"Residual Range: {np.nanmin(residual_a):.2f} nT to {np.nanmax(residual_a):.2f} nT"
        )
    else:
        logger.error(
            "Analysis period residual is entirely non-finite. Baseline computation failed."
        )

    quiet_rms = float(np.nanstd(residual_a[quiet_mask])) if np.any(quiet_mask) else None
    drift = _baseline_drift_nt(baseline, last_good_coeffs, reference_coeffs)
    if drift is not None:
        logger.info(f"Baseline offset drift vs last good fit: {drift:+.2f} nT")

    flag_counts = _flag_counts(flags_a)
    result = {
        "status": "ok",
        "coverage": coverage,
        "median_fill_frac": median_fill_frac,
        "baseline": baseline_a,
        "residual": residual_a,
        "flags": flags_a,
        "validation": validation_a,
        "metrics": metrics,
        "flag_counts": flag_counts,
        "max_local_level": _max_local_level(flag_counts),
        "validation_source": validation_source,
        "health": assess_health(
            coverage,
            median_fill_frac,
            quiet_rms,
            data_latency_min,
            requested_coverage,
            drift,
            live,
        ),
    }

    _maybe_alert(result, observatory)
    _maybe_save_state(state, last_good_coeffs, seed_coeffs, seed_storm_frac)
    return result


def _maybe_save_state(
    state: Optional[PipelineState],
    last_good_coeffs: Optional[np.ndarray],
    seed_coeffs: Optional[np.ndarray],
    seed_storm_frac: float,
) -> None:
    if state is None or not STATE_AUTO_SAVE:
        return
    state.last_good_coeffs = last_good_coeffs
    state.seed_coeffs = seed_coeffs
    state.seed_storm_frac = seed_storm_frac
    # Observatory is set by caller in main()


def _maybe_alert(result: Dict[str, Any], observatory: str = "-") -> None:
    """POST an alert when a storm is detected or the run is unhealthy.

    Fires on: a local flag at or above `ALERT_WEBHOOK_MIN_LEVEL`, a failed
    quality gate, or any health check that is not `ok` (stale data, baseline
    drift, excessive station noise) — an operator needs the last group most,
    because a silent pipeline and a quiet magnetosphere look identical.
    """
    if not ALERT_WEBHOOK_URL:
        return

    status = result.get("status", "")
    flags = result.get("flags")
    health = result.get("health", {})

    # STORM_LEVEL_ORDER is an ordered tuple: indexing a set here (as this
    # previously did) picks a random severity floor per process.
    try:
        min_idx = STORM_LEVEL_ORDER.index(ALERT_WEBHOOK_MIN_LEVEL)
    except ValueError:
        min_idx = 0
    at_or_above = set(STORM_LEVEL_ORDER[min_idx:])

    counts = result.get("flag_counts") or {}
    if counts:
        has_storm = any(counts.get(level, 0) > 0 for level in at_or_above)
    elif flags is not None:
        has_storm = bool(
            np.isin(np.asarray(flags, dtype=object), list(at_or_above)).any()
        )
    else:
        has_storm = False

    gate_failed = status == "insufficient_data"
    unhealthy = [name for name, ok in health.get("checks", {}).items() if not ok]

    if not has_storm and not gate_failed and not unhealthy:
        return

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "observatory": observatory,
        "version": __version__,
        "status": status,
        "storm_detected": has_storm,
        "gate_failed": gate_failed,
        "failed_health_checks": unhealthy,
        "max_local_level": result.get("max_local_level"),
        "coverage": result.get("coverage"),
        "median_fill_frac": result.get("median_fill_frac"),
        "data_latency_min": health.get("data_latency_min"),
        "baseline_drift_nt": health.get("baseline_drift_nt"),
        "metrics": {
            k: v
            for k, v in (result.get("metrics") or {}).items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        },
    }
    headers = {}
    token = os.environ.get(ALERT_TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        # HTTP_CLIENT carries the retry/backoff policy; a bare requests.post
        # would drop the alert on a single 503 from the receiver.
        resp = HTTP_CLIENT.post(
            ALERT_WEBHOOK_URL,
            json=payload,
            timeout=ALERT_WEBHOOK_TIMEOUT_S,
            headers=headers,
        )
        if resp.status_code >= 400:
            logger.error(f"Alert webhook rejected the alert: HTTP {resp.status_code}")
        else:
            logger.info(f"Alert webhook accepted the alert (HTTP {resp.status_code})")
    except Exception as e:
        # Alerting is best-effort: never fail a good run because the receiver
        # is down, but make the miss loud in the logs.
        logger.error(f"Alert webhook unreachable, alert dropped: {e}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def _prefetch_global_indices(start_date: str, duration_days: int) -> None:
    """Warm the response cache for Kp/Dst over the requested window."""
    try:
        start = pd.to_datetime(start_date, utc=True)
        end = start + pd.Timedelta(days=duration_days) - pd.Timedelta(minutes=1)
        months = sorted(
            {(dt.year, dt.month) for dt in pd.date_range(start, end, freq="D")}
        )
        jobs: List[Callable[[], Any]] = [
            lambda: fetch_kp_gfz(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        ]
        jobs += [partial(fetch_dst_kyoto, y, m) for y, m in months]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for future in [pool.submit(job) for job in jobs]:
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"Index prefetch failed (will retry inline): {e}")
    except Exception as e:
        logger.debug(f"Index prefetch skipped: {e}")


def _positive_int(val: str) -> int:
    iv = int(val)
    if iv <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {val}")
    return iv


def _non_negative_float(val: str) -> float:
    fv = float(val)
    if fv < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {val}")
    return fv


def _positive_float(val: str) -> float:
    fv = float(val)
    if fv <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {val}")
    return fv


def main():
    ap = argparse.ArgumentParser(
        description="Magnetometer Pipeline v2 — Production Edition"
    )
    ap.add_argument("--config", default=None, help="Path to YAML/JSON config file")
    ap.add_argument("--fetch-real-data", action="store_true")
    # Defaults are resolved after the config file loads, so an operator's
    # observatory: block is honoured instead of silently ignored.
    ap.add_argument("--observatory", default=None)
    ap.add_argument("--days", type=_positive_int, default=7)
    ap.add_argument("--start-date", default=None)
    ap.add_argument(
        "--warmup-days",
        type=_non_negative_float,
        default=3,
        help="Quiet days to fetch before start-date to seed the baseline.",
    )
    ap.add_argument("--cadence-s", type=_positive_int, default=None)
    ap.add_argument("--column", default=None)
    ap.add_argument("--cross-check-indices", action="store_true")
    ap.add_argument(
        "--output-json", default=None, help="Write machine-readable result to this path"
    )
    ap.add_argument("--log-format", choices=["text", "json"], default="text")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data availability without full compute",
    )
    ap.add_argument(
        "--state-file", default=STATE_FILE, help="Path for persistent QDC state"
    )
    ap.add_argument(
        "--no-state-save", action="store_true", help="Disable auto-save of QDC state"
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for cached upstream responses (default .magnetometer_cache)",
    )
    ap.add_argument(
        "--cache-ttl-hours",
        type=_non_negative_float,
        default=None,
        help="Max age of cached upstream responses before refetching.",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Always refetch upstream data instead of using the response cache.",
    )
    ap.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Fetch global indices after the magnetometer data instead of concurrently.",
    )
    ap.add_argument(
        "--flag-window-min",
        type=_non_negative_float,
        default=None,
        help="Window (minutes) for the disturbance amplitude; 0 = per-sample.",
    )
    ap.add_argument(
        "--flag-mode",
        choices=["range", "hybrid", "max", "instant"],
        default=None,
        help="Amplitude measure for activity tiers (default range).",
    )
    ap.add_argument(
        "--flag-centered",
        action="store_true",
        help="Center the amplitude window (retrospective only: uses future samples).",
    )
    ap.add_argument(
        "--legacy-flags",
        action="store_true",
        help="Restore the pre-calibration per-sample thresholds.",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help=(
            "Analyse the window ending now (near-real-time): enforces the "
            "data-freshness check and never serves cached upstream data."
        ),
    )
    ap.add_argument(
        "--loop-interval-min",
        type=_positive_float,
        default=None,
        help=(
            "Repeat the run every N minutes instead of exiting (supervised "
            "mode). Upstream/gate/stale failures are logged and retried."
        ),
    )
    ap.add_argument(
        "--max-latency-min",
        type=_non_negative_float,
        default=None,
        help="Fail the freshness check if the newest sample is older than this.",
    )
    ap.add_argument(
        "--fail-on-unhealthy",
        action="store_true",
        help="Exit non-zero when any health check fails (for schedulers).",
    )
    ap.add_argument(
        "--metrics-file",
        default=None,
        help="Write Prometheus textfile-collector metrics about this run here.",
    )
    ap.add_argument(
        "--alert-webhook",
        default=None,
        help=(
            f"Alert receiver URL (token from ${ALERT_TOKEN_ENV}, never an " "argument)."
        ),
    )
    ap.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args()

    setup_logging(level=getattr(logging, args.log_level), fmt=args.log_format)
    logger.debug(f"run_id={RUN_ID} argv={' '.join(sys.argv[1:])}")

    # Load config first, then CLI args override
    if args.config:
        load_config(args.config)

    if args.observatory is None:
        args.observatory = DEFAULT_OBSERVATORY
    if args.cadence_s is None:
        args.cadence_s = DEFAULT_CADENCE_S
    if args.column is None:
        args.column = DEFAULT_COLUMN
    LOG_CONTEXT.observatory = args.observatory

    # CLI overrides for state / caching / classification
    global STATE_AUTO_SAVE, HTTP_CACHE_ENABLED, HTTP_CACHE_DIR, HTTP_CACHE_TTL_HOURS
    global FLAG_AMPLITUDE_WINDOW_MIN, FLAG_AMPLITUDE_MODE, FLAG_AMPLITUDE_CENTERED
    if args.legacy_flags:
        for name, value in LEGACY_FLAG_SETTINGS.items():
            globals()[name] = value
    if args.flag_window_min is not None:
        FLAG_AMPLITUDE_WINDOW_MIN = args.flag_window_min
    if args.flag_mode is not None:
        FLAG_AMPLITUDE_MODE = args.flag_mode
    if args.flag_centered:
        FLAG_AMPLITUDE_CENTERED = True
    if args.no_state_save:
        STATE_AUTO_SAVE = False
    if args.no_cache:
        HTTP_CACHE_ENABLED = False
    if args.cache_dir is not None:
        HTTP_CACHE_DIR = args.cache_dir
    if args.cache_ttl_hours is not None:
        HTTP_CACHE_TTL_HOURS = args.cache_ttl_hours
    global ALERT_WEBHOOK_URL, MAX_DATA_LATENCY_MIN
    if args.alert_webhook is not None:
        ALERT_WEBHOOK_URL = args.alert_webhook
    if args.max_latency_min is not None:
        MAX_DATA_LATENCY_MIN = args.max_latency_min
    if args.live:
        # A live run must not be answered from the historical response cache,
        # and a fixed --start-date would contradict "ending now".
        HTTP_CACHE_ENABLED = False
        if args.start_date:
            raise PipelineError(
                "--live analyses the window ending now; drop --start-date.",
                EXIT_USAGE,
            )

    # CLI overrides go through the same validation as config files.
    validate_settings({k: globals()[k] for k in _SETTING_TYPES})

    state = PipelineState(args.state_file)
    if args.observatory and state.observatory and state.observatory != args.observatory:
        logger.warning(
            f"State observatory mismatch ({state.observatory} vs {args.observatory}). "
            f"Discarding stale state."
        )
        # Keep the same path (the next save overwrites it) but start cold: a
        # QDC fitted at another station is worse than no seed at all.
        state = PipelineState(args.state_file, load=False)
        state.observatory = args.observatory

    if args.fetch_real_data:
        if args.live:
            # Trailing window ending today; the freshness check below decides
            # whether what came back is recent enough to act on.
            now = datetime.now(timezone.utc)
            start_dt = pd.Timestamp(now).floor("D") - pd.Timedelta(days=args.days - 1)
            baseline_start_dt = start_dt - pd.Timedelta(days=args.warmup_days)
            baseline_start_str = baseline_start_dt.strftime("%Y-%m-%d")
            total_days = int(args.warmup_days + args.days)
            analysis_start_time = start_dt.to_pydatetime()
        elif args.start_date:
            start_dt = pd.to_datetime(args.start_date, utc=True)
            baseline_start_dt = start_dt - pd.Timedelta(days=args.warmup_days)
            baseline_start_str = baseline_start_dt.strftime("%Y-%m-%d")
            total_days = int(args.warmup_days + args.days)
            analysis_start_time = start_dt.to_pydatetime()
        else:
            baseline_start_str = None
            total_days = args.days
            analysis_start_time = None

        # The global indices only depend on the requested window, so warm them
        # in the background while the magnetometer fetch + parse runs. The
        # sequential code below then hits the cache instead of the network.
        prefetch = None
        if args.cross_check_indices and not args.no_prefetch:
            prefetch = ThreadPoolExecutor(max_workers=1)
            prefetch.submit(
                _prefetch_global_indices, baseline_start_str or "2024-01-01", total_days
            )

        try:
            iaga_text = fetch_intermagnet_iaga2002(
                observatory=args.observatory,
                start_date=baseline_start_str,
                duration_days=total_days,
            )
            df = parse_iaga2002_to_dataframe(iaga_text)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"INTERMAGNET data unavailable for {args.observatory}: {e}",
                EXIT_UPSTREAM_UNAVAILABLE,
            )
        finally:
            if prefetch is not None:
                prefetch.shutdown(wait=True)
        label = f"INTERMAGNET {args.observatory}"

        if args.live:
            # Day-granularity requests for "today" come back as a full day's
            # worth of rows: INTERMAGNET pads minutes that haven't happened
            # yet with the 99999 missing-value sentinel rather than omitting
            # them. parse_iaga2002_to_dataframe keeps the (real) timestamp for
            # every row it sees, sentinel or not, so those not-yet-happened
            # minutes survive as NaN rows stamped hours into the future. Left
            # in, they inflate the trailing gap past max_gap_samples (tanking
            # coverage/finite-sample counts) and make df.index.max() a
            # timestamp that hasn't occurred yet (tripping the clock-skew
            # check below on every single live run). Drop them before any
            # analysis or health check sees the frame - they are not missing
            # data, they are data that does not exist yet.
            _now = pd.Timestamp.now(tz="UTC")
            _pre_trim = len(df)

            # Ensure index comparison matches index timezone status
            if df.index.tz is None:
                df = df[df.index <= _now.tz_localize(None)]
            else:
                df = df[df.index <= _now]

            _dropped = _pre_trim - len(df)
            if _dropped:
                logger.info(
                    f"Trimmed {_dropped} not-yet-elapsed sample(s) from the live "
                    f"fetch (upstream pads today's file through midnight)."
                )
    else:
        raise PipelineError(
            "Must supply --fetch-real-data (or use --dry-run with --fetch-real-data)",
            EXIT_USAGE,
        )

    if df.empty:
        raise PipelineError(
            f"INTERMAGNET returned no samples for {args.observatory}.",
            EXIT_UPSTREAM_UNAVAILABLE,
        )
    if args.column not in df.columns:
        raise PipelineError(
            f"Column {args.column!r} not in upstream data "
            f"(available: {', '.join(df.columns)})",
            EXIT_USAGE,
        )

    # How much of the window we asked for actually came back, and how old its
    # newest sample is. Both are health signals, not gates on their own: a
    # historical re-run is legitimately "stale".
    newest = pd.to_datetime(df.index.max(), utc=True)
    data_latency_min = (pd.Timestamp.now(tz="UTC") - newest).total_seconds() / 60.0
    # expected_samples = total_days * 24 * 3600 / args.cadence_s
    if args.live:
        # The nominal window (baseline_start_dt + total_days) still runs
        # through end-of-day, but only the portion up to "now" could
        # possibly have arrived. Score coverage against what has actually
        # elapsed, not against hours that haven't happened - otherwise every
        # live run understates coverage by however much of today is left.
        _nominal_end = baseline_start_dt + pd.Timedelta(days=total_days)
        _elapsed_end = min(_nominal_end, pd.Timestamp.now(tz="UTC"))
        elapsed_seconds = max(0.0, (_elapsed_end - baseline_start_dt).total_seconds())
        expected_samples = elapsed_seconds / args.cadence_s
    else:
        expected_samples = total_days * 24 * 3600 / args.cadence_s

    finite_samples = int(
        np.isfinite(pd.to_numeric(df[args.column], errors="coerce")).sum()
    )

    requested_coverage = min(1.0, finite_samples / max(expected_samples, 1.0))
    if data_latency_min < -CLOCK_SKEW_TOLERANCE_MIN:
        logger.warning(
            f"Newest sample is timestamped {-data_latency_min:.0f} min in the future: "
            "local clock and station timestamps disagree. Freshness cannot be trusted."
        )
    else:
        logger.info(
            f"Upstream returned {requested_coverage:.1%} of the requested window; "
            f"newest sample is {max(data_latency_min, 0.0):.0f} min old."
        )

    x = handle_gaps(df[args.column], max_gap_samples=MAX_GAP_SAMPLES).values

    dst_series, kp_series = None, None
    if args.cross_check_indices:
        start_dt = pd.to_datetime(df.index.min())
        end_dt = pd.to_datetime(df.index.max())

        try:
            kp_series = fetch_kp_gfz(
                start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
            )
        except Exception as e:
            logger.warning(f"Kp cross-validation disabled: {e}")

        try:
            months = sorted(
                {
                    (dt.year, dt.month)
                    for dt in pd.date_range(start_dt, end_dt, freq="D")
                }
            )
            dst_parts = [fetch_dst_kyoto(y, m) for y, m in months]
            valid_dst = [p for p in dst_parts if p is not None]
            if valid_dst:
                dst_series = pd.concat(valid_dst).sort_index()
        except Exception as e:
            logger.warning(f"Dst cross-validation disabled: {e}")

    result = run_analysis(
        x,
        args.cadence_s,
        label=label,
        start_time=pd.to_datetime(df.index.min()).to_pydatetime(),
        analysis_start_time=analysis_start_time,
        dst_series=dst_series,
        kp_series=kp_series,
        state=state,
        dry_run=args.dry_run,
        live=args.live,
        data_latency_min=data_latency_min,
        requested_coverage=requested_coverage,
        observatory=args.observatory,
    )

    # Provenance
    if OUTPUT_INCLUDE_PROVENANCE:
        result["provenance"] = {
            "version": __version__,
            "run_id": RUN_ID,
            "git_hash": _git_hash(),
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "live" if args.live else "historical",
            "observatory": args.observatory,
            "column": args.column,
            "start_date": args.start_date,
            "warmup_days": args.warmup_days,
            "analysis_days": args.days,
            "cadence_s": args.cadence_s,
            "thresholds": {
                "unsettled": FLAG_THRESHOLD_UNSETTLED_NT,
                "active": FLAG_THRESHOLD_ACTIVE_NT,
                "minor_storm": FLAG_THRESHOLD_MINOR_STORM_NT,
                "major_storm": FLAG_THRESHOLD_MAJOR_STORM_NT,
                "severe_storm": FLAG_THRESHOLD_SEVERE_STORM_NT,
                "anomaly_jump": FLAG_THRESHOLD_ANOMALY_JUMP_NT,
                "amplitude_window_min": FLAG_AMPLITUDE_WINDOW_MIN,
                "amplitude_mode": FLAG_AMPLITUDE_MODE,
                "amplitude_centered": FLAG_AMPLITUDE_CENTERED,
                "max_plausible": MAX_PLAUSIBLE_RESIDUAL_NT,
                "min_plausible": MIN_PLAUSIBLE_RESIDUAL_NT,
            },
            "quality_gates": {
                "min_coverage": MIN_ANALYSIS_COVERAGE,
                "max_median_fill": MAX_MEDIAN_FILL_FRACTION,
            },
            "validation_source": result.get("validation_source", "none"),
        }

    # Save state with observatory tag
    if state is not None and STATE_AUTO_SAVE:
        state.observatory = args.observatory
        state.save(args.observatory)

    # JSON output
    if args.output_json:
        _write_json_output(result, args.output_json)
    if args.metrics_file:
        write_metrics_file(result, args.metrics_file, args.observatory)

    if result.get("status") == "insufficient_data":
        raise PipelineError(
            "Run did not pass the data-quality gate — no metrics were computed. "
            "See coverage/median_fill_frac above; re-run with a cleaner window "
            "or adjust MIN_ANALYSIS_COVERAGE / MAX_MEDIAN_FILL_FRACTION if this "
            "is expected for your data source.",
            EXIT_QUALITY_GATE,
        )

    if args.dry_run:
        logger.info("Dry run successful — all systems ready for production compute.")
        return result

    health = result.get("health", {})
    if not health.get("healthy", True):
        failed = [k for k, ok in health.get("checks", {}).items() if not ok]
        message = f"Run completed but is unhealthy: {', '.join(failed)}"
        if args.fail_on_unhealthy:
            # Stale data gets its own code so a scheduler can retry it rather
            # than page someone about a broken pipeline.
            code = (
                EXIT_STALE_DATA if failed == ["data_freshness"] else EXIT_QUALITY_GATE
            )
            raise PipelineError(message, code)
        logger.warning(message)

    return result


# Exit codes worth retrying on a schedule: the upstream or the data is at
# fault, not the invocation.
RETRYABLE_EXIT_CODES = (EXIT_QUALITY_GATE, EXIT_UPSTREAM_UNAVAILABLE, EXIT_STALE_DATA)


def _install_termination_handler() -> None:
    """Make SIGTERM behave like Ctrl-C so a supervised loop shuts down cleanly."""

    def handler(signum, frame):  # pragma: no cover - signal delivery
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, AttributeError):  # not the main thread / no SIGTERM
        pass


def run_loop(interval_min: float, argv: Optional[List[str]] = None) -> int:
    """Repeat the run every ``interval_min`` minutes until told to stop.

    A monitor must survive the things that routinely break: a GIN outage, a
    window that fails the quality gate, data that has gone stale. Those are
    logged and retried on the next tick; only a bad invocation or a bad config
    aborts, because retrying those forever would just hide the mistake.
    """
    _install_termination_handler()
    interval_s = max(1.0, interval_min * 60.0)
    logger.info(
        f"Supervised mode: running every {interval_min:g} min "
        f"(SIGTERM/Ctrl-C to stop)."
    )
    last_code = EXIT_OK
    while True:
        started = time.monotonic()
        code = run_cli(argv)
        last_code = code
        if code == 130:
            return EXIT_OK
        if code not in (EXIT_OK,) + RETRYABLE_EXIT_CODES:
            logger.error(f"Exiting supervised mode: exit code {code} is not retryable.")
            return code
        if code != EXIT_OK:
            logger.warning(f"Cycle failed with exit code {code}; retrying next tick.")
        sleep_s = max(0.0, interval_s - (time.monotonic() - started))
        try:
            time.sleep(sleep_s)
        except KeyboardInterrupt:
            logger.warning("Interrupted; shutting down.")
            return EXIT_OK
    return last_code  # pragma: no cover - unreachable


def run_cli(argv: Optional[List[str]] = None) -> int:
    """Translate outcomes into the documented exit codes (see RUNBOOK.md).

    Expected failures exit with their own code and a single-line message;
    anything unexpected exits EXIT_INTERNAL with a traceback, because an
    operator needs to tell "upstream is down" apart from "this program is
    broken".
    """
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    try:
        main()
        return EXIT_OK
    except PipelineError as e:
        logger.error(str(e))
        return e.exit_code
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130
    except SystemExit as e:  # argparse --help/--version and bad usage
        code = int(e.code or EXIT_OK)
        # argparse exits 2 on a usage error, which collides with the quality
        # gate; report bad invocations under the documented usage code.
        return EXIT_USAGE if code == 2 else code
    except Exception:
        logger.critical("Unhandled internal error", exc_info=True)
        return EXIT_INTERNAL


def write_metrics_file(result: Dict[str, Any], path: str, observatory: str) -> None:
    """Emit Prometheus textfile-collector metrics for this run.

    Written atomically because the node_exporter textfile collector may scrape
    the file at any moment and a half-written file is a parse error.
    """
    labels = f'observatory="{observatory}",version="{__version__}"'
    health = result.get("health", {})
    lines = [
        "# HELP mag_run_ok 1 if the run produced metrics that passed the quality gate.",
        "# TYPE mag_run_ok gauge",
        f"mag_run_ok{{{labels}}} {int(result.get('status') == 'ok')}",
        "# HELP mag_run_healthy 1 if every health check passed.",
        "# TYPE mag_run_healthy gauge",
        f"mag_run_healthy{{{labels}}} {int(bool(health.get('healthy')))}",
        "# HELP mag_run_timestamp_seconds Unix time this run finished.",
        "# TYPE mag_run_timestamp_seconds gauge",
        f"mag_run_timestamp_seconds{{{labels}}} {time.time():.0f}",
    ]

    def gauge(name: str, help_text: str, value: Any, extra: str = "") -> None:
        if value is None or not isinstance(value, (int, float)):
            return
        if not np.isfinite(value):
            return
        lines.extend(
            [
                f"# HELP {name} {help_text}",
                f"# TYPE {name} gauge",
                f"{name}{{{labels}{extra}}} {float(value):.6g}",
            ]
        )

    gauge(
        "mag_coverage_ratio",
        "Finite-sample coverage of the analysis window.",
        result.get("coverage"),
    )
    gauge(
        "mag_median_fill_ratio",
        "Fraction of baseline samples interpolated.",
        result.get("median_fill_frac"),
    )
    gauge(
        "mag_data_latency_minutes",
        "Age of the newest upstream sample.",
        health.get("data_latency_min"),
    )
    gauge(
        "mag_requested_coverage_ratio",
        "Fraction of the requested window returned.",
        health.get("requested_coverage"),
    )
    gauge(
        "mag_baseline_drift_nt",
        "Quiet-baseline offset shift vs the last good fit.",
        health.get("baseline_drift_nt"),
    )
    gauge(
        "mag_quiet_rms_nt",
        "Residual RMS over quiet-flagged samples.",
        health.get("quiet_rms_nt"),
    )

    # One series per check so an alert rule can name the failure instead of
    # only seeing mag_run_healthy go to 0.
    for check, ok in (health.get("checks") or {}).items():
        gauge(
            "mag_health_check",
            "1 if this health check passed.",
            int(bool(ok)),
            extra=f',check="{check}"',
        )

    for name, value in (result.get("metrics") or {}).items():
        gauge(f"mag_metric_{name}", f"Validation metric {name}.", value)
    for level, count in (result.get("flag_counts") or {}).items():
        gauge(
            "mag_flag_samples",
            "Samples per activity tier.",
            count,
            extra=f',level="{level}"',
        )

    body = "\n".join(lines) + "\n"
    try:
        target = Path(path)
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(body)
        tmp.replace(target)
        logger.info(f"Metrics written to {target}")
    except Exception as e:
        logger.error(f"Failed to write metrics file: {e}")


def _write_json_output(result: Dict[str, Any], path: str) -> None:
    """Serialize result to JSON atomically, handling numpy arrays gracefully."""
    serializable = _make_json_safe(result)
    try:
        target = Path(path)
        if target.parent and str(target.parent) != "":
            target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)
        logger.info(f"JSON output written to {target}")
    except Exception as e:
        # A consumer waiting on this file must not be handed a stale one
        # silently, so this is fatal.
        raise PipelineError(
            f"Failed to write JSON output to {path}: {e}", EXIT_INTERNAL
        )


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        if OUTPUT_INCLUDE_ARRAYS:
            return obj.tolist()
        else:
            return {"__type__": "ndarray", "shape": obj.shape, "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def main_entry(argv: Optional[List[str]] = None) -> int:
    """Console entry point: one shot, or a supervised loop with a bare
    ``--loop-interval-min`` pre-scan so the flag works before argparse runs."""
    args = list(sys.argv[1:] if argv is None else argv)
    interval: Optional[float] = None
    for i, token in enumerate(args):
        if token == "--loop-interval-min" and i + 1 < len(args):
            value = args[i + 1]
        elif token.startswith("--loop-interval-min="):
            value = token.split("=", 1)[1]
        else:
            continue
        try:
            interval = float(value)
        except ValueError:
            interval = None  # let argparse produce the usage error
        break
    if interval is not None and interval > 0:
        return run_loop(interval, args)
    return run_cli(argv)


if __name__ == "__main__":
    sys.exit(main_entry())
