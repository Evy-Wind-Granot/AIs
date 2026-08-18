"""Shared runtime settings for the magnetometer pipeline.

This module contains only configuration defaults, exit codes, and small immutable
constants. Scientific algorithms live in the sibling modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

__version__ = "2.0.0"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_QUALITY_GATE = 2
EXIT_UPSTREAM_UNAVAILABLE = 3
EXIT_CONFIG_INVALID = 4
EXIT_STALE_DATA = 5
EXIT_INTERNAL = 70

INTERMAGNET_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
KP_GFZ_URL = "https://kp.gfz-potsdam.de/app/json/"
DEFAULT_OBSERVATORY = "VIC"
DEFAULT_CADENCE_S = 60
DEFAULT_COLUMN = "x_nt"
DEFAULT_SAMPLES_PER_DAY = "Minute"
USER_AGENT = f"MagnetometerProductionPipeline/{__version__}"

MIN_ANALYSIS_COVERAGE = 0.90
MAX_MEDIAN_FILL_FRACTION = 0.20
INPUT_NAN_WARNING_THRESHOLD = 0.01

FLAG_AMPLITUDE_WINDOW_MIN = 180.0
FLAG_AMPLITUDE_MODE = "range"
FLAG_AMPLITUDE_CENTERED = False
FLAG_THRESHOLD_UNSETTLED_NT = 20.0
FLAG_THRESHOLD_ACTIVE_NT = 30.0
FLAG_THRESHOLD_MINOR_STORM_NT = 100.0
FLAG_THRESHOLD_MAJOR_STORM_NT = 400.0
FLAG_THRESHOLD_SEVERE_STORM_NT = 800.0
FLAG_THRESHOLD_ANOMALY_JUMP_NT = 100.0
MAX_PLAUSIBLE_RESIDUAL_NT = 3000.0
MIN_PLAUSIBLE_RESIDUAL_NT = -3000.0

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

BASELINE_N_ITER = 4
BASELINE_OUTLIER_THRESHOLD_NT = 30.0
BASELINE_WINDOW_HOURS = 24
BASELINE_STEP_HOURS = 12
MAX_GAP_SAMPLES = 3
STORM_FRACTION_THRESHOLD = 0.05

STATE_FILE = ".magnetometer_state.json"
STATE_AUTO_SAVE = True
STATE_MAX_AGE_HOURS = 168.0

OUTPUT_INCLUDE_ARRAYS = False
OUTPUT_INCLUDE_PROVENANCE = True

HTTP_CACHE_ENABLED = True
HTTP_CACHE_DIR = ".magnetometer_cache"
HTTP_CACHE_TTL_HOURS = 24.0
BASELINE_CACHE_ENABLED = True
BASELINE_CACHE_SIZE = 4

ALERT_WEBHOOK_URL = None
ALERT_WEBHOOK_MIN_LEVEL = "minor_storm"
ALERT_WEBHOOK_TIMEOUT_S = 10.0
ALERT_TOKEN_ENV = "MAG_ALERT_TOKEN"

MAX_DATA_LATENCY_MIN = 90.0
CLOCK_SKEW_TOLERANCE_MIN = 5.0
MIN_REQUESTED_COVERAGE = 0.80
MAX_BASELINE_DRIFT_NT = 50.0
EXPECTED_QUIET_RMS_MAX_NT = 30.0

STORM_LEVEL_ORDER = ("minor_storm", "major_storm", "severe_storm")
VALID_AMPLITUDE_MODES = ("range", "hybrid", "max", "instant")

SETTING_TYPES: Dict[str, Tuple[type, ...]] = {
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

CONFIG_MAPPING = {
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

RETRYABLE_EXIT_CODES = (EXIT_QUALITY_GATE, EXIT_UPSTREAM_UNAVAILABLE, EXIT_STALE_DATA)
DEFAULT_STATE_PATH = Path(STATE_FILE)
