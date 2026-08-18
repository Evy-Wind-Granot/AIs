"""Persistent QDC state for warm restarts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import settings


class PipelineState:
    def __init__(self, path: str = settings.STATE_FILE, load: bool = True):
        self.path = Path(path)
        self.last_good_coeffs: Optional[np.ndarray] = None
        self.seed_coeffs: Optional[np.ndarray] = None
        self.seed_storm_frac = 1.0
        self.timestamp: Optional[str] = None
        self.observatory: Optional[str] = None
        if load:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.last_good_coeffs = np.asarray(data["last_good_coeffs"], dtype=float) if data.get("last_good_coeffs") else None
            self.seed_coeffs = np.asarray(data["seed_coeffs"], dtype=float) if data.get("seed_coeffs") else None
            for name in ("last_good_coeffs", "seed_coeffs"):
                values = getattr(self, name)
                if values is not None and not np.all(np.isfinite(values)):
                    raise ValueError(f"{name} contains non-finite values")
            self.seed_storm_frac = float(data.get("seed_storm_frac", 1.0))
            self.timestamp = data.get("timestamp")
            self.observatory = data.get("observatory")
        except Exception:
            self.clear()

    def clear(self) -> None:
        self.last_good_coeffs = None
        self.seed_coeffs = None
        self.seed_storm_frac = 1.0
        self.timestamp = None
        self.observatory = None

    def is_fresh(self, max_age_hours: Optional[float] = None) -> bool:
        if not self.timestamp:
            return False
        try:
            saved = datetime.fromisoformat(self.timestamp)
            age = datetime.now(timezone.utc) - saved
            return age < timedelta(hours=settings.STATE_MAX_AGE_HOURS if max_age_hours is None else max_age_hours)
        except Exception:
            return False

    def save(self, observatory: str) -> None:
        payload = {
            "last_good_coeffs": self.last_good_coeffs.tolist() if self.last_good_coeffs is not None else None,
            "seed_coeffs": self.seed_coeffs.tolist() if self.seed_coeffs is not None else None,
            "seed_storm_frac": float(self.seed_storm_frac),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observatory": observatory,
            "version": settings.__version__,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)


def assess_health(coverage: float, median_fill_frac: float, quiet_rms_nt: Optional[float], data_latency_min: Optional[float], requested_coverage: Optional[float], baseline_drift_nt: Optional[float], live: bool):
    checks = {
        "coverage": coverage >= settings.MIN_ANALYSIS_COVERAGE,
        "baseline_fit": median_fill_frac <= settings.MAX_MEDIAN_FILL_FRACTION,
    }
    if live:
        checks["data_freshness"] = data_latency_min is not None and -settings.CLOCK_SKEW_TOLERANCE_MIN <= data_latency_min <= settings.MAX_DATA_LATENCY_MIN
    if requested_coverage is not None:
        checks["requested_window_returned"] = requested_coverage >= settings.MIN_REQUESTED_COVERAGE
    if baseline_drift_nt is not None and np.isfinite(baseline_drift_nt):
        checks["baseline_stability"] = abs(baseline_drift_nt) <= settings.MAX_BASELINE_DRIFT_NT
    if quiet_rms_nt is not None and np.isfinite(quiet_rms_nt):
        checks["station_noise"] = quiet_rms_nt <= settings.EXPECTED_QUIET_RMS_MAX_NT
    return {
        "healthy": all(checks.values()),
        "checks": checks,
        "data_latency_min": data_latency_min,
        "requested_coverage": requested_coverage,
        "baseline_drift_nt": baseline_drift_nt,
        "quiet_rms_nt": quiet_rms_nt,
    }
