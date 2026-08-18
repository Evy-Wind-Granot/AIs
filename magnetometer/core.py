"""Public magnetometer pipeline API.

This is the stable compatibility facade used by the CLI. Implementation
responsibilities are progressively delegated to focused submodules while the
legacy public call surface remains available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import legacy_core as _legacy
from .legacy_core import *  # noqa: F401,F403
from .legacy_core import main_entry, run_cli, run_loop

from .baseline import build_design_matrix, handle_gaps, robust_harmonic_baseline
from . import classification as _classification
from .live import LiveConfig, LiveDetector
from .config_strict import load_config as _strict_load_config


def load_config(path: str):
    """Load strict config and synchronize the compatibility module namespace."""
    cfg = _strict_load_config(path)
    for name in _legacy._SETTING_TYPES:
        globals()[name] = getattr(_legacy, name)
    return cfg


def disturbance_amplitude(residual, cadence_s):
    return _classification.disturbance_amplitude(
        residual,
        cadence_s,
        window_min=_legacy.FLAG_AMPLITUDE_WINDOW_MIN,
        mode=_legacy.FLAG_AMPLITUDE_MODE,
        centered=_legacy.FLAG_AMPLITUDE_CENTERED,
    )


def flag_activity(residual, cadence_s=60.0):
    return _classification.flag_activity(
        residual,
        cadence_s,
        window_min=_legacy.FLAG_AMPLITUDE_WINDOW_MIN,
        mode=_legacy.FLAG_AMPLITUDE_MODE,
        centered=_legacy.FLAG_AMPLITUDE_CENTERED,
        unsettled_nt=_legacy.FLAG_THRESHOLD_UNSETTLED_NT,
        active_nt=_legacy.FLAG_THRESHOLD_ACTIVE_NT,
        minor_storm_nt=_legacy.FLAG_THRESHOLD_MINOR_STORM_NT,
        major_storm_nt=_legacy.FLAG_THRESHOLD_MAJOR_STORM_NT,
        severe_storm_nt=_legacy.FLAG_THRESHOLD_SEVERE_STORM_NT,
        anomaly_jump_nt=_legacy.FLAG_THRESHOLD_ANOMALY_JUMP_NT,
        max_plausible_nt=_legacy.MAX_PLAUSIBLE_RESIDUAL_NT,
        min_plausible_nt=_legacy.MIN_PLAUSIBLE_RESIDUAL_NT,
    )


def cross_validate_flags(local_flags, dst_vals, kp_vals):
    return _classification.cross_validate_flags(local_flags, dst_vals, kp_vals)


def _attach_ml_forecast(result: Dict[str, Any], cadence_s: float, observatory: str) -> Dict[str, Any]:
    """Attach forecasts when a compatible trained artifact is available.

    The ML layer is intentionally fail-safe: absence/corruption of an artifact
    never changes the deterministic classification result or makes monitoring
    fail.  The artifact path is configurable through the
    ``MAGNETOMETER_FORECAST_MODEL`` environment variable; otherwise the
    conventional per-observatory path under ``models/artifacts`` is used.
    """
    import os
    from datetime import datetime, timezone
    import pandas as pd

    result["forecast"] = {"enabled": False, "status": "unavailable"}
    if result.get("status") != "ok":
        result["forecast"] = {"enabled": False, "status": "data_quality_gate"}
        return result

    default_path = Path("models") / "artifacts" / f"{observatory.lower()}_forecaster.pkl"
    artifact = Path(os.environ.get("MAGNETOMETER_FORECAST_MODEL", str(default_path)))
    if not artifact.exists():
        result["forecast"] = {
            "enabled": False,
            "status": "model_not_trained",
            "artifact": str(artifact),
        }
        return result

    try:
        from models.forecaster import build_training_data, load_model

        model = load_model(artifact)
        residual = np.asarray(result.get("residual"), dtype=float)
        start = result.get("analysis_start_time") or result.get("start_time")
        if start is None:
            # The legacy result does not always expose the timestamp, so use a
            # stable synthetic index only as a last-resort shape-preserving path.
            index = pd.date_range(
                datetime(1970, 1, 1, tzinfo=timezone.utc),
                periods=len(residual),
                freq=pd.Timedelta(seconds=cadence_s),
            )
        else:
            index = pd.date_range(
                pd.to_datetime(start, utc=True),
                periods=len(residual),
                freq=pd.Timedelta(seconds=cadence_s),
            )
        kp = result.get("kp")
        dst = result.get("dst")
        kp_series = pd.Series(np.asarray(kp, dtype=float), index=index) if kp is not None else None
        dst_series = pd.Series(np.asarray(dst, dtype=float), index=index) if dst is not None else None
        residual_series = pd.Series(residual, index=index)
        features, _ = build_training_data(
            residual_series,
            kp_series,
            dst_series,
            cadence_s=cadence_s,
            config=model.config,
        )
        forecasts = model.predict(features)

        levels = ("quiet", "unsettled", "active", "minor_storm", "major_storm", "severe_storm")
        current = str(np.asarray(result.get("flags"), dtype=object)[-1]) if len(result.get("flags", [])) else "unknown"
        current_rank = levels.index(current) if current in levels else -1
        peak_delta = 0
        for forecast in forecasts.values():
            rank = levels.index(forecast["predicted_tier"])
            peak_delta = max(peak_delta, rank - current_rank)
        result["forecast"] = {
            "enabled": True,
            "status": "ok",
            "artifact": str(artifact),
            "horizons": {str(k): v for k, v in forecasts.items()},
            "current_tier": current,
            "max_tier_delta": int(peak_delta),
            "early_warning": bool(peak_delta >= 2),
        }
    except Exception as exc:  # pragma: no cover - exercised by deployment failures
        _legacy.logger.exception("ML forecast unavailable; deterministic result retained")
        result["forecast"] = {
            "enabled": False,
            "status": "inference_error",
            "error": str(exc),
            "artifact": str(artifact),
        }
    return result


def run_analysis(*args, **kwargs):
    """Run the deterministic pipeline, then attach optional ML forecasts."""
    result = _legacy.run_analysis(*args, **kwargs)
    cadence_s = float(args[1] if len(args) > 1 else kwargs.get("cadence_s", 60.0))
    observatory = str(kwargs.get("observatory", "-"))
    return _attach_ml_forecast(result, cadence_s, observatory)


# Make the legacy module's internal run_loop/run_cli use the same wrapper. This
# preserves the existing CLI and live loop while adding the ML layer after each
# completed deterministic analysis.
_legacy.run_analysis = run_analysis
_legacy.disturbance_amplitude = disturbance_amplitude
_legacy.flag_activity = flag_activity
_legacy.cross_validate_flags = cross_validate_flags
_legacy.load_config = load_config

__all__ = [name for name in globals() if not name.startswith("_")]
