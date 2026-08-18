"""Compatibility facade for the public magnetometer API.

The implementation is now organized under ``magnetometer.pipeline`` and the
specialized sibling modules. This file intentionally contains no scientific
implementation of its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .pipeline import settings
from .pipeline.analysis import run_analysis as _run_deterministic_analysis, write_json_output
from .pipeline.cli import main_entry, run_cli, run_loop, setup_logging
from .pipeline.config import load_config
from .classification import disturbance_amplitude as _disturbance_amplitude
from .classification import flag_activity as _flag_activity
from .classification import cross_validate_flags
from .live import LiveConfig, LiveDetector

__version__ = settings.__version__

# Public configuration constants are intentionally kept here as a facade for
# existing callers. The source of truth is magnetometer.pipeline.settings.
def _sync_settings() -> None:
    for name in settings.SETTING_TYPES:
        globals()[name] = getattr(settings, name)
    for name in (
        "INTERMAGNET_BASE", "KP_GFZ_URL", "DEFAULT_OBSERVATORY", "DEFAULT_CADENCE_S",
        "DEFAULT_COLUMN", "STATE_FILE", "STATE_AUTO_SAVE", "HTTP_CACHE_ENABLED",
        "HTTP_CACHE_DIR", "HTTP_CACHE_TTL_HOURS", "OUTPUT_INCLUDE_ARRAYS",
        "OUTPUT_INCLUDE_PROVENANCE", "MAX_PLAUSIBLE_RESIDUAL_NT",
        "MIN_PLAUSIBLE_RESIDUAL_NT",
    ):
        globals()[name] = getattr(settings, name)

_sync_settings()


def load_config_facade(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    _sync_settings()
    return cfg

load_config = load_config_facade


def disturbance_amplitude(residual, cadence_s):
    return _disturbance_amplitude(
        residual,
        cadence_s,
        window_min=settings.FLAG_AMPLITUDE_WINDOW_MIN,
        mode=settings.FLAG_AMPLITUDE_MODE,
        centered=settings.FLAG_AMPLITUDE_CENTERED,
    )


def flag_activity(residual, cadence_s=60.0):
    return _flag_activity(
        residual,
        cadence_s,
        window_min=settings.FLAG_AMPLITUDE_WINDOW_MIN,
        mode=settings.FLAG_AMPLITUDE_MODE,
        centered=settings.FLAG_AMPLITUDE_CENTERED,
        unsettled_nt=settings.FLAG_THRESHOLD_UNSETTLED_NT,
        active_nt=settings.FLAG_THRESHOLD_ACTIVE_NT,
        minor_storm_nt=settings.FLAG_THRESHOLD_MINOR_STORM_NT,
        major_storm_nt=settings.FLAG_THRESHOLD_MAJOR_STORM_NT,
        severe_storm_nt=settings.FLAG_THRESHOLD_SEVERE_STORM_NT,
        anomaly_jump_nt=settings.FLAG_THRESHOLD_ANOMALY_JUMP_NT,
        max_plausible_nt=settings.MAX_PLAUSIBLE_RESIDUAL_NT,
        min_plausible_nt=settings.MIN_PLAUSIBLE_RESIDUAL_NT,
    )


def run_analysis(*args, **kwargs):
    """Run deterministic analysis and attach an optional production ML forecast."""
    result = _run_deterministic_analysis(*args, **kwargs)
    try:
        _attach_ml_forecast(result, kwargs)
    except Exception:
        # ML must never break the deterministic pipeline.
        result.setdefault("forecast", {"enabled": False, "status": "inference_unavailable"})
    return result


_LEVELS = ("quiet", "unsettled", "active", "minor_storm", "major_storm", "severe_storm")


def _attach_ml_forecast(result: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
    """Attach approved ML horizons when a production artifact is available."""
    result["forecast"] = {"enabled": False, "status": "unavailable"}
    if result.get("status") != "ok":
        result["forecast"] = {"enabled": False, "status": "data_quality_gate"}
        return
    observatory = str(kwargs.get("observatory", "-"))
    artifact = Path(__file__).resolve().parent / "models" / "artifacts" / f"{observatory.lower()}_forecaster.pkl"
    import os
    artifact = Path(os.environ.get("MAGNETOMETER_FORECAST_MODEL", str(artifact)))
    if not artifact.exists():
        result["forecast"] = {"enabled": False, "status": "model_not_trained", "artifact": str(artifact)}
        return

    import pandas as pd
    from .models.forecaster import build_training_data, load_model

    model = load_model(artifact)
    approved = {int(h) for h in model.training_metadata.get("approved_horizons_hours", [])}
    if not approved:
        result["forecast"] = {"enabled": False, "status": "no_approved_horizons", "artifact": str(artifact)}
        return

    residual = np.asarray(result.get("residual"), dtype=float)
    anchor = kwargs.get("analysis_start_time") or kwargs.get("start_time")
    if anchor is None:
        raise ValueError("inference requires start_time or analysis_start_time")
    cadence_s = float(args_cadence := (kwargs.get("cadence_s") if "cadence_s" in kwargs else 60.0))
    index = pd.date_range(pd.to_datetime(anchor, utc=True), periods=len(residual), freq=pd.Timedelta(seconds=cadence_s))
    residual_series = pd.Series(residual, index=index)

    def align(source):
        if source is None:
            return None
        if isinstance(source, pd.Series):
            src = source.astype(float).copy()
            src.index = pd.DatetimeIndex(pd.to_datetime(src.index, utc=True))
            return src.reindex(index, method="ffill")
        values = np.asarray(source, dtype=float)
        return pd.Series(values, index=index) if len(values) == len(residual) else None

    features, _ = build_training_data(
        residual_series,
        align(kwargs.get("kp_series")),
        align(kwargs.get("dst_series")),
        cadence_s=cadence_s,
        config=model.config,
    )
    predictions = model.predict(features)
    approved_predictions = {h: value for h, value in predictions.items() if int(h) in approved}
    experimental = {h: value for h, value in predictions.items() if int(h) not in approved}

    flags = np.asarray(result.get("flags", []), dtype=object)
    current = str(flags[-1]) if len(flags) else "unknown"
    rank = _LEVELS.index(current) if current in _LEVELS else None
    deltas = {}
    if rank is not None:
        for horizon, forecast in approved_predictions.items():
            deltas[int(horizon)] = _LEVELS.index(forecast["predicted_tier"]) - rank
    max_signed = max(deltas.values(), default=0)
    max_abs = max((abs(v) for v in deltas.values()), default=0)
    direction = "escalating" if max_signed >= 2 else "decaying" if min(deltas.values(), default=0) <= -2 else "none"

    result["forecast"] = {
        "enabled": True,
        "status": "ok",
        "artifact": str(artifact),
        "model_health": model.health_check(),
        "horizons": {str(k): v for k, v in approved_predictions.items()},
        "experimental_horizons": {str(k): v for k, v in experimental.items()},
        "current_tier": current,
        "max_tier_delta": int(max_abs),
        "signed_tier_delta": int(max_signed),
        "early_warning": bool(max_signed >= 2),
    }
    result["hybrid"] = {
        "real_time": {"tier": current, "source": "deterministic_qdc"},
        "forecasted_status": {str(k): v["predicted_tier"] for k, v in approved_predictions.items()},
        "model_confidence": {str(k): float(v["confidence"]) for k, v in approved_predictions.items()},
        "divergence": {
            "tier_delta": int(max_abs),
            "signed_tier_delta": int(max_signed),
            "significant": bool(max_abs >= 2),
            "direction": direction,
            "anomaly_delta": int(max_abs),
        },
        "experimental_horizons": sorted(int(h) for h in experimental),
    }


__all__ = [name for name in globals() if not name.startswith("_")]
