"""Public magnetometer pipeline API with optional ML forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import legacy_core as _legacy
from .legacy_core import *  # noqa: F401,F403
from .legacy_core import main_entry, run_cli, run_loop
from .baseline import build_design_matrix, handle_gaps, robust_harmonic_baseline
from . import classification as _classification
from .live import LiveConfig, LiveDetector
from .config_strict import load_config as _strict_load_config

_DETERMINISTIC_RUN_ANALYSIS = _legacy.run_analysis
_LEVELS = ("quiet", "unsettled", "active", "minor_storm", "major_storm", "severe_storm")


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


def _set_deterministic_hybrid(result: Dict[str, Any]) -> None:
    """Populate unified status fields when ML is unavailable."""
    flags = np.asarray(result.get("flags", []), dtype=object)
    current = str(flags[-1]) if len(flags) else "unknown"
    result["hybrid"] = {
        "real_time": {"tier": current},
        "forecasted_status": {},
        "model_confidence": None,
        "divergence": {
            "tier_delta": 0,
            "significant": False,
            "direction": "none",
        },
    }


def _attach_ml_forecast(
    result: Dict[str, Any],
    *,
    cadence_s: float,
    observatory: str,
    start_time: Any,
    analysis_start_time: Any,
    kp_series: Any,
    dst_series: Any,
) -> Dict[str, Any]:
    """Attach a fail-safe short-horizon ML forecast."""
    import os
    import pandas as pd

    _set_deterministic_hybrid(result)
    result["forecast"] = {"enabled": False, "status": "unavailable"}
    if result.get("status") != "ok":
        result["forecast"] = {
            "enabled": False,
            "status": "data_quality_gate",
        }
        return result

    default_artifact = Path("models") / "artifacts" / (
        f"{observatory.lower()}_forecaster.pkl"
    )
    artifact = Path(
        os.environ.get("MAGNETOMETER_FORECAST_MODEL", str(default_artifact))
    )
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
        anchor = analysis_start_time if analysis_start_time is not None else start_time
        if anchor is None:
            raise ValueError("inference requires start_time or analysis_start_time")
        index = pd.date_range(
            pd.to_datetime(anchor, utc=True),
            periods=len(residual),
            freq=pd.Timedelta(seconds=cadence_s),
        )
        residual_series = pd.Series(residual, index=index)

        def align(source: Any) -> pd.Series | None:
            if source is None:
                return None
            if isinstance(source, pd.Series):
                src = source.astype(float).copy()
                src.index = pd.DatetimeIndex(pd.to_datetime(src.index, utc=True))
                return src.reindex(index, method="ffill")
            values = np.asarray(source, dtype=float)
            if len(values) != len(residual):
                return None
            return pd.Series(values, index=index)

        features, _ = build_training_data(
            residual_series,
            align(kp_series),
            align(dst_series),
            cadence_s=cadence_s,
            config=model.config,
        )
        forecasts = model.predict(features)
        flags = np.asarray(result.get("flags", []), dtype=object)
        current = str(flags[-1]) if len(flags) else "unknown"
        current_rank = _LEVELS.index(current) if current in _LEVELS else -1
        deltas = [
            _LEVELS.index(f["predicted_tier"]) - current_rank
            for f in forecasts.values()
        ]
        max_delta = max(deltas, default=0)
        max_abs_delta = max((abs(delta) for delta in deltas), default=0)
        direction = "none"
        if max_delta >= 2:
            direction = "escalating"
        elif min(deltas, default=0) <= -2:
            direction = "decaying"
        confidence = float(
            np.mean([f["confidence"] for f in forecasts.values()])
        )

        result["forecast"] = {
            "enabled": True,
            "status": "ok",
            "artifact": str(artifact),
            "horizons": {str(k): v for k, v in forecasts.items()},
            "current_tier": current,
            "max_tier_delta": int(max_delta),
            "early_warning": bool(max_delta >= 2),
        }
        result["hybrid"] = {
            "real_time": {"tier": current},
            "forecasted_status": {
                str(k): v["predicted_tier"] for k, v in forecasts.items()
            },
            "model_confidence": confidence,
            "divergence": {
                "tier_delta": int(max_abs_delta),
                "signed_tier_delta": int(max_delta),
                "significant": bool(max_abs_delta >= 2),
                "direction": direction,
            },
        }
    except Exception as exc:  # pragma: no cover - deployment failure path
        _legacy.logger.exception(
            "ML forecast unavailable; deterministic result retained"
        )
        result["forecast"] = {
            "enabled": False,
            "status": "inference_error",
            "error": str(exc),
            "artifact": str(artifact),
        }
    return result


def run_analysis(*args, **kwargs):
    """Run deterministic analysis and append optional ML forecasts."""
    result = _DETERMINISTIC_RUN_ANALYSIS(*args, **kwargs)
    cadence_s = float(args[1] if len(args) > 1 else kwargs.get("cadence_s", 60.0))
    return _attach_ml_forecast(
        result,
        cadence_s=cadence_s,
        observatory=str(kwargs.get("observatory", "-")),
        start_time=kwargs.get("start_time"),
        analysis_start_time=kwargs.get("analysis_start_time"),
        kp_series=kwargs.get("kp_series"),
        dst_series=kwargs.get("dst_series"),
    )


# Existing legacy batch/live loops resolve run_analysis by module-global lookup.
_legacy.run_analysis = run_analysis
_legacy.disturbance_amplitude = disturbance_amplitude
_legacy.flag_activity = flag_activity
_legacy.cross_validate_flags = cross_validate_flags
_legacy.load_config = load_config

__all__ = [name for name in globals() if not name.startswith("_")]
