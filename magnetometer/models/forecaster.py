#!/usr/bin/env python3
"""Compatibility entry point for the certified production geomagnetic forecaster.

The implementation remains in ``production_forecaster.py``; this compatibility
module exposes the certification wrapper so existing imports and training
entry points receive the hardened temporal threshold/calibration behavior.
"""
from __future__ import annotations

from .production_forecaster import ForecastConfig, ForecastResult, evaluate_forecast
from .certified_forecaster import GeomagneticForecaster

__all__ = [
    "ForecastConfig",
    "ForecastResult",
    "GeomagneticForecaster",
    "evaluate_forecast",
]
