#!/usr/bin/env python3
"""Compatibility entry point for the production geomagnetic forecaster.

The implementation lives in ``production_forecaster.py``. This module keeps
historical imports stable for callers and tests.
"""
from __future__ import annotations

from .production_forecaster import (
    ForecastConfig,
    ForecastResult,
    GeomagneticForecaster,
    evaluate_forecast,
)

__all__ = [
    "ForecastConfig",
    "ForecastResult",
    "GeomagneticForecaster",
    "evaluate_forecast",
]
