"""Compatibility exports for the historical validation implementation.

The implementation now lives in ``magnetometer.validation.historical``.
This module preserves the established import path for older calibration,
validation, and replay tooling while the repository is reorganized.
"""
from .historical.validate_historical_magnetometer import *  # noqa: F401,F403
