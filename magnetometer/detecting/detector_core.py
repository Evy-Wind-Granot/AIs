"""Canonical import path for the production detector.

The implementation remains in ``magnetometer.detector_core`` during the
transition so existing CLI entry points keep working. New code should import
from ``magnetometer.detecting`` instead.
"""

from ..detector_core import (
    DetectorProfile,
    detect_activity_masks,
    flag_activity,
    load_detector_profile,
)

__all__ = ["DetectorProfile", "detect_activity_masks", "flag_activity", "load_detector_profile"]
