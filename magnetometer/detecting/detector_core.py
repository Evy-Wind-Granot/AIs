"""Canonical import path for the production detector."""

from ..detector_core import (
    DETECTOR_VERSION,
    DetectorProfile,
    detect_activity_masks,
    flag_activity,
    load_detector_profile,
)

__all__ = [
    "DETECTOR_VERSION",
    "DetectorProfile",
    "detect_activity_masks",
    "flag_activity",
    "load_detector_profile",
]
