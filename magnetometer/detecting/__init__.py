"""Geomagnetic activity detection package.

Canonical detector imports live under ``magnetometer.detecting``.  Legacy
modules at ``magnetometer/`` remain compatibility entry points for now.
"""

from .detector_core import DetectorProfile, detect_activity_masks, flag_activity, load_detector_profile

__all__ = ["DetectorProfile", "detect_activity_masks", "flag_activity", "load_detector_profile"]
