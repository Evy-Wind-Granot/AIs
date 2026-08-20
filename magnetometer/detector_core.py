"""Legacy import shim for the detector core.

Canonical detector code now lives in ``magnetometer.detecting.detector_core``.
This module remains so older scripts and release gates keep working while the
package layout settles.
"""

from magnetometer.detecting.detector_core import DetectorProfile, detect_activity_masks, flag_activity, load_detector_profile

__all__ = ["DetectorProfile", "detect_activity_masks", "flag_activity", "load_detector_profile"]
