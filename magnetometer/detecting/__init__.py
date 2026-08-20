"""Production geomagnetic activity detection package."""

from .detector_core import DetectorProfile, detect_activity_masks, flag_activity, load_detector_profile
from .live_detector import DetectionResult, MagnetometerDetector

__all__ = [
    "DetectorProfile",
    "detect_activity_masks",
    "flag_activity",
    "load_detector_profile",
    "DetectionResult",
    "MagnetometerDetector",
]
