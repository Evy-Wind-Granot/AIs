"""Public magnetometer package surface.

The executable implementation lives in ``magnetometer.pipeline`` and the
specialized sibling modules. ``core`` remains a small compatibility facade for
public imports and ML forecast integration.
"""
from .core import *  # noqa: F401,F403
from .acquisition import AcquisitionClient, DEFAULT_ACQUISITION
from .cache import ResponseCache
from .parsing import parse_iaga2002_to_dataframe
from .live import LiveConfig, LiveDetector

__all__ = [name for name in globals() if not name.startswith("_")]
