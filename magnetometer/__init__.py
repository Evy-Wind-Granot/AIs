"""Magnetometer production package.

Public processing functions are re-exported from :mod:`magnetometer.core`
while the CLI compatibility module remains at ``magnetometer_demo.py``.
"""

from .core import *  # noqa: F401,F403

try:
    from .core import main_entry
except ImportError:  # pragma: no cover - defensive for partial installs
    main_entry = None

__all__ = [name for name in globals() if not name.startswith("_")]
