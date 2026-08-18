#!/usr/bin/env python3
"""Compatibility CLI for the production magnetometer package.

The implementation lives in :mod:`magnetometer.core`. This module remains at
its historical path so existing scripts, imports, and documented commands keep
working without modification.
"""

import sys

from magnetometer import core as _core
from magnetometer.core import *  # noqa: F401,F403
from magnetometer.core import main_entry


def load_config(path: str):
    """Load configuration and keep this legacy compatibility module in sync.

    ``from magnetometer.core import *`` copies scalar settings at import time.
    Without this wrapper, callers of the historical ``magnetometer_demo`` API
    would continue seeing stale FLAG_* values after a config reload.
    """
    cfg = _core.load_config(path)
    for name in _core._legacy._SETTING_TYPES:
        globals()[name] = getattr(_core._legacy, name)
    return cfg


if __name__ == "__main__":
    sys.exit(main_entry())
