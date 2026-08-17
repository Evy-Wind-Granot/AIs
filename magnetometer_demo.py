#!/usr/bin/env python3
"""Compatibility CLI for the production magnetometer package.

The implementation lives in :mod:`magnetometer.core`.  This module remains at
its historical path so existing scripts, imports, and documented commands keep
working without modification.
"""

import sys

from magnetometer.core import *  # noqa: F401,F403
from magnetometer.core import main_entry


if __name__ == "__main__":
    sys.exit(main_entry())
