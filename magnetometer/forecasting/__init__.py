"""Causal magnetometer forecasting package.

The original forecasting modules use local imports, so the package keeps its
forecasting directory importable for backward compatibility during this
reorganization.
"""
from pathlib import Path
import sys

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)
