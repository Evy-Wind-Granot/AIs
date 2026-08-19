"""Magnetometer detection and forecasting package.

The public package paths are ``magnetometer.detecting`` and
``magnetometer.forecasting``.  A small compatibility layer keeps older
root-level modules importable while those implementation files are migrated to
package-relative imports.
"""

from __future__ import annotations

import importlib
import sys

# Legacy implementation modules still use historical top-level imports such as
# ``from feature_engineering import ...`` and ``from models.forecaster import``.
# Register those names only as aliases to the package implementations so the
# canonical package imports work from pytest, ``python -m``, and direct scripts.
if "feature_engineering" not in sys.modules:
    sys.modules["feature_engineering"] = importlib.import_module(".feature_engineering", __name__)
if "models" not in sys.modules:
    sys.modules["models"] = importlib.import_module(".models", __name__)

__all__ = ["detecting", "forecasting"]
