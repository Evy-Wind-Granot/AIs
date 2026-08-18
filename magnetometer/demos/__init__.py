"""Demo and runner entry points.

``magnetometer_demo`` is retained only as an import-level alias for older
internal callers; there is no magnetometer_demo.py implementation anymore.
"""
from importlib import import_module


def __getattr__(name):
    if name == "magnetometer_demo":
        return import_module("magnetometer.pipeline")
    raise AttributeError(name)


__all__ = ["magnetometer_demo"]
