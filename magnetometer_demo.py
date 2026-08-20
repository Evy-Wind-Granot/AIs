#!/usr/bin/env python3
"""Backward-compatible entry point for the magnetometer detector demo."""
import runpy

if __name__ == "__main__":
    runpy.run_module("magnetometer.detector.magnetometer_demo", run_name="__main__")
