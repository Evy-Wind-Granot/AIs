#!/usr/bin/env python3
"""Backward-compatible entry point for the seismometer demo."""
import runpy

if __name__ == "__main__":
    runpy.run_module("seisometer.seismic_demo", run_name="__main__")
