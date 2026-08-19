#!/usr/bin/env python3
"""Production magnetometer metrics and validation primitives."""
from __future__ import annotations

# NOTE: This file is intentionally not replaced in this commit. The repository
# uses performance_metrics.py as the canonical metrics module. This compatibility
# module exists only if an older deployment imports production_metrics.
from performance_metrics import *  # noqa: F401,F403
