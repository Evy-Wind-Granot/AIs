"""Baseline/QDC processing API."""
from .legacy_core import (
    build_design_matrix,
    robust_harmonic_baseline,
    handle_gaps,
)

__all__ = ["build_design_matrix", "robust_harmonic_baseline", "handle_gaps"]
