"""Public magnetometer pipeline API.

This is the stable compatibility facade used by the CLI. Implementation
responsibilities are progressively delegated to focused submodules while the
legacy public call surface remains available.
"""

from . import legacy_core as _legacy
from .legacy_core import *  # noqa: F401,F403
from .legacy_core import main_entry, run_cli, run_loop

# Numerical baseline implementation extracted from the monolith.
from .baseline import build_design_matrix, handle_gaps, robust_harmonic_baseline

# Activity classification implementation extracted from the monolith.  The
# wrappers below preserve the legacy module's runtime configuration: config
# files and CLI options still update legacy_core globals, while the actual
# numerical work now executes in the focused classification module.
from . import classification as _classification

# Strict config loader: rejects unknown keys and accepts flat FLAG_* JSON.
from .config_strict import load_config as _strict_load_config


def load_config(path: str):
    """Load YAML/JSON config; unknown keys are fatal (EXIT_CONFIG_INVALID)."""
    return _strict_load_config(path)


def disturbance_amplitude(residual, cadence_s):
    return _classification.disturbance_amplitude(
        residual,
        cadence_s,
        window_min=_legacy.FLAG_AMPLITUDE_WINDOW_MIN,
        mode=_legacy.FLAG_AMPLITUDE_MODE,
        centered=_legacy.FLAG_AMPLITUDE_CENTERED,
    )


def flag_activity(residual, cadence_s=60.0):
    return _classification.flag_activity(
        residual,
        cadence_s,
        window_min=_legacy.FLAG_AMPLITUDE_WINDOW_MIN,
        mode=_legacy.FLAG_AMPLITUDE_MODE,
        centered=_legacy.FLAG_AMPLITUDE_CENTERED,
        unsettled_nt=_legacy.FLAG_THRESHOLD_UNSETTLED_NT,
        active_nt=_legacy.FLAG_THRESHOLD_ACTIVE_NT,
        minor_storm_nt=_legacy.FLAG_THRESHOLD_MINOR_STORM_NT,
        major_storm_nt=_legacy.FLAG_THRESHOLD_MAJOR_STORM_NT,
        severe_storm_nt=_legacy.FLAG_THRESHOLD_SEVERE_STORM_NT,
        anomaly_jump_nt=_legacy.FLAG_THRESHOLD_ANOMALY_JUMP_NT,
        max_plausible_nt=_legacy.MAX_PLAUSIBLE_RESIDUAL_NT,
        min_plausible_nt=_legacy.MIN_PLAUSIBLE_RESIDUAL_NT,
    )


def cross_validate_flags(local_flags, dst_vals, kp_vals):
    return _classification.cross_validate_flags(local_flags, dst_vals, kp_vals)


# Patch the legacy module's globals as well. Functions defined there resolve
# their collaborators at call time, so the existing run_analysis orchestration
# automatically uses the extracted classifier without changing its public API.
_legacy.disturbance_amplitude = disturbance_amplitude
_legacy.flag_activity = flag_activity
_legacy.cross_validate_flags = cross_validate_flags
_legacy.load_config = load_config

__all__ = [name for name in globals() if not name.startswith("_")]
