"""Public magnetometer pipeline API.

This is the stable compatibility facade used by the CLI. Implementation
responsibilities are progressively delegated to focused submodules while the
legacy public call surface remains available.
"""

from . import legacy_core as _legacy
from .legacy_core import *  # noqa: F401,F403
from .legacy_core import main_entry, run_cli, run_loop

from .baseline import build_design_matrix, handle_gaps, robust_harmonic_baseline
from . import classification as _classification
from .live import LiveConfig, LiveDetector
from .config_strict import load_config as _strict_load_config


def load_config(path: str):
    """Load strict config and synchronize the compatibility module namespace.

    ``magnetometer_demo`` historically imports this facade with ``import *``.
    Updating only ``legacy_core`` therefore left stale FLAG_* values in the
    compatibility module.  Keep both namespaces synchronized after every load.
    """
    cfg = _strict_load_config(path)
    for name in _legacy._SETTING_TYPES:
        globals()[name] = getattr(_legacy, name)
    return cfg


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


_legacy.disturbance_amplitude = disturbance_amplitude
_legacy.flag_activity = flag_activity
_legacy.cross_validate_flags = cross_validate_flags
_legacy.load_config = load_config

__all__ = [name for name in globals() if not name.startswith("_")]
