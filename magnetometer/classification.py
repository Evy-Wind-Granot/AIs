"""Vectorized geomagnetic activity classification.

This module contains the production classification implementation extracted from
``legacy_core``.  The public functions accept the classification settings
explicitly so the legacy configuration system can continue to override values
without coupling this module back to the monolith.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


_FLAG_INVALID, _FLAG_QUIET, _FLAG_UNSETTLED, _FLAG_ACTIVE, _FLAG_MINOR, _FLAG_MAJOR, _FLAG_SEVERE, _FLAG_ANOMALY = range(8)
_FLAG_LABELS = np.array(
    [
        "invalid",
        "quiet",
        "unsettled",
        "active",
        "minor_storm",
        "major_storm",
        "severe_storm",
        "anomaly",
    ],
    dtype=object,
)


def disturbance_amplitude(
    residual: np.ndarray,
    cadence_s: float,
    *,
    window_min: float,
    mode: str,
    centered: bool,
) -> np.ndarray:
    """Return per-sample disturbance amplitude in nT.

    ``range`` uses the peak-to-peak spread, ``hybrid`` also considers sustained
    offset, ``max`` uses the largest absolute excursion, and ``instant`` keeps
    the legacy absolute-residual behaviour.
    """
    residual = np.asarray(residual, dtype=float)
    r = np.abs(residual)
    window = int(round(window_min * 60.0 / max(cadence_s, 1e-9)))
    if mode == "instant" or window <= 1 or len(residual) == 0:
        return r

    window = min(window, len(residual))
    roll = pd.Series(residual).rolling(
        window, center=bool(centered), min_periods=1
    )
    if mode == "range":
        amp = (roll.max() - roll.min()).to_numpy()
    elif mode == "hybrid":
        amp = np.maximum(
            (roll.max() - roll.min()).to_numpy(),
            2.0 * np.abs(roll.mean().to_numpy()),
        )
    elif mode == "max":
        amp = np.maximum(roll.max().abs().to_numpy(), roll.min().abs().to_numpy())
    else:
        raise ValueError(f"Unknown FLAG_AMPLITUDE_MODE: {mode}")

    amp[~np.isfinite(residual)] = np.nan
    return amp


def flag_activity(
    residual: np.ndarray,
    cadence_s: float = 60.0,
    *,
    window_min: float,
    mode: str,
    centered: bool,
    unsettled_nt: float,
    active_nt: float,
    minor_storm_nt: float,
    major_storm_nt: float,
    severe_storm_nt: float,
    anomaly_jump_nt: float,
    max_plausible_nt: float,
    min_plausible_nt: float,
) -> np.ndarray:
    """Classify residual samples into the production activity tiers."""
    residual = np.asarray(residual, dtype=float)
    r = np.abs(residual)
    finite = np.isfinite(residual)
    amplitude = disturbance_amplitude(
        residual,
        cadence_s,
        window_min=window_min,
        mode=mode,
        centered=centered,
    )

    # Integer codes avoid repeated writes into an object array; labels are
    # materialized once at the end.
    codes = np.full(len(residual), _FLAG_INVALID, dtype=np.int8)
    codes[finite] = _FLAG_QUIET

    for code, threshold in (
        (_FLAG_UNSETTLED, unsettled_nt),
        (_FLAG_ACTIVE, active_nt),
        (_FLAG_MINOR, minor_storm_nt),
        (_FLAG_MAJOR, major_storm_nt),
        (_FLAG_SEVERE, severe_storm_nt),
    ):
        codes[finite & (amplitude > threshold)] = code

    diff = np.diff(residual, prepend=residual[0])
    rebound = np.roll(diff, -1)
    rebound[-1] = 0.0
    big = np.abs(diff) > anomaly_jump_nt
    spike = (
        finite
        & np.isfinite(diff)
        & np.isfinite(rebound)
        & big
        & (np.abs(rebound) > anomaly_jump_nt)
        & (np.sign(rebound) != np.sign(diff))
    )
    codes[spike] = _FLAG_ANOMALY

    implausible = finite & (
        (r > max_plausible_nt) | (residual < min_plausible_nt)
    )
    if np.any(implausible):
        codes[implausible] = _FLAG_INVALID

    return _FLAG_LABELS[codes]


def cross_validate_flags(
    local_flags: np.ndarray,
    dst_vals: np.ndarray,
    kp_vals: np.ndarray,
) -> np.ndarray:
    """Compare local classifications against global Dst/Kp activity."""
    local_flags = np.asarray(local_flags, dtype=object)
    dst_vals = np.asarray(dst_vals, dtype=float)
    kp_vals = np.asarray(kp_vals, dtype=float)

    with np.errstate(invalid="ignore"):
        main_phase = (dst_vals < -50) | (kp_vals >= 6)
        active = (dst_vals < -30) | (kp_vals >= 4)

    quiet = local_flags == "quiet"
    big_storm = (local_flags == "major_storm") | (local_flags == "severe_storm")

    validation = np.full(len(local_flags), "ok", dtype=object)
    validation[quiet & main_phase] = "missed_global_event"
    validation[quiet & ~main_phase & active] = "under_reacting"
    validation[big_storm & ~(active | main_phase)] = "unconfirmed_storm"
    return validation


__all__ = ["disturbance_amplitude", "flag_activity", "cross_validate_flags"]
