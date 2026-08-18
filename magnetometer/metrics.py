"""Validation metrics used by the magnetometer pipeline."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


class MetricsEngine:
    LOCAL_LEVELS = {
        "quiet": 0,
        "unsettled": 1,
        "active": 2,
        "minor_storm": 3,
        "major_storm": 4,
        "severe_storm": 4,
    }

    @staticmethod
    def _global_levels(kp_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp_vals, dtype=float)
        dst = np.asarray(dst_vals, dtype=float)
        with np.errstate(invalid="ignore"):
            kp_level = np.select([kp <= 2, kp <= 4, kp < 6, kp < 8], [0.0, 1.0, 2.0, 3.0], default=4.0)
            dst_level = np.select([dst >= -10, dst >= -30, dst >= -50, dst >= -100], [0.0, 1.0, 2.0, 3.0], default=4.0)
        kp_level[~np.isfinite(kp)] = np.nan
        dst_level[~np.isfinite(dst)] = np.nan
        return np.fmax(kp_level, dst_level)

    def compute(self, residual: np.ndarray, flags: np.ndarray, validation: np.ndarray, kp_vals: np.ndarray, dst_vals: np.ndarray) -> Dict[str, Any]:
        flags = np.asarray(flags, dtype=object)
        residual = np.asarray(residual, dtype=float)
        n = len(flags)
        local_levels = np.full(n, np.nan)
        for label, level in self.LOCAL_LEVELS.items():
            local_levels[flags == label] = level
        global_levels = self._global_levels(kp_vals, dst_vals)
        has_global = np.isfinite(global_levels)
        both = has_global & np.isfinite(local_levels)

        metrics: Dict[str, Any] = {}
        quiet = (flags == "quiet") & has_global & (global_levels == 0) & np.isfinite(residual)
        metrics["quiet_rms_nt"] = float(np.sqrt(np.nanmean(residual[quiet] ** 2))) if np.any(quiet) else np.nan

        truth_storm = has_global & (global_levels >= 3)
        pred_storm = np.isfinite(local_levels) & (local_levels >= 3)
        tp = int(np.sum(pred_storm & truth_storm))
        fn = int(np.sum(~pred_storm & truth_storm))
        fp = int(np.sum(pred_storm & has_global & (global_levels < 3)))
        tn = int(np.sum(~pred_storm & has_global & (global_levels < 3)))
        metrics["storm_detection_rate"] = float(tp / (tp + fn)) if tp + fn else np.nan
        metrics["false_alarm_rate"] = float(fp / (fp + tn)) if fp + tn else np.nan

        missed = int(np.sum(has_global & (global_levels >= 3) & np.isfinite(local_levels) & (local_levels < 2)))
        metrics["missed_global_event_rate"] = float(missed / np.sum(truth_storm)) if np.sum(truth_storm) else np.nan
        under = int(np.sum(has_global & (global_levels >= 2) & np.isfinite(local_levels) & (local_levels < 2)))
        active_total = int(np.sum(has_global & (global_levels >= 2)))
        metrics["under_reacting_rate"] = float(under / active_total) if active_total else np.nan
        local_storm_count = int(np.sum(pred_storm))
        unconfirmed = int(np.sum(pred_storm & has_global & (global_levels < 3)))
        metrics["unconfirmed_storm_rate"] = float(unconfirmed / local_storm_count) if local_storm_count else np.nan
        metrics["mean_abs_level_error"] = float(np.mean(np.abs(local_levels[both] - global_levels[both]))) if np.any(both) else np.nan
        metrics["validation_yield_ok"] = float(np.sum(np.asarray(validation, dtype=object) == "ok") / len(validation)) if len(validation) else np.nan
        metrics["samples_with_global_data"] = int(np.sum(has_global))
        metrics["total_samples"] = n
        return metrics
