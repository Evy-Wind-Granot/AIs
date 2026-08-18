#!/usr/bin/env python3
"""Strict release gate for the ML forecasting subsystem."""
from __future__ import annotations

from typing import Any, Mapping

DEFAULT_CRITERIA = {
    "1": {"min_recall": 0.75, "min_precision": 0.85, "min_f1": 0.80, "max_far": 0.01, "max_ece": 0.08, "require_mae_beats_persistence": True},
    "3": {"min_recall": 0.60, "min_precision": 0.75, "min_f1": 0.65, "max_far": 0.01, "max_ece": 0.08, "require_mae_beats_persistence": True},
    "6": {"min_recall": 0.50, "min_precision": 0.70, "min_f1": 0.55, "max_far": 0.01, "max_ece": 0.08, "require_mae_beats_persistence": True},
}


def _ok(value: float | None, minimum: float | None = None, maximum: float | None = None) -> bool:
    if value is None:
        return False
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def evaluate_forecast_release(
    final_report: Mapping[str, Any],
    *,
    criteria: Mapping[str, Mapping[str, Any]] = DEFAULT_CRITERIA,
    minimum_final_samples_per_horizon: int = 1000,
) -> dict[str, Any]:
    """Return machine-enforceable certification checks for all horizons."""
    horizon_checks: dict[str, Any] = {}
    for horizon, rule in criteria.items():
        metrics = final_report.get(horizon)
        if not metrics:
            horizon_checks[horizon] = {"passed": False, "reason": "missing_horizon"}
            continue
        storm = metrics.get("storm", {})
        checks = {
            "minimum_final_samples": int(metrics.get("samples", 0)) >= minimum_final_samples_per_horizon,
            "storm_recall": _ok(storm.get("recall"), rule.get("min_recall")),
            "storm_precision": _ok(storm.get("precision"), rule.get("min_precision")),
            "storm_f1": _ok(storm.get("f1"), rule.get("min_f1")),
            "storm_false_alarm_rate": _ok(storm.get("false_alarm_rate"), maximum=rule.get("max_far")),
            "probability_ece": _ok(storm.get("ece"), maximum=rule.get("max_ece")),
        }
        if rule.get("require_mae_beats_persistence", True):
            checks["amplitude_mae_beats_persistence"] = bool(metrics.get("beats_persistence_mae", False))
        passed = all(checks.values())
        horizon_checks[horizon] = {
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
            "criteria": dict(rule),
        }
    return {
        "passed": all(v.get("passed", False) for v in horizon_checks.values()),
        "criteria": {k: dict(v) for k, v in criteria.items()},
        "minimum_final_samples_per_horizon": minimum_final_samples_per_horizon,
        "horizons": horizon_checks,
    }


if __name__ == "__main__":
    raise SystemExit("Import evaluate_forecast_release or use train_magnetometer_forecaster.py.")
