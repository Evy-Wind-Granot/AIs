#!/usr/bin/env python3
"""Certification wrapper for the production geomagnetic forecaster.

The certified detector uses richer causal multi-timescale event features and
bounded class weighting. Calibration and threshold selection remain strictly
chronological and final-test data remain untouched.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from . import production_forecaster as _production

CERTIFIED_MODEL_VERSION = "3.0.1"
_production.MODEL_VERSION = CERTIFIED_MODEL_VERSION
_BASE_CHOOSE_OPERATING_THRESHOLD = _production._choose_operating_threshold


def _metric_value(metrics: Dict[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name)
    return default if value is None else float(value)


def _candidate_thresholds(p: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([
        np.linspace(0.05, 0.99, 189),
        np.quantile(p, np.linspace(0.01, 0.99, 99)),
    ]))


def _choose_robust_threshold(y_true: np.ndarray, probability: np.ndarray, *, min_precision: float, max_far: float, target_far: float) -> Tuple[float, Dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if len(y) < 400:
        return _conservative_fallback(y, p, min_precision, max_far, target_far)
    split = min(max(int(len(y) * 0.60), 200), len(y) - 200)
    y_early, y_late, p_early, p_late = y[:split], y[split:], p[:split], p[split:]
    feasible = []
    for threshold in _candidate_thresholds(p):
        early = _production._binary_metrics(y_early, p_early, float(threshold))
        late = _production._binary_metrics(y_late, p_late, float(threshold))
        if (_metric_value(early, "precision") >= min_precision and
                _metric_value(late, "precision") >= min_precision and
                _metric_value(early, "false_alarm_rate", 1.0) <= target_far and
                _metric_value(late, "false_alarm_rate", 1.0) <= target_far):
            worst_recall = min(_metric_value(early, "recall"), _metric_value(late, "recall"))
            avg_recall = 0.5 * (_metric_value(early, "recall") + _metric_value(late, "recall"))
            worst_f1 = min(_metric_value(early, "f1"), _metric_value(late, "f1"))
            worst_far = max(_metric_value(early, "false_alarm_rate", 1.0), _metric_value(late, "false_alarm_rate", 1.0))
            feasible.append((worst_recall, avg_recall, worst_f1, -worst_far, float(threshold)))
    if feasible:
        feasible.sort(reverse=True)
        threshold = feasible[0][-1]
        return threshold, {
            "method": "two_window_maximize_worst_recall",
            "target_false_alarm_rate": target_far,
            "max_false_alarm_rate": max_far,
            "min_precision": min_precision,
            "calibration_split": split,
            "feasible_candidates": len(feasible),
            "early_metrics": _production._binary_metrics(y_early, p_early, threshold),
            "late_metrics": _production._binary_metrics(y_late, p_late, threshold),
            "selected_metrics": _production._binary_metrics(y, p, threshold),
        }
    return _conservative_fallback(y, p, min_precision, max_far, target_far, split=split)


def _conservative_fallback(y: np.ndarray, p: np.ndarray, min_precision: float, max_far: float, target_far: float, *, split: int | None = None) -> Tuple[float, Dict[str, Any]]:
    if len(y) == 0:
        raise ValueError("Cannot select a threshold from an empty calibration set")
    if split is None:
        split = min(max(int(len(y) * 0.60), 1), max(1, len(y) - 1))
    y_early, y_late, p_early, p_late = y[:split], y[split:], p[:split], p[split:]
    safe = []
    for threshold in _candidate_thresholds(p):
        early = _production._binary_metrics(y_early, p_early, float(threshold))
        late = _production._binary_metrics(y_late, p_late, float(threshold))
        precision = min(_metric_value(early, "precision"), _metric_value(late, "precision"))
        if precision < min_precision:
            continue
        worst_far = max(_metric_value(early, "false_alarm_rate", 1.0), _metric_value(late, "false_alarm_rate", 1.0))
        worst_recall = min(_metric_value(early, "recall"), _metric_value(late, "recall"))
        safe.append((worst_far, -worst_recall, float(threshold)))
    if safe:
        safe.sort()
        threshold = safe[0][-1]
        method = "precision_safe_lowest_worst_far"
    else:
        safest = []
        for threshold in _candidate_thresholds(p):
            metrics = _production._binary_metrics(y, p, float(threshold))
            safest.append((_metric_value(metrics, "false_alarm_rate", 1.0), -_metric_value(metrics, "recall"), float(threshold)))
        safest.sort()
        threshold = safest[0][-1]
        method = "lowest_far_no_precision_feasible"
    return threshold, {"method": method, "target_false_alarm_rate": target_far, "max_false_alarm_rate": max_far, "min_precision": min_precision, "calibration_split": split, "feasible_candidates": 0, "selected_metrics": _production._binary_metrics(y, p, threshold)}


class _IsotonicCalibratedClassifier:
    def __init__(self, base: Any) -> None:
        self.base = base
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def _raw_score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(self.base, "decision_function"):
            score = np.asarray(self.base.decision_function(x), dtype=float)
            return score[:, -1] if score.ndim > 1 else score
        probability = np.clip(np.asarray(self.base.predict_proba(x)[:, 1], dtype=float), 1e-7, 1.0 - 1e-7)
        return np.log(probability / (1.0 - probability))

    def fit(self, x_cal: pd.DataFrame, y_cal: np.ndarray) -> "_IsotonicCalibratedClassifier":
        y = np.asarray(y_cal, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("Both storm classes are required for probability calibration.")
        self.calibrator.fit(self._raw_score(x_cal), y)
        return self

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(self.calibrator.predict(self._raw_score(x)), dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class GeomagneticForecaster(_production.GeomagneticForecaster):
    """Certified forecaster with a stronger class-balanced storm detector."""

    def _classifier(self) -> Any:
        backend = self.config.backend.lower()
        if backend == "lightgbm":
            if _production.lgb is None:
                raise ImportError("LightGBM backend requested but lightgbm is not installed.")
            return _production.lgb.LGBMClassifier(
                objective="binary", n_estimators=max(self.config.n_estimators, 900),
                learning_rate=min(self.config.learning_rate, 0.04), num_leaves=max(self.config.num_leaves, 63),
                max_depth=-1, subsample=0.9, colsample_bytree=0.85,
                reg_alpha=0.05, reg_lambda=max(self.config.l2_regularization, 2.0),
                min_child_samples=max(self.config.min_samples_leaf, 50),
                random_state=self.config.random_state, n_jobs=-1, verbosity=-1,
            )
        return HistGradientBoostingClassifier(
            max_iter=max(self.config.max_iter, 800), learning_rate=min(self.config.learning_rate, 0.04),
            max_leaf_nodes=max(self.config.max_leaf_nodes, 63), l2_regularization=max(self.config.l2_regularization, 2.0),
            min_samples_leaf=max(self.config.min_samples_leaf, 50), early_stopping=True,
            validation_fraction=0.10, n_iter_no_change=60, random_state=self.config.random_state,
        )

    @staticmethod
    def _class_weights(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=int)
        positives = max(int(np.sum(y == 1)), 1)
        negatives = max(int(np.sum(y == 0)), 1)
        positive_weight = float(np.clip(np.sqrt(negatives / positives), 1.5, 4.0))
        weights = np.ones(len(y), dtype=float)
        weights[y == 1] = positive_weight
        return weights

    def _calibrated_classifier(self, x_train: pd.DataFrame, y_train: np.ndarray, x_cal: pd.DataFrame, y_cal: np.ndarray) -> Any:
        base = self._classifier()
        y = np.asarray(y_train, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("Both storm classes are required for classifier training.")
        base.fit(x_train, y, sample_weight=self._class_weights(y))
        index = getattr(self, "_calibration_horizon_index", 0)
        if index == len(self.config.horizons_hours) - 1:
            return _IsotonicCalibratedClassifier(base).fit(x_cal, np.asarray(y_cal, dtype=int))
        return _production._SigmoidCalibratedClassifier(base).fit(x_cal, np.asarray(y_cal, dtype=int))

    def fit(self, frame: pd.DataFrame, *, cadence_s: float = 60.0) -> Dict[str, Any]:
        # Force the detector to consume the richer causal feature schema while
        # retaining user-configured backend/loss/split settings.
        self.config = replace(
            self.config,
            windows_minutes=(5, 15, 30, 60, 180, 360, 720),
            lags_minutes=(1, 5, 15, 30, 60, 180, 360),
        )
        self._calibration_horizon_index = 0
        horizon_counter = {"index": 0}
        original_selector = _production._choose_operating_threshold
        original_calibration = self._calibrated_classifier

        def selector(y_true: np.ndarray, probability: np.ndarray, *, min_precision: float, max_far: float) -> Tuple[float, Dict[str, Any]]:
            index = horizon_counter["index"]
            horizon_counter["index"] += 1
            target_far = 0.005 if index == 0 else max_far
            return _choose_robust_threshold(y_true, probability, min_precision=min_precision, max_far=max_far, target_far=target_far)

        def calibrated_with_counter(*args: Any, **kwargs: Any) -> Any:
            result = original_calibration(*args, **kwargs)
            self._calibration_horizon_index += 1
            return result

        self._calibrated_classifier = calibrated_with_counter  # type: ignore[method-assign]
        _production._choose_operating_threshold = selector
        try:
            result = super().fit(frame, cadence_s=cadence_s)
            self.training_metadata["detector_version"] = CERTIFIED_MODEL_VERSION
            self.training_metadata["detector_features"] = {
                "windows_minutes": list(self.config.windows_minutes),
                "lags_minutes": list(self.config.lags_minutes),
                "thresholds_nt": [10, 15, 25, 35, 50, 75, 100],
                "class_weighting": "sqrt_inverse_frequency_capped_4x",
            }
            return result
        finally:
            _production._choose_operating_threshold = original_selector
            self._calibrated_classifier = original_calibration  # type: ignore[method-assign]
