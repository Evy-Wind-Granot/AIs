#!/usr/bin/env python3
"""Certification wrapper for the production geomagnetic forecaster.

This layer keeps the existing causal training architecture while tightening two
production failure modes observed in certification:

* operating thresholds are selected against two chronological calibration
  sub-windows, reducing sensitivity to a single calibration regime;
* the longest (+6h) horizon uses monotonic isotonic probability calibration,
  which is better suited to the non-linear score/probability relationship at
  that horizon.

No final-test observations are used by either procedure.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import production_forecaster as _production

CERTIFIED_MODEL_VERSION = "2.1.1"
_production.MODEL_VERSION = CERTIFIED_MODEL_VERSION

# Keep a stable reference to the original selector.  fit() temporarily replaces
# the module-level symbol, so calling _production._choose_operating_threshold
# from inside our replacement would otherwise recurse forever.
_ORIGINAL_THRESHOLD_SELECTOR = _production._choose_operating_threshold


class _IsotonicCalibratedClassifier:
    """Pickle-safe causal isotonic calibration around a fitted classifier."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def _raw_score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(self.base, "decision_function"):
            score = np.asarray(self.base.decision_function(x), dtype=float)
            if score.ndim > 1:
                score = score[:, -1]
            return score
        probability = np.asarray(self.base.predict_proba(x)[:, 1], dtype=float)
        probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
        return np.log(probability / (1.0 - probability))

    def fit(self, x_cal: pd.DataFrame, y_cal: np.ndarray) -> "_IsotonicCalibratedClassifier":
        y = np.asarray(y_cal, dtype=int)
        if np.unique(y).size < 2:
            raise ValueError("Both storm classes are required for probability calibration.")
        self.calibrator.fit(self._raw_score(x_cal), y)
        return self

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        probability = np.asarray(self.calibrator.predict(self._raw_score(x)), dtype=float)
        probability = np.clip(probability, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def _fast_threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float | None]:
    """Compute only the classification quantities needed during threshold search.

    _production._binary_metrics also computes Brier score.  Recomputing that
    metric hundreds of times made certification unnecessarily slow.  Brier is
    still computed once for the selected threshold below by the production
    selector/reporting path.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    positives = tp + fn
    negatives = tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    far = fp / negatives if negatives else None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "false_alarm_rate": float(far) if far is not None else None,
    }


def _choose_robust_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    min_precision: float,
    max_far: float,
) -> Tuple[float, Dict[str, Any]]:
    """Select a threshold that is safe in both chronological calibration regimes."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if len(y) < 400:
        return _ORIGINAL_THRESHOLD_SELECTOR(
            y, p, min_precision=min_precision, max_far=max_far
        )

    split = int(len(y) * 0.60)
    split = min(max(split, 200), len(y) - 200)
    y_early, y_late = y[:split], y[split:]
    p_early, p_late = p[:split], p[split:]

    # A compact deterministic grid is sufficient here.  Quantile candidates
    # preserve resolution where the model actually places probability mass.
    candidates = np.unique(
        np.concatenate([
            np.linspace(0.05, 0.99, 95),
            np.quantile(p, np.linspace(0.01, 0.99, 49)),
        ])
    )

    feasible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        early = _fast_threshold_metrics(y_early, p_early, float(threshold))
        late = _fast_threshold_metrics(y_late, p_late, float(threshold))
        early_precision = float(early["precision"] or 0.0)
        late_precision = float(late["precision"] or 0.0)
        early_far = float(early["false_alarm_rate"] if early["false_alarm_rate"] is not None else 1.0)
        late_far = float(late["false_alarm_rate"] if late["false_alarm_rate"] is not None else 1.0)
        if (
            early_precision >= min_precision
            and late_precision >= min_precision
            and early_far <= max_far
            and late_far <= max_far
        ):
            early_recall = float(early["recall"] or 0.0)
            late_recall = float(late["recall"] or 0.0)
            avg_recall = 0.5 * (early_recall + late_recall)
            worst_far = max(early_far, late_far)
            feasible.append((min(early_recall, late_recall), avg_recall, -worst_far, float(threshold)))

    if feasible:
        feasible.sort(reverse=True)
        threshold = feasible[0][3]
        return threshold, {
            "method": "maximize_worst_window_recall_subject_to_precision_far",
            "min_precision": min_precision,
            "max_false_alarm_rate": max_far,
            "calibration_split": split,
            "feasible_candidates": len(feasible),
            "early_metrics": _production._binary_metrics(y_early, p_early, threshold),
            "late_metrics": _production._binary_metrics(y_late, p_late, threshold),
            "selected_metrics": _production._binary_metrics(y, p, threshold),
        }

    # Important: call the saved original selector, not the temporarily patched
    # module-level symbol.  This is both correct and prevents infinite recursion.
    threshold, fallback = _ORIGINAL_THRESHOLD_SELECTOR(
        y, p, min_precision=min_precision, max_far=max_far
    )
    fallback["method"] = "fallback_single_window_no_robust_constraint"
    fallback["calibration_split"] = split
    return threshold, fallback


class GeomagneticForecaster(_production.GeomagneticForecaster):
    """Production forecaster with stricter temporal certification behavior."""

    def _calibrated_classifier(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_cal: pd.DataFrame,
        y_cal: np.ndarray,
    ) -> Any:
        horizon_index = getattr(self, "_calibration_horizon_index", 0)
        self._calibration_horizon_index = horizon_index + 1
        base = self._classifier()
        base.fit(x_train, np.asarray(y_train, dtype=int))
        if horizon_index == len(self.config.horizons_hours) - 1:
            return _IsotonicCalibratedClassifier(base).fit(x_cal, np.asarray(y_cal, dtype=int))
        return _production._SigmoidCalibratedClassifier(base).fit(
            x_cal, np.asarray(y_cal, dtype=int)
        )

    def fit(self, frame: pd.DataFrame, *, cadence_s: float = 60.0) -> Dict[str, Any]:
        self._calibration_horizon_index = 0
        original_selector = _production._choose_operating_threshold
        _production._choose_operating_threshold = _choose_robust_threshold
        try:
            return super().fit(frame, cadence_s=cadence_s)
        finally:
            _production._choose_operating_threshold = original_selector
