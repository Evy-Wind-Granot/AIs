#!/usr/bin/env python3
"""Production short-horizon geomagnetic forecaster.

The deterministic QDC/Harmonic layer remains authoritative for current state.
This module forecasts future peak absolute QDC residuals and storm-threshold
breaches at +1h, +3h, and +6h using causal engineered features.

Production safeguards:
- strict chronological train / calibration / final-test split;
- time-aware probability calibration (no random CV);
- operating-threshold selection on calibration data only;
- robust absolute-error regression by default;
- persistence benchmark;
- probability calibration diagnostics (Brier score / ECE);
- deterministic feature schema and versioned serialization;
- no backward filling of features (avoids future leakage);
- low-latency single-row inference.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import build_supervised_dataset, make_forecast_features

try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover
    lgb = None

MODEL_VERSION = "2.0.0"
DEFAULT_HORIZONS_HOURS = (1, 3, 6)
DEFAULT_STORM_THRESHOLD_NT = 35.0
DEFAULT_FEATURE_WINDOWS_MINUTES = (15, 60, 180, 360)
DEFAULT_LAGS_MINUTES = (1, 5, 15, 30, 60, 180)


def _ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if np.any(mask):
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float | None]:
    truth = np.asarray(y_true, dtype=bool)
    prob = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    pred = prob >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    far = float(fp / (fp + tn)) if fp + tn else None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": far,
        "threshold": float(threshold),
        "brier_score": float(brier_score_loss(truth, prob)) if len(truth) else None,
        "ece": _ece(truth.astype(int), prob),
        "log_loss": float(log_loss(truth, np.column_stack([1.0 - prob, prob]), labels=[0, 1])) if len(np.unique(truth)) == 2 else None,
        "positive_rate": float(np.mean(truth)) if len(truth) else None,
    }


def _choose_operating_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    min_precision: float,
    max_far: float,
) -> Tuple[float, Dict[str, Any]]:
    """Choose the highest-recall safe threshold using calibration data only."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), np.quantile(p, np.linspace(0.01, 0.99, 99))]))
    feasible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        metrics = _binary_metrics(y, p, float(threshold))
        precision = float(metrics["precision"] or 0.0)
        recall = float(metrics["recall"] or 0.0)
        f1 = float(metrics["f1"] or 0.0)
        far = float(metrics["false_alarm_rate"] if metrics["false_alarm_rate"] is not None else 1.0)
        if precision >= min_precision and far <= max_far:
            feasible.append((recall, f1, -far, float(threshold)))
    if feasible:
        feasible.sort(reverse=True)
        threshold = feasible[0][3]
        return threshold, {
            "method": "maximize_recall_subject_to_precision_far",
            "min_precision": min_precision,
            "max_false_alarm_rate": max_far,
            "feasible_candidates": len(feasible),
            "selected_metrics": _binary_metrics(y, p, threshold),
        }

    fallback: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        metrics = _binary_metrics(y, p, float(threshold))
        fallback.append((float(metrics["f1"] or 0.0), float(metrics["precision"] or 0.0), -float(metrics["false_alarm_rate"] if metrics["false_alarm_rate"] is not None else 1.0), float(threshold)))
    fallback.sort(reverse=True)
    threshold = fallback[0][3]
    return threshold, {
        "method": "fallback_maximize_f1_no_feasible_constraint",
        "min_precision": min_precision,
        "max_false_alarm_rate": max_far,
        "feasible_candidates": 0,
        "selected_metrics": _binary_metrics(y, p, threshold),
    }


@dataclass(frozen=True)
class ForecastConfig:
    backend: str = "sklearn"
    horizons_hours: Tuple[int, ...] = DEFAULT_HORIZONS_HOURS
    storm_threshold_nt: float = DEFAULT_STORM_THRESHOLD_NT
    probability_threshold: float = 0.50
    threshold_min_precision: float = 0.80
    threshold_max_false_alarm_rate: float = 0.01
    sequence_length: int = 60
    windows_minutes: Tuple[int, ...] = DEFAULT_FEATURE_WINDOWS_MINUTES
    lags_minutes: Tuple[int, ...] = DEFAULT_LAGS_MINUTES
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 20260818
    regression_loss: str = "absolute_error"
    max_iter: int = 500
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    min_samples_leaf: int = 30
    n_estimators: int = 700
    num_leaves: int = 31
    early_stopping_rounds: int = 60
    confidence_floor: float = 0.55
    calibration_splits: int = 4
    calibration_gap_hours: int = 6


@dataclass
class ForecastResult:
    generated_at: str
    horizons: Dict[str, Dict[str, float | str | bool | None]]
    current_residual_nt: float | None
    anomaly_delta: float
    divergence: bool
    model_version: str


@dataclass
class GeomagneticForecaster:
    """Certified causal multi-horizon residual/storm forecaster."""

    config: ForecastConfig = field(default_factory=ForecastConfig)
    feature_names: list[str] = field(default_factory=list)
    regressors: Dict[int, Any] = field(default_factory=dict)
    classifiers: Dict[int, Any] = field(default_factory=dict)
    thresholds: Dict[int, float] = field(default_factory=dict)
    fitted: bool = False
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    def _regressor(self, loss: str) -> Any:
        backend = self.config.backend.lower()
        if backend == "lightgbm":
            if lgb is None:
                raise ImportError("LightGBM backend requested but lightgbm is not installed.")
            objective = "regression_l1" if loss == "absolute_error" else "regression"
            return lgb.LGBMRegressor(
                objective=objective,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=self.config.l2_regularization,
                min_child_samples=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        if backend != "sklearn":
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        return HistGradientBoostingRegressor(
            loss="absolute_error" if loss == "absolute_error" else "squared_error",
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            min_samples_leaf=self.config.min_samples_leaf,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=self.config.random_state,
        )

    def _classifier(self) -> Any:
        backend = self.config.backend.lower()
        if backend == "lightgbm":
            if lgb is None:
                raise ImportError("LightGBM backend requested but lightgbm is not installed.")
            return lgb.LGBMClassifier(
                objective="binary",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=self.config.l2_regularization,
                min_child_samples=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        if backend != "sklearn":
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        return HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            min_samples_leaf=self.config.min_samples_leaf,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=self.config.random_state,
        )

    @staticmethod
    def _sanitize_features(frame: pd.DataFrame) -> pd.DataFrame:
        # Never backfill: it would inject future observations into early rows.
        return frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).ffill()

    def _calibrated_classifier(self, x_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        base = self._classifier()
        gap = max(1, int(round(self.config.calibration_gap_hours * 3600.0 / 60.0)))
        n_splits = min(self.config.calibration_splits, max(2, len(x_train) // 500))
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        try:
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=splitter, ensemble=False).fit(x_train, y_train)
        except TypeError:  # sklearn < 1.2 compatibility
            return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=splitter, ensemble=False).fit(x_train, y_train)

    def fit(self, frame: pd.DataFrame, *, cadence_s: float = 60.0) -> Dict[str, Any]:
        """Fit using strict chronological train/calibration/final-test partitions."""
        features, targets = build_supervised_dataset(
            frame,
            cadence_s=cadence_s,
            windows_minutes=self.config.windows_minutes,
            lags_minutes=self.config.lags_minutes,
            horizons_hours=self.config.horizons_hours,
            storm_threshold_nt=self.config.storm_threshold_nt,
        )
        x = self._sanitize_features(features)
        self.feature_names = list(x.columns)
        if len(x) < 2000:
            raise ValueError("At least 2000 time-aligned samples are required.")

        cal_frac = self.config.validation_fraction
        test_frac = self.config.test_fraction
        if not (0.05 <= cal_frac < 0.40 and 0.05 <= test_frac < 0.40) or cal_frac + test_frac >= 0.50:
            raise ValueError("Calibration/test fractions must each be [0.05, 0.40) and sum below 0.50.")
        train_end = int(len(x) * (1.0 - cal_frac - test_frac))
        cal_end = int(len(x) * (1.0 - test_frac))
        if train_end < 1000 or cal_end - train_end < 300 or len(x) - cal_end < 300:
            raise ValueError("Chronological split is too small for reliable certification.")

        x_train = x.iloc[:train_end]
        x_cal = x.iloc[train_end:cal_end]
        x_test = x.iloc[cal_end:]
        self.regressors.clear(); self.classifiers.clear(); self.thresholds.clear()
        calibration_report: Dict[str, Any] = {}
        final_report: Dict[str, Any] = {}

        for horizon in self.config.horizons_hours:
            peak_col = f"target_peak_abs_{horizon}h"
            storm_col = f"target_storm_{horizon}h"
            train_mask = targets[peak_col].iloc[:train_end].notna() & targets[storm_col].iloc[:train_end].notna()
            cal_mask = targets[peak_col].iloc[train_end:cal_end].notna() & targets[storm_col].iloc[train_end:cal_end].notna()
            test_mask = targets[peak_col].iloc[cal_end:].notna() & targets[storm_col].iloc[cal_end:].notna()
            if train_mask.sum() < 500 or cal_mask.sum() < 200 or test_mask.sum() < 200:
                raise ValueError(f"Insufficient target coverage for {horizon}h.")

            xtr = x_train.loc[train_mask]; xcal = x_cal.loc[cal_mask]; xtest = x_test.loc[test_mask]
            yreg_tr = targets[peak_col].iloc[:train_end][train_mask].to_numpy(float)
            yreg_cal = targets[peak_col].iloc[train_end:cal_end][cal_mask].to_numpy(float)
            yreg_test = targets[peak_col].iloc[cal_end:][test_mask].to_numpy(float)
            ycls_tr = targets[storm_col].iloc[:train_end][train_mask].to_numpy(int)
            ycls_cal = targets[storm_col].iloc[train_end:cal_end][cal_mask].to_numpy(int)
            ycls_test = targets[storm_col].iloc[cal_end:][test_mask].to_numpy(int)
            if np.unique(ycls_tr).size < 2 or np.unique(ycls_cal).size < 2:
                raise ValueError(f"Both storm classes are required for {horizon}h.")

            reg = self._regressor(self.config.regression_loss)
            reg.fit(xtr, yreg_tr)
            cal_reg_pred = np.clip(reg.predict(xcal), 0.0, None)

            clf = self._calibrated_classifier(xtr, ycls_tr)
            cal_prob = clf.predict_proba(xcal)[:, 1]
            threshold, threshold_info = _choose_operating_threshold(
                ycls_cal, cal_prob,
                min_precision=self.config.threshold_min_precision,
                max_far=self.config.threshold_max_false_alarm_rate,
            )

            test_reg_pred = np.clip(reg.predict(xtest), 0.0, None)
            test_prob = np.clip(clf.predict_proba(xtest)[:, 1], 0.0, 1.0)
            current_abs = np.abs(features.iloc[cal_end:]["residual"].loc[test_mask].to_numpy(float))
            test_rmse = float(np.sqrt(mean_squared_error(yreg_test, test_reg_pred)))
            test_mae = float(mean_absolute_error(yreg_test, test_reg_pred))
            persistence_mae = float(mean_absolute_error(yreg_test, current_abs))
            persistence_rmse = float(np.sqrt(mean_squared_error(yreg_test, current_abs)))

            self.regressors[int(horizon)] = reg
            self.classifiers[int(horizon)] = clf
            self.thresholds[int(horizon)] = float(threshold)
            calibration_report[str(horizon)] = {
                "samples": int(cal_mask.sum()),
                "regression_mae_nt": float(mean_absolute_error(yreg_cal, cal_reg_pred)),
                "storm": _binary_metrics(ycls_cal, cal_prob, threshold),
                "threshold_selection": threshold_info,
            }
            final_report[str(horizon)] = {
                "samples": int(test_mask.sum()),
                "test_start": x_test.index[test_mask.argmax()].isoformat() if test_mask.any() else None,
                "regression_rmse_nt": test_rmse,
                "regression_mae_nt": test_mae,
                "storm": _binary_metrics(ycls_test, test_prob, threshold),
                "persistence_rmse_nt": persistence_rmse,
                "persistence_mae_nt": persistence_mae,
                "beats_persistence_rmse": bool(test_rmse < persistence_rmse),
                "beats_persistence_mae": bool(test_mae < persistence_mae),
            }

        self.training_metadata = {
            "model_version": MODEL_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "samples": int(len(x)),
            "train_samples": int(train_end),
            "calibration_samples": int(cal_end - train_end),
            "final_test_samples": int(len(x) - cal_end),
            "cadence_seconds": float(cadence_s),
            "train_end": x_train.index[-1].isoformat(),
            "calibration_start": x_cal.index[0].isoformat(),
            "calibration_end": x_cal.index[-1].isoformat(),
            "final_test_start": x_test.index[0].isoformat(),
            "final_test_end": x_test.index[-1].isoformat(),
            "feature_count": len(self.feature_names),
            "backend": self.config.backend,
            "regression_loss": self.config.regression_loss,
            "backend_available_lightgbm": lgb is not None,
            "calibration_report": calibration_report,
            "final_test_report": final_report,
        }
        self.fitted = True
        return {"backend": self.config.backend, "regression_loss": self.config.regression_loss, "calibration": calibration_report, "final_test": final_report}

    def _frame_to_matrix(self, frame: pd.DataFrame, cadence_s: float) -> pd.DataFrame:
        features = make_forecast_features(frame, cadence_s=cadence_s, windows_minutes=self.config.windows_minutes, lags_minutes=self.config.lags_minutes)
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"Inference feature schema mismatch: missing {missing}")
        return self._sanitize_features(features.loc[:, self.feature_names])

    @staticmethod
    def _tier_from_amplitude(value: float | None) -> str:
        if value is None or not np.isfinite(value): return "unknown"
        value = abs(float(value))
        if value >= 200.0: return "severe_storm"
        if value >= 100.0: return "major_storm"
        if value >= 35.0: return "minor_storm"
        if value >= 15.0: return "active"
        if value >= 10.0: return "unsettled"
        return "quiet"

    @staticmethod
    def _tier_score(tier: str) -> int:
        return {"unknown": 0, "quiet": 0, "unsettled": 1, "active": 2, "minor_storm": 3, "major_storm": 4, "severe_storm": 5}.get(tier, 0)

    def predict(self, frame: pd.DataFrame, *, cadence_s: float = 60.0, current_rule_tier: str | None = None) -> ForecastResult:
        if not self.fitted: raise RuntimeError("Forecaster is not fitted or loaded.")
        if frame.empty: raise ValueError("forecast frame cannot be empty")
        x = self._frame_to_matrix(frame, cadence_s)
        latest = x.iloc[[-1]]
        current = float(frame["residual"].iloc[-1]) if "residual" in frame and np.isfinite(frame["residual"].iloc[-1]) else None
        horizons: Dict[str, Dict[str, float | str | bool | None]] = {}
        for horizon in self.config.horizons_hours:
            amplitude = max(0.0, float(self.regressors[int(horizon)].predict(latest)[0]))
            probability = float(np.clip(self.classifiers[int(horizon)].predict_proba(latest)[0, 1], 0.0, 1.0))
            threshold = float(self.thresholds.get(int(horizon), self.config.probability_threshold))
            tier = self._tier_from_amplitude(amplitude)
            if probability >= threshold and self._tier_score(tier) < 3: tier = "minor_storm"
            confidence = max(probability, 1.0 - probability)
            horizons[str(horizon)] = {
                "predicted_amplitude_nt": amplitude,
                "storm_probability": probability,
                "storm_probability_threshold": threshold,
                "model_confidence": confidence if confidence >= self.config.confidence_floor else None,
                "forecast_tier": tier,
                "divergence": current_rule_tier is not None and abs(self._tier_score(tier) - self._tier_score(current_rule_tier)) >= 2,
            }
        current_score = self._tier_score(current_rule_tier or "quiet")
        max_score = max((self._tier_score(v["forecast_tier"]) for v in horizons.values()), default=0)
        anomaly_delta = abs(max_score - current_score) / 5.0
        return ForecastResult(generated_at=frame.index[-1].isoformat(), horizons=horizons, current_residual_nt=current, anomaly_delta=float(anomaly_delta), divergence=bool(anomaly_delta >= 0.40), model_version=MODEL_VERSION)

    def save_model(self, path: str | Path) -> None:
        if not self.fitted: raise RuntimeError("Cannot serialize an unfitted forecaster.")
        base = Path(path); base.parent.mkdir(parents=True, exist_ok=True)
        bundle = {"config": asdict(self.config), "feature_names": self.feature_names, "regressors": self.regressors, "classifiers": self.classifiers, "thresholds": self.thresholds, "training_metadata": self.training_metadata, "model_version": MODEL_VERSION}
        joblib.dump(bundle, base.with_suffix(".joblib"), compress=3, protocol=5)
        base.with_suffix(".json").write_text(json.dumps({"model_version": MODEL_VERSION, "artifact": base.with_suffix(".joblib").name, "config": asdict(self.config), "feature_names": self.feature_names, "thresholds": self.thresholds, "training_metadata": self.training_metadata}, indent=2))

    @classmethod
    def load_model(cls, path: str | Path) -> "GeomagneticForecaster":
        base = Path(path); manifest_path = base.with_suffix(".json"); artifact_path = base.with_suffix(".joblib")
        if not manifest_path.exists() or not artifact_path.exists(): raise FileNotFoundError(f"Expected {manifest_path} and {artifact_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model_version") != MODEL_VERSION: raise ValueError(f"Unsupported model version {manifest.get('model_version')!r}; expected {MODEL_VERSION!r}")
        bundle = joblib.load(artifact_path)
        model = cls(config=ForecastConfig(**bundle["config"]), feature_names=list(bundle["feature_names"]), regressors=dict(bundle["regressors"]), classifiers=dict(bundle["classifiers"]), thresholds={int(k): float(v) for k, v in bundle.get("thresholds", {}).items()}, fitted=True, training_metadata=dict(bundle.get("training_metadata", {})))
        if not model.feature_names or set(model.thresholds) != set(model.config.horizons_hours): raise ValueError("Serialized model feature/threshold schema is invalid.")
        return model


def evaluate_forecast(model: GeomagneticForecaster, frame: pd.DataFrame, *, cadence_s: float = 60.0) -> Dict[str, Any]:
    """Return the immutable final-test report recorded during fitting."""
    report = model.training_metadata.get("final_test_report")
    if report is None: raise ValueError("Model has no recorded final-test report.")
    return report


if __name__ == "__main__":
    print("Production GeomagneticForecaster module; use train_magnetometer_forecaster.py for training.")
