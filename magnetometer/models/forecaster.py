#!/usr/bin/env python3
"""Production-oriented short-horizon geomagnetic forecasting model.

The model deliberately forecasts the future peak absolute QDC residual rather
than replacing the deterministic classifier. Each horizon has both a regressor
(magnitude) and a binary classifier (storm-threshold breach probability).

Backend choices:
* ``sklearn``: HistGradientBoostingRegressor/Classifier; zero extra ML dependency.
* ``lightgbm``: optional LightGBM backend for faster/larger training when installed.

The model is causal, low-latency at inference, independently serialized, and
stores its feature schema/configuration alongside the estimators.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from feature_engineering import build_supervised_dataset, make_forecast_features

try:  # Optional performance backend.
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover - depends on environment.
    lgb = None


MODEL_VERSION = "1.0.0"
DEFAULT_HORIZONS_HOURS = (1, 3, 6)
DEFAULT_STORM_THRESHOLD_NT = 35.0
DEFAULT_FEATURE_WINDOWS_MINUTES = (15, 60, 180, 360)
DEFAULT_LAGS_MINUTES = (1, 5, 15, 30, 60, 180)


def _clean_scalar(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float | None]:
    pred = np.asarray(probability) >= threshold
    truth = np.asarray(y_true).astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0
    )
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    far = float(fp / (fp + tn)) if fp + tn else None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": far,
        "threshold": float(threshold),
    }


@dataclass(frozen=True)
class ForecastConfig:
    backend: str = "sklearn"
    horizons_hours: Tuple[int, ...] = DEFAULT_HORIZONS_HOURS
    storm_threshold_nt: float = DEFAULT_STORM_THRESHOLD_NT
    probability_threshold: float = 0.50
    sequence_length: int = 60
    windows_minutes: Tuple[int, ...] = DEFAULT_FEATURE_WINDOWS_MINUTES
    lags_minutes: Tuple[int, ...] = DEFAULT_LAGS_MINUTES
    validation_fraction: float = 0.20
    random_state: int = 20260818
    max_iter: int = 400
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    min_samples_leaf: int = 30
    n_estimators: int = 500
    num_leaves: int = 31
    early_stopping_rounds: int = 50
    confidence_floor: float = 0.55


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
    """Multi-horizon residual-amplitude and storm-probability forecaster."""

    config: ForecastConfig = field(default_factory=ForecastConfig)
    feature_names: list[str] = field(default_factory=list)
    regressors: Dict[int, Any] = field(default_factory=dict)
    classifiers: Dict[int, Any] = field(default_factory=dict)
    fitted: bool = False
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    def _estimator_pair(self) -> Tuple[Any, Any]:
        backend = self.config.backend.lower()
        if backend == "lightgbm":
            if lgb is None:
                raise ImportError("LightGBM backend requested but lightgbm is not installed.")
            reg = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=self.config.l2_regularization,
                min_child_samples=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            clf = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=self.config.l2_regularization,
                min_child_samples=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            return reg, clf

        if backend != "sklearn":
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        reg = HistGradientBoostingRegressor(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            min_samples_leaf=self.config.min_samples_leaf,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=self.config.random_state,
        )
        clf = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization,
            min_samples_leaf=self.config.min_samples_leaf,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=self.config.random_state,
        )
        return reg, clf

    @staticmethod
    def _sanitize_features(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.select_dtypes(include=[np.number]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        # Causal forward fill; a limited backfill only fills the initial warmup
        # area where the model cannot otherwise infer a complete feature vector.
        numeric = numeric.ffill().bfill()
        return numeric

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        cadence_s: float = 60.0,
        validation_fraction: float | None = None,
    ) -> Dict[str, Any]:
        """Fit all horizon regressors/classifiers on chronologically ordered data."""
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

        if len(x) < 1000:
            raise ValueError("At least 1000 time-aligned samples are required for forecasting model training.")
        fraction = self.config.validation_fraction if validation_fraction is None else validation_fraction
        if not 0.05 <= fraction < 0.50:
            raise ValueError("validation_fraction must be in [0.05, 0.50)")
        split = int(len(x) * (1.0 - fraction))
        if split < 500 or len(x) - split < 100:
            raise ValueError("Training/validation split is too small for reliable forecasting.")

        x_train = x.iloc[:split]
        x_valid = x.iloc[split:]
        report: Dict[str, Any] = {"backend": self.config.backend, "horizons": {}}
        self.regressors.clear()
        self.classifiers.clear()

        for horizon in self.config.horizons_hours:
            peak_col = f"target_peak_abs_{horizon}h"
            storm_col = f"target_storm_{horizon}h"
            valid_train = targets[peak_col].iloc[:split].notna() & targets[storm_col].iloc[:split].notna()
            valid_test = targets[peak_col].iloc[split:].notna() & targets[storm_col].iloc[split:].notna()
            if valid_train.sum() < 200 or valid_test.sum() < 50:
                raise ValueError(f"Insufficient target coverage for {horizon}h horizon.")

            y_reg_train = targets[peak_col].iloc[:split][valid_train].to_numpy(dtype=float)
            y_reg_valid = targets[peak_col].iloc[split:][valid_test].to_numpy(dtype=float)
            y_cls_train = targets[storm_col].iloc[:split][valid_train].to_numpy(dtype=int)
            y_cls_valid = targets[storm_col].iloc[split:][valid_test].to_numpy(dtype=int)
            xtr = x_train.loc[valid_train]
            xva = x_valid.loc[valid_test]

            if np.unique(y_cls_train).size < 2:
                raise ValueError(f"Training target for {horizon}h has only one storm class.")

            reg, clf = self._estimator_pair()
            if self.config.backend.lower() == "lightgbm":
                callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]  # type: ignore[union-attr]
                reg.fit(xtr, y_reg_train, eval_set=[(xva, y_reg_valid)], callbacks=callbacks)
                callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]  # type: ignore[union-attr]
                clf.fit(xtr, y_cls_train, eval_set=[(xva, y_cls_valid)], callbacks=callbacks)
            else:
                reg.fit(xtr, y_reg_train)
                clf.fit(xtr, y_cls_train)

            reg_pred = np.clip(reg.predict(xva), 0.0, None)
            cls_prob = clf.predict_proba(xva)[:, 1]
            report["horizons"][str(horizon)] = {
                "validation_samples": int(len(xva)),
                "rmse_nt": float(np.sqrt(mean_squared_error(y_reg_valid, reg_pred))),
                "mae_nt": float(mean_absolute_error(y_reg_valid, reg_pred)),
                "storm_metrics": _binary_metrics(
                    y_cls_valid, cls_prob, self.config.probability_threshold
                ),
                "positive_rate": float(np.mean(y_cls_valid)),
            }
            self.regressors[int(horizon)] = reg
            self.classifiers[int(horizon)] = clf

        self.training_metadata = {
            "model_version": MODEL_VERSION,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "samples": int(len(x)),
            "train_samples": int(split),
            "validation_samples": int(len(x) - split),
            "cadence_seconds": float(cadence_s),
            "validation_fraction": float(fraction),
            "feature_count": len(self.feature_names),
            "training_end": x_train.index[-1].isoformat(),
            "validation_start": x_valid.index[0].isoformat(),
            "backend_available_lightgbm": lgb is not None,
            "validation_report": report,
        }
        self.fitted = True
        return report

    def _frame_to_matrix(self, frame: pd.DataFrame, cadence_s: float) -> pd.DataFrame:
        if not self.feature_names:
            features = make_forecast_features(
                frame,
                cadence_s=cadence_s,
                windows_minutes=self.config.windows_minutes,
                lags_minutes=self.config.lags_minutes,
            )
        else:
            features = make_forecast_features(
                frame,
                cadence_s=cadence_s,
                windows_minutes=self.config.windows_minutes,
                lags_minutes=self.config.lags_minutes,
            )
        for name in self.feature_names:
            if name not in features.columns:
                raise ValueError(f"Inference feature schema mismatch: missing '{name}'.")
        extra = [name for name in features.columns if name not in self.feature_names]
        # Extra engineered columns are harmless but intentionally ignored so the
        # serialized feature schema remains authoritative.
        del extra
        return self._sanitize_features(features.loc[:, self.feature_names])

    @staticmethod
    def _tier_from_amplitude(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "unknown"
        value = abs(float(value))
        if value >= 200.0:
            return "severe_storm"
        if value >= 100.0:
            return "major_storm"
        if value >= 35.0:
            return "minor_storm"
        if value >= 15.0:
            return "active"
        if value >= 10.0:
            return "unsettled"
        return "quiet"

    @staticmethod
    def _tier_score(tier: str) -> int:
        return {
            "unknown": 0,
            "quiet": 0,
            "unsettled": 1,
            "active": 2,
            "minor_storm": 3,
            "major_storm": 4,
            "severe_storm": 5,
        }.get(tier, 0)

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        cadence_s: float = 60.0,
        current_rule_tier: str | None = None,
    ) -> ForecastResult:
        """Produce +1h/+3h/+6h forecasts from the latest causal feature row."""
        if not self.fitted:
            raise RuntimeError("Forecaster is not fitted or loaded.")
        if frame.empty:
            raise ValueError("forecast frame cannot be empty")

        x = self._frame_to_matrix(frame, cadence_s)
        latest = x.iloc[[-1]]
        horizons: Dict[str, Dict[str, float | str | bool | None]] = {}
        current = _clean_scalar(frame["residual"].iloc[-1]) if "residual" in frame else None

        for horizon in self.config.horizons_hours:
            reg = self.regressors[int(horizon)]
            clf = self.classifiers[int(horizon)]
            amplitude = max(0.0, float(reg.predict(latest)[0]))
            probability = float(clf.predict_proba(latest)[0, 1])
            tier = self._tier_from_amplitude(amplitude)
            if probability >= self.config.probability_threshold and self._tier_score(tier) < 3:
                tier = "minor_storm"
            confidence = max(probability, 1.0 - probability)
            confidence = confidence if confidence >= self.config.confidence_floor else None
            divergence = False
            if current_rule_tier is not None:
                divergence = abs(self._tier_score(tier) - self._tier_score(current_rule_tier)) >= 2
            horizons[str(horizon)] = {
                "predicted_amplitude_nt": amplitude,
                "storm_probability": probability,
                "model_confidence": confidence,
                "forecast_tier": tier,
                "divergence": divergence,
            }

        max_forecast_score = max(self._tier_score(v["forecast_tier"]) for v in horizons.values()) if horizons else 0
        current_score = self._tier_score(current_rule_tier or "quiet")
        anomaly_delta = abs(max_forecast_score - current_score) / 5.0
        divergence = anomaly_delta >= 0.40
        return ForecastResult(
            generated_at=frame.index[-1].isoformat(),
            horizons=horizons,
            current_residual_nt=current,
            anomaly_delta=float(anomaly_delta),
            divergence=bool(divergence),
            model_version=MODEL_VERSION,
        )

    def save_model(self, path: str | Path) -> None:
        """Save estimator bundle plus a JSON manifest.

        Joblib is intentionally used only for trusted, locally-produced model
        artifacts; it is pickle-based and must never load untrusted files.
        """
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted forecaster.")
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "config": asdict(self.config),
            "feature_names": self.feature_names,
            "regressors": self.regressors,
            "classifiers": self.classifiers,
            "training_metadata": self.training_metadata,
            "model_version": MODEL_VERSION,
        }
        joblib.dump(bundle, base.with_suffix(".joblib"), compress=3, protocol=5)
        manifest = {
            "model_version": MODEL_VERSION,
            "artifact": base.with_suffix(".joblib").name,
            "config": asdict(self.config),
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata,
        }
        base.with_suffix(".json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load_model(cls, path: str | Path) -> "GeomagneticForecaster":
        """Load a trusted local model bundle and validate its schema."""
        base = Path(path)
        manifest_path = base.with_suffix(".json")
        artifact_path = base.with_suffix(".joblib")
        if not manifest_path.exists() or not artifact_path.exists():
            raise FileNotFoundError(f"Expected {manifest_path} and {artifact_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model_version") != MODEL_VERSION:
            raise ValueError(
                f"Unsupported model version {manifest.get('model_version')!r}; expected {MODEL_VERSION!r}."
            )
        bundle = joblib.load(artifact_path)
        config = ForecastConfig(**bundle["config"])
        model = cls(
            config=config,
            feature_names=list(bundle["feature_names"]),
            regressors=dict(bundle["regressors"]),
            classifiers=dict(bundle["classifiers"]),
            fitted=True,
            training_metadata=dict(bundle.get("training_metadata", {})),
        )
        if not model.feature_names or any(not isinstance(name, str) for name in model.feature_names):
            raise ValueError("Invalid serialized feature schema.")
        return model


def evaluate_forecast(
    model: GeomagneticForecaster,
    frame: pd.DataFrame,
    *,
    cadence_s: float = 60.0,
) -> Dict[str, Any]:
    """Evaluate chronological holdout-style forecasts against future targets."""
    features, targets = build_supervised_dataset(
        frame,
        cadence_s=cadence_s,
        windows_minutes=model.config.windows_minutes,
        lags_minutes=model.config.lags_minutes,
        horizons_hours=model.config.horizons_hours,
        storm_threshold_nt=model.config.storm_threshold_nt,
    )
    x = model._sanitize_features(features.loc[:, model.feature_names])
    report: Dict[str, Any] = {"horizons": {}}
    for horizon in model.config.horizons_hours:
        peak_col = f"target_peak_abs_{horizon}h"
        storm_col = f"target_storm_{horizon}h"
        mask = targets[peak_col].notna() & targets[storm_col].notna()
        if not mask.any():
            report["horizons"][str(horizon)] = None
            continue
        pred_peak = np.clip(model.regressors[int(horizon)].predict(x.loc[mask]), 0.0, None)
        pred_prob = model.classifiers[int(horizon)].predict_proba(x.loc[mask])[:, 1]
        y_peak = targets.loc[mask, peak_col].to_numpy(dtype=float)
        y_storm = targets.loc[mask, storm_col].to_numpy(dtype=int)
        persistence_peak = np.abs(frame.loc[x.index[mask], "residual"].to_numpy(dtype=float))
        report["horizons"][str(horizon)] = {
            "samples": int(mask.sum()),
            "rmse_nt": float(np.sqrt(mean_squared_error(y_peak, pred_peak))),
            "mae_nt": float(mean_absolute_error(y_peak, pred_peak)),
            "storm": _binary_metrics(y_storm, pred_prob, model.config.probability_threshold),
            "persistence_rmse_nt": float(np.sqrt(mean_squared_error(y_peak, persistence_peak))),
            "persistence_mae_nt": float(mean_absolute_error(y_peak, persistence_peak)),
        }
    return report


if __name__ == "__main__":
    print("GeomagneticForecaster module: import this module from the training/inference pipeline.")
