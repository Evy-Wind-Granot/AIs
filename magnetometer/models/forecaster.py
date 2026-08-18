"""Production short-horizon hybrid geomagnetic forecaster."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

from magnetometer.features import FeatureConfig, build_features, build_targets

_LEVELS = ("quiet", "unsettled", "active", "minor_storm", "major_storm", "severe_storm")


@dataclass(frozen=True)
class ForecastConfig:
    """Operational model settings."""
    horizons_hours: tuple[int, ...] = (1, 3, 6)
    lookback_hours: float = 12.0
    feature_windows_min: tuple[int, ...] = (15, 30, 60, 180, 360, 720)
    amplitude_window_min: int = 180
    minor_storm_nt: float = 100.0
    major_storm_nt: float = 400.0
    severe_storm_nt: float = 800.0
    unsettled_nt: float = 20.0
    active_nt: float = 30.0
    random_state: int = 42
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    regression_loss: str = "absolute_error"
    early_stopping: bool = False
    confidence_min_samples: int = 100

    def __post_init__(self) -> None:
        if not self.horizons_hours or tuple(sorted(set(self.horizons_hours))) != self.horizons_hours:
            raise ValueError("horizons_hours must be sorted, unique and non-empty")
        if any(h <= 0 for h in self.horizons_hours):
            raise ValueError("forecast horizons must be positive")
        if not self.feature_windows_min:
            raise ValueError("feature_windows_min cannot be empty")
        if max(self.feature_windows_min) > self.lookback_hours * 60:
            raise ValueError("largest feature window cannot exceed lookback_hours")
        if self.amplitude_window_min <= 0:
            raise ValueError("amplitude_window_min must be positive")
        # ``huber`` is retained as a backwards-compatible configuration alias.
        # HistGradientBoostingRegressor itself does not implement Huber loss;
        # the alias is mapped to its robust absolute-error objective below.
        if self.regression_loss not in {"squared_error", "absolute_error", "huber", "quantile"}:
            raise ValueError("unsupported regression_loss")
        if self.learning_rate <= 0 or self.max_iter <= 0 or self.min_samples_leaf <= 0:
            raise ValueError("invalid boosting hyperparameters")


@dataclass(frozen=True)
class ForecastEvaluation:
    """Evaluation metrics for one forecast horizon."""
    horizon_hours: int
    n_samples: int
    rmse_nt: float
    mae_nt: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return {"horizon_hours": self.horizon_hours, "n_samples": self.n_samples, "rmse_nt": self.rmse_nt, "mae_nt": self.mae_nt, "storm_precision": self.precision, "storm_recall": self.recall, "storm_f1": self.f1}


@dataclass
class _HorizonModel:
    regression: HistGradientBoostingRegressor
    classifier: Any


@dataclass
class GeomagneticForecaster:
    """Multi-horizon gradient boosting model with persistence safety blending."""
    config: ForecastConfig = field(default_factory=ForecastConfig)
    feature_columns: tuple[str, ...] = ()
    models: dict[int, _HorizonModel] = field(default_factory=dict)
    blend_weights: dict[int, float] = field(default_factory=dict)
    fitted: bool = False
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def _regressor(self) -> HistGradientBoostingRegressor:
        loss = "absolute_error" if self.config.regression_loss == "huber" else self.config.regression_loss
        kwargs: dict[str, Any] = {"loss": loss, "learning_rate": self.config.learning_rate, "max_iter": self.config.max_iter, "max_leaf_nodes": self.config.max_leaf_nodes, "min_samples_leaf": self.config.min_samples_leaf, "l2_regularization": self.config.l2_regularization, "early_stopping": self.config.early_stopping, "random_state": self.config.random_state}
        if loss == "quantile":
            kwargs["quantile"] = 0.5
        return HistGradientBoostingRegressor(**kwargs)

    def _classifier(self, y: np.ndarray) -> Any:
        if np.unique(y).size < 2:
            return DummyClassifier(strategy="prior")
        return HistGradientBoostingClassifier(learning_rate=self.config.learning_rate, max_iter=self.config.max_iter, max_leaf_nodes=self.config.max_leaf_nodes, min_samples_leaf=self.config.min_samples_leaf, l2_regularization=self.config.l2_regularization, early_stopping=self.config.early_stopping, random_state=self.config.random_state)

    @staticmethod
    def _prepare_X(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("feature frame cannot be empty")
        X = frame.copy()
        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing[:8]}")
            X = X.loc[:, list(columns)]
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise TypeError(f"non-numeric feature columns: {non_numeric[:8]}")
        return X.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _baseline(X: pd.DataFrame) -> pd.Series:
        if "persistence_amplitude_nt" not in X.columns:
            raise ValueError("features must contain persistence_amplitude_nt")
        return X["persistence_amplitude_nt"].astype(float)

    def _raw_predictions(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        bundle = self.models[horizon]
        baseline = self._baseline(X).to_numpy(dtype=float)
        return np.maximum(0.0, baseline + bundle.regression.predict(X))

    def fit(self, features: pd.DataFrame, targets: Mapping[int, pd.Series], *, training_start: str | None = None, training_end: str | None = None) -> "GeomagneticForecaster":
        """Fit one delta regressor and storm classifier per horizon."""
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features must use a DatetimeIndex")
        if not features.index.is_monotonic_increasing or features.index.has_duplicates:
            raise ValueError("features must be chronologically ordered and unique")
        if not targets:
            raise ValueError("targets cannot be empty")
        self.feature_columns = tuple(str(c) for c in features.columns)
        X_all = self._prepare_X(features, self.feature_columns)
        baseline_all = self._baseline(X_all)
        self.models = {}
        self.blend_weights = {int(h): 1.0 for h in self.config.horizons_hours}
        sample_counts: dict[str, int] = {}
        for horizon in self.config.horizons_hours:
            if horizon not in targets:
                raise ValueError(f"missing target for horizon {horizon}h")
            y = pd.Series(targets[horizon], index=features.index, dtype=float)
            valid = y.notna() & baseline_all.notna() & X_all.notna().any(axis=1)
            X, y_values = X_all.loc[valid], y.loc[valid].to_numpy(dtype=float)
            baseline = baseline_all.loc[valid].to_numpy(dtype=float)
            if len(y_values) < self.config.confidence_min_samples:
                raise ValueError(f"not enough training samples for {horizon}h: {len(y_values)}")
            if np.unique(y_values).size < 2:
                raise ValueError(f"target has no variation for {horizon}h")
            reg = self._regressor()
            reg.fit(X, y_values - baseline)
            storm = (y_values >= self.config.minor_storm_nt).astype(np.int8)
            clf = self._classifier(storm)
            clf.fit(X, storm)
            self.models[int(horizon)] = _HorizonModel(reg, clf)
            sample_counts[str(horizon)] = int(len(y_values))
        self.fitted = True
        self.training_metadata = {"schema_version": 3, "n_samples": int(len(features)), "feature_count": int(len(self.feature_columns)), "training_samples_by_horizon": sample_counts, "training_start": training_start or features.index.min().isoformat(), "training_end": training_end or features.index.max().isoformat(), "horizons_hours": list(self.config.horizons_hours), "target": "future peak-to-peak amplitude", "regression_target": "future amplitude minus causal persistence amplitude", "persistence_feature": "persistence_amplitude_nt", "regression_loss": self.config.regression_loss, "effective_regression_loss": "absolute_error" if self.config.regression_loss == "huber" else self.config.regression_loss, "early_stopping": self.config.early_stopping}
        return self

    def calibrate_blend(self, features: pd.DataFrame, targets: Mapping[int, pd.Series]) -> dict[int, float]:
        """Select a validation-only ML/persistence blend weight by MAE."""
        if not self.fitted:
            raise RuntimeError("forecaster is not fitted")
        X = self._prepare_X(features, self.feature_columns)
        baseline = self._baseline(X).to_numpy(dtype=float)
        for horizon in self.config.horizons_hours:
            y = pd.Series(targets[horizon], index=features.index, dtype=float).to_numpy(dtype=float)
            raw = self._raw_predictions(X, horizon)
            valid = np.isfinite(y) & np.isfinite(baseline) & np.isfinite(raw)
            if valid.sum() < self.config.confidence_min_samples:
                raise ValueError(f"not enough validation samples for +{horizon}h")
            best_weight, best_mae = 1.0, float("inf")
            for weight in np.linspace(0.0, 1.0, 21):
                pred = baseline[valid] + weight * (raw[valid] - baseline[valid])
                mae = float(mean_absolute_error(y[valid], pred))
                if mae < best_mae:
                    best_mae, best_weight = mae, float(weight)
            self.blend_weights[int(horizon)] = best_weight
        self.training_metadata["blend_weights"] = {str(k): float(v) for k, v in self.blend_weights.items()}
        return dict(self.blend_weights)

    def _blended_predictions(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        baseline = self._baseline(X).to_numpy(dtype=float)
        raw = self._raw_predictions(X, horizon)
        weight = float(self.blend_weights.get(horizon, 1.0))
        return np.maximum(0.0, baseline + weight * (raw - baseline))

    def predict(self, features: pd.DataFrame) -> dict[int, dict[str, Any]]:
        """Predict all horizons from the final causal feature row."""
        if not self.fitted or not self.models:
            raise RuntimeError("forecaster is not fitted")
        X = self._prepare_X(features, self.feature_columns).tail(1)
        baseline = float(self._baseline(X).iloc[0])
        if not np.isfinite(baseline):
            raise ValueError("latest causal persistence amplitude is unavailable")
        quality = 1.0
        for column in ("residual_missing_15m", "residual_missing_3h"):
            if column in X:
                quality *= max(0.0, 1.0 - float(X[column].iloc[0]))
        for column in ("kp_available", "dst_available"):
            if column in X:
                quality *= 0.75 + 0.25 * float(X[column].iloc[0])
        quality = float(max(0.0, min(1.0, quality)))
        results: dict[int, dict[str, Any]] = {}
        for horizon in self.config.horizons_hours:
            bundle = self.models[horizon]
            amplitude = float(self._blended_predictions(X, horizon)[0])
            raw = float(self._raw_predictions(X, horizon)[0])
            weight = float(self.blend_weights.get(horizon, 1.0))
            probabilities = bundle.classifier.predict_proba(X)[0]
            classes = list(getattr(bundle.classifier, "classes_", [0, 1]))
            storm_probability = float(probabilities[classes.index(1)] if 1 in classes else 0.0)
            tier = self.tier_from_amplitude(amplitude)
            classifier_confidence = abs(storm_probability - 0.5) * 2.0
            model_delta = amplitude - baseline
            blend_disagreement = abs(raw - baseline) / max(1.0, baseline)
            confidence = classifier_confidence * quality * (1.0 - min(0.5, blend_disagreement) * 0.5)
            results[int(horizon)] = {"horizon_hours": int(horizon), "predicted_amplitude_nt": amplitude, "persistence_amplitude_nt": baseline, "model_delta_nt": float(model_delta), "storm_probability": storm_probability, "predicted_tier": tier, "confidence": float(max(0.0, min(1.0, confidence))), "data_quality": quality, "blend_weight": weight, "model_persistence_disagreement": float(blend_disagreement)}
        return results

    def evaluate(self, features: pd.DataFrame, targets: Mapping[int, pd.Series]) -> dict[int, ForecastEvaluation]:
        """Evaluate a fitted model on a chronological holdout."""
        X = self._prepare_X(features, self.feature_columns)
        baseline = self._baseline(X)
        evaluations: dict[int, ForecastEvaluation] = {}
        for horizon in self.config.horizons_hours:
            y = pd.Series(targets[horizon], index=features.index, dtype=float)
            valid = y.notna() & baseline.notna() & X.notna().any(axis=1)
            xv, yv = X.loc[valid], y.loc[valid].to_numpy(dtype=float)
            pred = self._blended_predictions(xv, horizon)
            storm_true, storm_pred = yv >= self.config.minor_storm_nt, pred >= self.config.minor_storm_nt
            evaluations[horizon] = ForecastEvaluation(horizon, int(len(yv)), float(np.sqrt(mean_squared_error(yv, pred))), float(mean_absolute_error(yv, pred)), float(precision_score(storm_true, storm_pred, zero_division=0)), float(recall_score(storm_true, storm_pred, zero_division=0)), float(f1_score(storm_true, storm_pred, zero_division=0)))
        return evaluations

    def tier_from_amplitude(self, amplitude_nt: float) -> str:
        if amplitude_nt >= self.config.severe_storm_nt:
            return "severe_storm"
        if amplitude_nt >= self.config.major_storm_nt:
            return "major_storm"
        if amplitude_nt >= self.config.minor_storm_nt:
            return "minor_storm"
        if amplitude_nt >= self.config.active_nt:
            return "active"
        if amplitude_nt >= self.config.unsettled_nt:
            return "unsettled"
        return "quiet"

    def health_check(self) -> dict[str, Any]:
        """Return deployment-facing artifact health information."""
        return {"fitted": bool(self.fitted), "schema_version": int(self.training_metadata.get("schema_version", 0)), "horizons_hours": list(self.config.horizons_hours), "feature_count": len(self.feature_columns), "production_gate": self.training_metadata.get("production_gate", "unknown"), "model_type": self.training_metadata.get("model_type", "unknown")}


def save_model(model: GeomagneticForecaster, path: str | Path) -> Path:
    """Atomically serialize a fitted production candidate."""
    if not model.fitted:
        raise ValueError("cannot save an unfitted forecaster")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return destination


def load_model(path: str | Path, *, require_production: bool = True) -> GeomagneticForecaster:
    """Load a model, requiring production approval by default."""
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, GeomagneticForecaster) or not model.fitted:
        raise ValueError("invalid or unfitted geomagnetic forecaster artifact")
    if model.training_metadata.get("schema_version", 0) < 3:
        raise ValueError("legacy forecast artifact requires retraining")
    if require_production and model.training_metadata.get("production_gate") != "passed":
        raise ValueError("forecast artifact is not production-approved")
    return model


def build_training_data(residual: pd.Series, kp: pd.Series | None, dst: pd.Series | None, *, cadence_s: float = 60.0, config: ForecastConfig = ForecastConfig()) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Build causal features and strictly-future targets."""
    feature_config = FeatureConfig(cadence_s=cadence_s, lookback_hours=config.lookback_hours, windows_min=config.feature_windows_min, amplitude_window_min=config.amplitude_window_min)
    features = build_features(residual, kp, dst, config=feature_config)
    targets = build_targets(residual, cadence_s=cadence_s, horizons_hours=config.horizons_hours, amplitude_window_min=config.amplitude_window_min)
    return features, targets


__all__ = ["ForecastConfig", "ForecastEvaluation", "GeomagneticForecaster", "build_training_data", "save_model", "load_model"]
