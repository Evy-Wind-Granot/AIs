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
    max_iter: int = 250
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    missing_row_fraction_max: float = 0.50
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
        return {
            "horizon_hours": self.horizon_hours,
            "n_samples": self.n_samples,
            "rmse_nt": self.rmse_nt,
            "mae_nt": self.mae_nt,
            "storm_precision": self.precision,
            "storm_recall": self.recall,
            "storm_f1": self.f1,
        }


@dataclass
class _HorizonModel:
    regression: HistGradientBoostingRegressor
    classifier: Any


@dataclass
class GeomagneticForecaster:
    """Multi-horizon gradient boosting model with a persistence residual target.

    Instead of relearning the absolute amplitude from scratch, regression learns
    the correction to the causal persistence amplitude. This makes persistence
    an explicit safety baseline and materially reduces the risk of a model that
    is worse than simply carrying the current disturbance forward.
    """

    config: ForecastConfig = field(default_factory=ForecastConfig)
    feature_columns: tuple[str, ...] = ()
    models: dict[int, _HorizonModel] = field(default_factory=dict)
    fitted: bool = False
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def _regressor(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=0.5,
            random_state=self.config.random_state,
        )

    def _classifier(self, y: np.ndarray) -> Any:
        if np.unique(y).size < 2:
            return DummyClassifier(strategy="prior")
        return HistGradientBoostingClassifier(
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=0.5,
            random_state=self.config.random_state,
        )

    @staticmethod
    def _prepare_X(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        X = frame.copy()
        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing[:8]}")
            X = X.loc[:, list(columns)]
        return X.replace([np.inf, -np.inf], np.nan)

    def _baseline(self, X: pd.DataFrame) -> pd.Series:
        if "persistence_amplitude_nt" not in X.columns:
            raise ValueError("features must contain persistence_amplitude_nt")
        return X["persistence_amplitude_nt"].astype(float)

    def fit(
        self,
        features: pd.DataFrame,
        targets: Mapping[int, pd.Series],
        *,
        training_start: str | None = None,
        training_end: str | None = None,
    ) -> "GeomagneticForecaster":
        """Fit one delta regressor and storm classifier per horizon."""
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features must use a DatetimeIndex")
        if not features.index.is_monotonic_increasing:
            raise ValueError("features must be chronologically ordered")
        if not targets:
            raise ValueError("targets cannot be empty")

        self.feature_columns = tuple(str(c) for c in features.columns)
        X_all = self._prepare_X(features, self.feature_columns)
        baseline_all = self._baseline(X_all)
        self.models = {}
        sample_counts: dict[str, int] = {}

        for horizon in self.config.horizons_hours:
            if horizon not in targets:
                raise ValueError(f"missing target for horizon {horizon}h")
            y = pd.Series(targets[horizon], index=features.index, dtype=float)
            valid = y.notna() & baseline_all.notna() & X_all.notna().any(axis=1)
            X = X_all.loc[valid]
            y_values = y.loc[valid].to_numpy(dtype=float)
            baseline = baseline_all.loc[valid].to_numpy(dtype=float)
            if len(y_values) < self.config.confidence_min_samples:
                raise ValueError(f"not enough training samples for {horizon}h: {len(y_values)}")
            if np.unique(y_values).size < 2:
                raise ValueError(f"target has no variation for {horizon}h")

            delta = y_values - baseline
            reg = self._regressor()
            reg.fit(X, delta)
            storm = (y_values >= self.config.minor_storm_nt).astype(np.int8)
            clf = self._classifier(storm)
            clf.fit(X, storm)
            self.models[int(horizon)] = _HorizonModel(reg, clf)
            sample_counts[str(horizon)] = int(len(y_values))

        self.fitted = True
        self.training_metadata = {
            "schema_version": 2,
            "n_samples": int(len(features)),
            "feature_count": int(len(self.feature_columns)),
            "training_samples_by_horizon": sample_counts,
            "training_start": training_start or features.index.min().isoformat(),
            "training_end": training_end or features.index.max().isoformat(),
            "horizons_hours": list(self.config.horizons_hours),
            "target": "future peak-to-peak amplitude",
            "regression_target": "future amplitude minus causal persistence amplitude",
            "persistence_feature": "persistence_amplitude_nt",
        }
        return self

    def _predict_amplitude(self, bundle: _HorizonModel, X: pd.DataFrame) -> float:
        baseline = float(self._baseline(X).iloc[0])
        delta = float(bundle.regression.predict(X)[0])
        return float(max(0.0, baseline + delta))

    def predict(self, features: pd.DataFrame) -> dict[int, dict[str, Any]]:
        """Predict all configured horizons from the final causal feature row."""
        if not self.fitted or not self.models:
            raise RuntimeError("forecaster is not fitted")
        if features.empty:
            raise ValueError("features cannot be empty")
        X = self._prepare_X(features, self.feature_columns).tail(1)
        results: dict[int, dict[str, Any]] = {}
        quality = 1.0
        for column in ("residual_missing_15m", "residual_missing_3h"):
            if column in X:
                quality *= max(0.0, 1.0 - float(X[column].iloc[0]))
        for column in ("kp_available", "dst_available"):
            if column in X:
                quality *= 0.75 + 0.25 * float(X[column].iloc[0])

        for horizon in self.config.horizons_hours:
            bundle = self.models[horizon]
            amplitude = self._predict_amplitude(bundle, X)
            probabilities = bundle.classifier.predict_proba(X)[0]
            classes = list(getattr(bundle.classifier, "classes_", [0, 1]))
            storm_probability = float(probabilities[classes.index(1)] if 1 in classes else 0.0)
            tier = self.tier_from_amplitude(amplitude)
            probability_confidence = min(1.0, abs(storm_probability - 0.5) * 2.0)
            confidence = float(max(0.0, probability_confidence * quality))
            results[int(horizon)] = {
                "horizon_hours": int(horizon),
                "predicted_amplitude_nt": amplitude,
                "storm_probability": storm_probability,
                "predicted_tier": tier,
                "confidence": confidence,
                "data_quality": float(quality),
            }
        return results

    def evaluate(self, features: pd.DataFrame, targets: Mapping[int, pd.Series]) -> dict[int, ForecastEvaluation]:
        """Evaluate a fitted model on a chronological holdout."""
        X = self._prepare_X(features, self.feature_columns)
        baseline = self._baseline(X)
        evaluations: dict[int, ForecastEvaluation] = {}
        for horizon in self.config.horizons_hours:
            y = pd.Series(targets[horizon], index=features.index, dtype=float)
            valid = y.notna() & baseline.notna() & X.notna().any(axis=1)
            xv = X.loc[valid]
            yv = y.loc[valid].to_numpy(dtype=float)
            pred = np.maximum(0.0, baseline.loc[valid].to_numpy(dtype=float) + self.models[horizon].regression.predict(xv))
            storm_true = yv >= self.config.minor_storm_nt
            storm_pred = pred >= self.config.minor_storm_nt
            evaluations[horizon] = ForecastEvaluation(
                horizon_hours=horizon,
                n_samples=int(len(yv)),
                rmse_nt=float(np.sqrt(mean_squared_error(yv, pred))),
                mae_nt=float(mean_absolute_error(yv, pred)),
                precision=float(precision_score(storm_true, storm_pred, zero_division=0)),
                recall=float(recall_score(storm_true, storm_pred, zero_division=0)),
                f1=float(f1_score(storm_true, storm_pred, zero_division=0)),
            )
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


def save_model(model: GeomagneticForecaster, path: str | Path) -> Path:
    """Atomically serialize a fitted forecaster."""
    if not model.fitted:
        raise ValueError("cannot save an unfitted forecaster")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return destination


def load_model(path: str | Path) -> GeomagneticForecaster:
    """Load and validate a production-approved forecaster artifact."""
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, GeomagneticForecaster) or not model.fitted:
        raise ValueError("invalid or unfitted geomagnetic forecaster artifact")
    if model.training_metadata.get("schema_version", 1) < 2:
        raise ValueError("legacy forecast artifact requires retraining")
    return model


def build_training_data(
    residual: pd.Series,
    kp: pd.Series | None,
    dst: pd.Series | None,
    *,
    cadence_s: float = 60.0,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Build causal features and future targets."""
    feature_config = FeatureConfig(
        cadence_s=cadence_s,
        lookback_hours=config.lookback_hours,
        windows_min=config.feature_windows_min,
        amplitude_window_min=config.amplitude_window_min,
    )
    features = build_features(residual, kp, dst, config=feature_config)
    targets = build_targets(
        residual,
        cadence_s=cadence_s,
        horizons_hours=config.horizons_hours,
        amplitude_window_min=config.amplitude_window_min,
    )
    return features, targets


__all__ = ["ForecastConfig", "ForecastEvaluation", "GeomagneticForecaster", "build_training_data", "save_model", "load_model"]
