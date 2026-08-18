"""Short-horizon hybrid geomagnetic forecasting."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from magnetometer.features import FeatureConfig, build_features, build_targets


_LEVELS = ("quiet", "unsettled", "active", "minor_storm", "major_storm", "severe_storm")


@dataclass(frozen=True)
class ForecastConfig:
    """Operational settings for the forecaster."""

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

    def __post_init__(self) -> None:
        if not self.horizons_hours:
            raise ValueError("at least one forecast horizon is required")
        if any(h <= 0 for h in self.horizons_hours):
            raise ValueError("forecast horizons must be positive")
        if tuple(sorted(set(self.horizons_hours))) != self.horizons_hours:
            raise ValueError("horizons_hours must be sorted and unique")
        if not self.feature_windows_min:
            raise ValueError("feature_windows_min cannot be empty")
        if max(self.feature_windows_min) > self.lookback_hours * 60:
            raise ValueError("largest feature window cannot exceed lookback_hours")


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
    """Multi-horizon gradient-boosted forecaster."""

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
    def _prepare_X(
        frame: pd.DataFrame, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        X = frame.copy()
        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing[:8]}")
            X = X.loc[:, list(columns)]
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(
        self,
        features: pd.DataFrame,
        targets: Mapping[int, pd.Series],
        *,
        training_start: str | None = None,
        training_end: str | None = None,
    ) -> "GeomagneticForecaster":
        """Fit one regression and one storm classifier per horizon."""
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features must use a DatetimeIndex")
        if not features.index.is_monotonic_increasing:
            raise ValueError("features must be chronologically ordered")
        if not targets:
            raise ValueError("targets cannot be empty")

        self.feature_columns = tuple(str(c) for c in features.columns)
        X_all = self._prepare_X(features, self.feature_columns)
        self.models = {}

        for horizon in self.config.horizons_hours:
            if horizon not in targets:
                raise ValueError(f"missing target for horizon {horizon}h")
            y = pd.Series(targets[horizon], index=features.index, dtype=float)
            valid = y.notna() & X_all.notna().any(axis=1)
            X = X_all.loc[valid]
            y_values = y.loc[valid].to_numpy(dtype=float)
            if len(y_values) < 100:
                raise ValueError(
                    f"not enough training samples for {horizon}h: {len(y_values)}"
                )
            if np.unique(y_values).size < 2:
                raise ValueError(f"target has no variation for {horizon}h")

            reg = self._regressor()
            reg.fit(X, y_values)
            storm = (y_values >= self.config.minor_storm_nt).astype(np.int8)
            clf = self._classifier(storm)
            clf.fit(X, storm)
            self.models[int(horizon)] = _HorizonModel(reg, clf)

        self.fitted = True
        self.training_metadata = {
            "n_samples": int(len(features)),
            "feature_count": int(len(self.feature_columns)),
            "training_start": training_start or features.index.min().isoformat(),
            "training_end": training_end or features.index.max().isoformat(),
            "horizons_hours": list(self.config.horizons_hours),
            "target": "future amplitude strictly after forecast time",
        }
        return self

    def predict(self, features: pd.DataFrame) -> dict[int, dict[str, Any]]:
        """Predict all horizons from the final row of a feature frame."""
        if not self.fitted or not self.models:
            raise RuntimeError("forecaster is not fitted")
        if features.empty:
            raise ValueError("features cannot be empty")
        X = self._prepare_X(features, self.feature_columns).tail(1)
        results: dict[int, dict[str, Any]] = {}
        for horizon in self.config.horizons_hours:
            bundle = self.models[horizon]
            amplitude = float(max(0.0, bundle.regression.predict(X)[0]))
            probabilities = bundle.classifier.predict_proba(X)[0]
            classes = list(getattr(bundle.classifier, "classes_", [0, 1]))
            storm_probability = float(
                probabilities[classes.index(1)] if 1 in classes else 0.0
            )
            tier = self.tier_from_amplitude(amplitude)
            confidence = float(min(1.0, abs(storm_probability - 0.5) * 2.0))
            results[int(horizon)] = {
                "horizon_hours": int(horizon),
                "predicted_amplitude_nt": amplitude,
                "storm_probability": storm_probability,
                "predicted_tier": tier,
                "confidence": confidence,
            }
        return results

    def evaluate(
        self,
        features: pd.DataFrame,
        targets: Mapping[int, pd.Series],
    ) -> dict[int, ForecastEvaluation]:
        """Evaluate a fitted model on a chronological holdout."""
        X = self._prepare_X(features, self.feature_columns)
        evaluations: dict[int, ForecastEvaluation] = {}
        for horizon in self.config.horizons_hours:
            y = pd.Series(targets[horizon], index=features.index, dtype=float)
            valid = y.notna() & X.notna().any(axis=1)
            xv = X.loc[valid]
            yv = y.loc[valid].to_numpy(dtype=float)
            bundle = self.models[horizon]
            pred = np.maximum(0.0, bundle.regression.predict(xv))
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
        """Map predicted amplitude to the configured deterministic tier scale."""
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
    """Atomically serialize a fitted forecaster with pickle."""
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
    """Load and validate a serialized forecaster."""
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, GeomagneticForecaster) or not model.fitted:
        raise ValueError("invalid or unfitted geomagnetic forecaster artifact")
    return model


def build_training_data(
    residual: pd.Series,
    kp: pd.Series | None,
    dst: pd.Series | None,
    *,
    cadence_s: float = 60.0,
    config: ForecastConfig = ForecastConfig(),
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Build causal features and future targets for model training."""
    feature_config = FeatureConfig(
        cadence_s=cadence_s,
        lookback_hours=config.lookback_hours,
        windows_min=config.feature_windows_min,
    )
    features = build_features(residual, kp, dst, config=feature_config)
    targets = build_targets(
        residual,
        cadence_s=cadence_s,
        horizons_hours=config.horizons_hours,
        amplitude_window_min=config.amplitude_window_min,
    )
    return features, targets


__all__ = [
    "ForecastConfig",
    "ForecastEvaluation",
    "GeomagneticForecaster",
    "build_training_data",
    "save_model",
    "load_model",
]
