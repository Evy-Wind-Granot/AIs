import numpy as np
import pandas as pd

from magnetometer.features import build_features, build_targets
from models.forecaster import ForecastConfig, GeomagneticForecaster


def _dataset(periods=1400):
    index = pd.date_range("2024-01-01", periods=periods, freq="min", tz="UTC")
    x = np.arange(periods, dtype=float)
    residual = pd.Series(10.0 * np.sin(x / 25.0) + 2.0 * np.sin(x / 7.0), index=index)
    config = ForecastConfig(horizons_hours=(1,), feature_windows_min=(15, 60, 180), amplitude_window_min=15)
    features = build_features(residual, config=config and __import__('magnetometer.features', fromlist=['FeatureConfig']).FeatureConfig(
        cadence_s=60.0, lookback_hours=3.0, windows_min=(15, 60, 180), amplitude_window_min=15
    ))
    targets = build_targets(residual, horizons_hours=(1,), amplitude_window_min=15)
    valid = features["persistence_amplitude_nt"].notna()
    return features.loc[valid], {1: targets[1].loc[valid]}, config


def test_forecaster_fit_predict_and_blend():
    features, targets, config = _dataset()
    model = GeomagneticForecaster(config).fit(features.iloc[:900], {1: targets[1].iloc[:900]})
    model.calibrate_blend(features.iloc[900:1100], {1: targets[1].iloc[900:1100]})
    result = model.predict(features.iloc[1100:])
    assert 1 in result
    assert result[1]["predicted_amplitude_nt"] >= 0
    assert 0.0 <= result[1]["storm_probability"] <= 1.0
    assert 0.0 <= result[1]["confidence"] <= 1.0
    assert 0.0 <= result[1]["blend_weight"] <= 1.0


def test_evaluation_is_finite():
    features, targets, config = _dataset()
    model = GeomagneticForecaster(config).fit(features.iloc[:900], {1: targets[1].iloc[:900]})
    metrics = model.evaluate(features.iloc[900:1100], {1: targets[1].iloc[900:1100]})
    assert np.isfinite(metrics[1].mae_nt)
    assert np.isfinite(metrics[1].rmse_nt)
