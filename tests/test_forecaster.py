import unittest

import numpy as np
import pandas as pd

from magnetometer.features import FeatureConfig, build_features, build_targets
from models.forecaster import ForecastConfig, GeomagneticForecaster


def _dataset(periods=1400):
    index = pd.date_range("2024-01-01", periods=periods, freq="min", tz="UTC")
    x = np.arange(periods, dtype=float)
    residual = pd.Series(10.0 * np.sin(x / 25.0) + 2.0 * np.sin(x / 7.0), index=index)
    config = ForecastConfig(horizons_hours=(1,), feature_windows_min=(15, 60, 180), amplitude_window_min=15)
    feature_config = FeatureConfig(cadence_s=60.0, lookback_hours=3.0, windows_min=(15, 60, 180), amplitude_window_min=15)
    features = build_features(residual, config=feature_config)
    targets = build_targets(residual, horizons_hours=(1,), amplitude_window_min=15)
    valid = features["persistence_amplitude_nt"].notna()
    return features.loc[valid], {1: targets[1].loc[valid]}, config


class ForecasterTests(unittest.TestCase):
    def test_fit_predict_and_blend(self):
        features, targets, config = _dataset()
        model = GeomagneticForecaster(config).fit(features.iloc[:900], {1: targets[1].iloc[:900]})
        model.calibrate_blend(features.iloc[900:1100], {1: targets[1].iloc[900:1100]})
        result = model.predict(features.iloc[1100:])
        self.assertIn(1, result)
        self.assertGreaterEqual(result[1]["predicted_amplitude_nt"], 0)
        self.assertTrue(0.0 <= result[1]["storm_probability"] <= 1.0)
        self.assertTrue(0.0 <= result[1]["confidence"] <= 1.0)
        self.assertTrue(0.0 <= result[1]["blend_weight"] <= 1.0)

    def test_evaluation_is_finite(self):
        features, targets, config = _dataset()
        model = GeomagneticForecaster(config).fit(features.iloc[:900], {1: targets[1].iloc[:900]})
        metrics = model.evaluate(features.iloc[900:1100], {1: targets[1].iloc[900:1100]})
        self.assertTrue(np.isfinite(metrics[1].mae_nt))
        self.assertTrue(np.isfinite(metrics[1].rmse_nt))


if __name__ == "__main__":
    unittest.main()
