import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from magnetometer.features import FeatureConfig, build_features, build_targets
from models.forecaster import ForecastConfig, GeomagneticForecaster, load_model, save_model


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
        self.assertGreaterEqual(result[1]["persistence_amplitude_nt"], 0)
        self.assertTrue(np.isfinite(result[1]["model_delta_nt"]))
        self.assertTrue(0.0 <= result[1]["storm_probability"] <= 1.0)
        self.assertTrue(0.0 <= result[1]["confidence"] <= 1.0)
        self.assertTrue(0.0 <= result[1]["data_quality"] <= 1.0)
        self.assertTrue(0.0 <= result[1]["blend_weight"] <= 1.0)

    def test_evaluation_is_finite(self):
        features, targets, config = _dataset()
        model = GeomagneticForecaster(config).fit(features.iloc[:900], {1: targets[1].iloc[:900]})
        metrics = model.evaluate(features.iloc[900:1100], {1: targets[1].iloc[900:1100]})
        self.assertTrue(np.isfinite(metrics[1].mae_nt))
        self.assertTrue(np.isfinite(metrics[1].rmse_nt))

    def test_health_and_production_serialization(self):
        features, targets, config = _dataset()
        model = GeomagneticForecaster(config).fit(features.iloc[:900], {1: targets[1].iloc[:900]})
        self.assertEqual(model.health_check()["schema_version"], 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecaster.pkl"
            model.training_metadata["production_gate"] = "failed"
            save_model(model, path)
            with self.assertRaises(ValueError):
                load_model(path)
            model.training_metadata["production_gate"] = "passed"
            save_model(model, path)
            loaded = load_model(path)
            self.assertTrue(loaded.fitted)
            self.assertEqual(loaded.health_check()["production_gate"], "passed")


if __name__ == "__main__":
    unittest.main()
