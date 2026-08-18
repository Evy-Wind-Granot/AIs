"""Offline tests for causal ML feature generation and inference."""
from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from magnetometer.features import FeatureConfig, build_features, build_targets
from models.forecaster import (
    ForecastConfig,
    GeomagneticForecaster,
    build_training_data,
    load_model,
    save_model,
)


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range(
            "2024-05-08", periods=1000, freq="min", tz="UTC"
        )
        t = np.arange(len(self.index))
        self.residual = pd.Series(
            8.0 * np.sin(2 * np.pi * t / 1440.0) + (t % 180) * 0.03,
            index=self.index,
        )

    def test_required_causal_features_exist(self) -> None:
        frame = build_features(
            self.residual,
            pd.Series(2.0, index=self.index),
            pd.Series(-5.0, index=self.index),
            config=FeatureConfig(
                windows_min=(15, 60, 180),
                lookback_hours=3,
            ),
        )
        for column in (
            "residual_std_15m",
            "residual_ptp_60m",
            "residual_energy_180m",
            "residual_dbdt",
            "residual_lag_60m",
            "kp",
            "dst",
        ):
            self.assertIn(column, frame)
        self.assertTrue(frame.index.equals(self.index))
        self.assertTrue(pd.isna(frame["kp"].iloc[0]))
        self.assertTrue(pd.isna(frame["dst"].iloc[0]))
        self.assertAlmostEqual(float(frame["kp"].iloc[180]), 2.0)
        self.assertAlmostEqual(float(frame["dst"].iloc[60]), -5.0)

    def test_feature_values_are_causal(self) -> None:
        cfg = FeatureConfig(windows_min=(15, 60), lookback_hours=1)
        original = build_features(self.residual, config=cfg)
        changed = self.residual.copy()
        changed.iloc[-1] += 10000.0
        changed_frame = build_features(changed, config=cfg)
        pd.testing.assert_frame_equal(original.iloc[:-1], changed_frame.iloc[:-1])

    def test_future_target_starts_after_horizon(self) -> None:
        targets = build_targets(
            self.residual,
            horizons_hours=(1,),
            amplitude_window_min=30,
        )
        changed = self.residual.copy()
        changed.iloc[30] += 10000.0
        changed_targets = build_targets(
            changed,
            horizons_hours=(1,),
            amplitude_window_min=30,
        )
        self.assertAlmostEqual(
            float(targets[1].iloc[0]),
            float(changed_targets[1].iloc[0]),
        )

    def test_future_target_window_is_strictly_after_horizon(self) -> None:
        targets = build_targets(
            self.residual,
            horizons_hours=(1,),
            amplitude_window_min=30,
        )
        changed = self.residual.copy()
        changed.iloc[59] += 10000.0
        changed_targets = build_targets(
            changed,
            horizons_hours=(1,),
            amplitude_window_min=30,
        )
        self.assertAlmostEqual(
            float(targets[1].iloc[0]),
            float(changed_targets[1].iloc[0]),
        )


class ForecasterTests(unittest.TestCase):
    def _dataset(self) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
        index = pd.date_range(
            "2024-05-08", periods=3000, freq="min", tz="UTC"
        )
        t = np.arange(len(index), dtype=float)
        residual = pd.Series(
            8.0 * np.sin(2 * np.pi * t / 1440.0), index=index
        )
        for start in (800, 1700, 2450):
            residual.iloc[start : start + 100] += (
                np.sin(np.linspace(0, 8 * np.pi, 100)) * 120
            )
        kp = pd.Series(2.0, index=index)
        dst = pd.Series(-5.0, index=index)
        cfg = self._config()
        return build_training_data(
            residual, kp, dst, cadence_s=60.0, config=cfg
        )

    def test_fit_predict_evaluate_and_serialization(self) -> None:
        features, targets = self._dataset()
        split = int(len(features) * 0.8)
        train = features.iloc[:split]
        train_targets = {h: y.iloc[:split] for h, y in targets.items()}
        test = features.iloc[split:]
        test_targets = {h: y.iloc[split:] for h, y in targets.items()}

        model = GeomagneticForecaster(self._config())
        model.fit(train, train_targets)
        predictions = model.predict(test.tail(1))
        self.assertEqual(set(predictions), {1, 3, 6})
        for value in predictions.values():
            self.assertGreaterEqual(value["predicted_amplitude_nt"], 0.0)
            self.assertGreaterEqual(value["storm_probability"], 0.0)
            self.assertLessEqual(value["storm_probability"], 1.0)

        metrics = model.evaluate(test, test_targets)
        self.assertEqual(set(metrics), {1, 3, 6})
        for evaluation in metrics.values():
            self.assertGreater(evaluation.n_samples, 0)
            self.assertTrue(np.isfinite(evaluation.rmse_nt))
            self.assertTrue(np.isfinite(evaluation.mae_nt))

        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(model, f"{tmp}/model.pkl")
            restored = load_model(path)
            self.assertEqual(restored.feature_columns, model.feature_columns)
            self.assertEqual(restored.predict(test.tail(1)), predictions)

    @staticmethod
    def _config() -> ForecastConfig:
        return ForecastConfig(
            horizons_hours=(1, 3, 6),
            lookback_hours=12,
            feature_windows_min=(15, 60, 180, 360, 720),
            max_iter=80,
            min_samples_leaf=10,
        )


if __name__ == "__main__":
    unittest.main()
