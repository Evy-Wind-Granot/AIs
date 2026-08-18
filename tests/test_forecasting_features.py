import unittest

import numpy as np
import pandas as pd

from magnetometer.features import FeatureConfig, build_features, build_targets


def _series(periods=900):
    index = pd.date_range("2024-01-01", periods=periods, freq="min", tz="UTC")
    values = np.sin(np.arange(periods) / 20.0) * 10.0
    return pd.Series(values, index=index)


class ForecastFeatureTests(unittest.TestCase):
    def test_features_are_causal_and_include_required_statistics(self):
        residual = _series(900)
        features = build_features(residual, config=FeatureConfig(windows_min=(15, 60, 180)))
        required = {
            "residual", "residual_dbdt", "residual_std_15m", "residual_ptp_60m",
            "residual_energy_180m", "persistence_amplitude_nt", "residual_lag_60m",
        }
        self.assertTrue(required.issubset(features.columns))
        expected = residual.iloc[86:101].max() - residual.iloc[86:101].min()
        self.assertAlmostEqual(float(features.loc[residual.index[100], "residual_ptp_15m"]), float(expected))

    def test_missing_external_indices_are_explicit(self):
        features = build_features(_series(900), kp=None, dst=None)
        self.assertTrue(features["kp_available"].eq(0).all())
        self.assertTrue(features["dst_available"].eq(0).all())
        self.assertGreaterEqual(float(features["kp_age_min"].iloc[-1]), 1e6)

    def test_targets_are_strictly_future(self):
        residual = _series(1000)
        targets = build_targets(residual, horizons_hours=(1,), amplitude_window_min=15)
        changed = residual.copy()
        changed.iloc[0] += 10000.0
        changed_targets = build_targets(changed, horizons_hours=(1,), amplitude_window_min=15)
        # Target at t is [t+1h, t+1h+15m), so changing t cannot alter it.
        self.assertAlmostEqual(float(targets[1].iloc[0]), float(changed_targets[1].iloc[0]))
        self.assertEqual(targets[1].first_valid_index(), residual.index[0])


if __name__ == "__main__":
    unittest.main()
