import unittest

from train_magnetometer_forecaster import _gate_horizon


class ProductionGateTests(unittest.TestCase):
    def _stable(self, *, min_rmse=-1.0, mean_rmse=2.0):
        return {
            "mean_mae_improvement_percent": 3.0,
            "median_mae_improvement_percent": 3.0,
            "min_mae_improvement_percent": -2.0,
            "positive_mae_fold_fraction": 0.75,
            "mean_rmse_improvement_percent": mean_rmse,
            "median_rmse_improvement_percent": 2.0,
            "min_rmse_improvement_percent": min_rmse,
        }

    def test_one_hour_requires_strict_rmse_stability(self):
        ok, failures = _gate_horizon(1, 1.0, 1.0, 2.0, 1.0, self._stable(min_rmse=-5.1))
        self.assertFalse(ok)
        self.assertIn("worst_fold_rmse", failures)

    def test_three_hour_allows_modest_single_fold_rmse_regression(self):
        ok, failures = _gate_horizon(3, 1.0, 1.0, 2.0, 1.0, self._stable(min_rmse=-6.9))
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_six_hour_still_fails_when_aggregate_mae_is_negative(self):
        stability = self._stable(min_rmse=-4.0)
        stability["mean_mae_improvement_percent"] = -3.2
        stability["median_mae_improvement_percent"] = -3.8
        ok, failures = _gate_horizon(6, 7.0, 4.0, -3.5, -30.0, stability)
        self.assertFalse(ok)
        self.assertIn("test_mae", failures)
        self.assertIn("mean_mae", failures)


if __name__ == "__main__":
    unittest.main()
