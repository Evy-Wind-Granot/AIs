import numpy as np
import pandas as pd

from magnetometer.forecasting.feature_engineering import make_forecast_features
from magnetometer.forecasting.models.certified_forecaster import GeomagneticForecaster


def _frame(n=1200):
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    residual = 4.0 * np.sin(np.arange(n) / 35.0)
    residual[500:620] += np.linspace(5.0, 60.0, 120)
    residual[620:700] += 60.0
    return pd.DataFrame({"residual": residual, "kp": 2.0, "dst": -8.0}, index=idx)


def test_detector_has_event_state_features():
    features = make_forecast_features(_frame())
    required = {
        "consecutive_above_35nt_min",
        "minutes_since_35nt",
        "above_35nt_fraction_60m",
        "residual_slope_60m",
        "dbdt_accel_max_60m",
        "kp_max_180m",
        "dst_change_180m",
    }
    assert required.issubset(features.columns)


def test_event_features_are_causal():
    frame = _frame()
    first = make_forecast_features(frame.iloc[:900])
    changed = frame.iloc[:900].copy()
    changed.iloc[-1, changed.columns.get_loc("residual")] += 500.0
    second = make_forecast_features(changed)
    cols = [
        "consecutive_above_35nt_min",
        "minutes_since_35nt",
        "above_35nt_fraction_60m",
        "residual_slope_60m",
        "dbdt_accel_max_60m",
    ]
    pd.testing.assert_frame_equal(first.iloc[:899][cols], second.iloc[:899][cols])


def test_positive_class_weight_is_bounded():
    y = np.array([0] * 990 + [1] * 10)
    weights = GeomagneticForecaster._class_weights(y)
    assert np.all(weights[y == 0] == 1.0)
    assert 1.5 <= float(weights[y == 1][0]) <= 4.0
