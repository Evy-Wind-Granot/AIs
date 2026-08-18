import numpy as np
import pandas as pd
import pytest

from magnetometer.features import FeatureConfig, build_features, build_targets


def _series(periods=900):
    index = pd.date_range("2024-01-01", periods=periods, freq="min", tz="UTC")
    values = np.sin(np.arange(periods) / 20.0) * 10.0
    return pd.Series(values, index=index)


def test_features_are_causal_and_include_required_statistics():
    residual = _series(900)
    features = build_features(residual, config=FeatureConfig(windows_min=(15, 60, 180)))
    required = {"residual", "residual_dbdt", "residual_std_15m", "residual_ptp_60m", "residual_energy_180m", "persistence_amplitude_nt", "residual_lag_60m"}
    assert required.issubset(features.columns)
    expected = residual.iloc[86:101].max() - residual.iloc[86:101].min()
    assert features.loc[residual.index[100], "residual_ptp_15m"] == pytest.approx(expected)


def test_missing_external_indices_are_explicit():
    residual = _series(900)
    features = build_features(residual, kp=None, dst=None)
    assert features["kp_available"].eq(0).all()
    assert features["dst_available"].eq(0).all()
    assert features["kp_age_min"].iloc[-1] >= 1e6


def test_targets_start_strictly_after_forecast_time():
    residual = _series(1000)
    targets = build_targets(residual, horizons_hours=(1,), amplitude_window_min=15)
    first = targets[1].first_valid_index()
    assert first is not None
    assert first > residual.index[0]
