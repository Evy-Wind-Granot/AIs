import warnings

import numpy as np
import pandas as pd

from magnetometer.forecasting.feature_engineering import (
    build_supervised_dataset,
    make_forecast_features,
    make_future_targets,
    sequence_tensor,
)


def synthetic_frame(n: int = 1000) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    t = np.arange(n, dtype=float)
    residual = 5.0 * np.sin(t / 50.0) + 0.5 * np.cos(t / 7.0)
    residual[300:420] += 50.0
    return pd.DataFrame({"residual": residual, "kp": 2.0, "dst": -10.0}, index=idx)


def test_features_are_causal() -> None:
    frame = synthetic_frame()
    first = make_forecast_features(frame.iloc[:700], cadence_s=60.0)
    changed = frame.iloc[:700].copy()
    changed.iloc[-1, changed.columns.get_loc("residual")] += 500.0
    second = make_forecast_features(changed, cadence_s=60.0)
    cols = [c for c in first.columns if c.startswith("residual_") or c.startswith("dbdt")]
    pd.testing.assert_frame_equal(first.iloc[:699][cols], second.iloc[:699][cols])


def test_future_targets_detect_storm_breach() -> None:
    frame = synthetic_frame(500)
    targets = make_future_targets(frame["residual"], cadence_s=60.0, horizons_hours=(1, 3), storm_threshold_nt=35.0)
    assert targets["target_storm_1h"].max() == 1.0
    assert targets["target_storm_3h"].max() == 1.0
    assert pd.isna(targets["target_peak_abs_3h"].iloc[-1])


def test_sequence_tensor_shape() -> None:
    frame = make_forecast_features(synthetic_frame(), cadence_s=60.0)
    names = ["residual", "residual_abs", "dbdt", "kp", "dst"]
    tensor, timestamps = sequence_tensor(frame, feature_names=names, sequence_length=60)
    assert tensor.ndim == 3
    assert tensor.shape[1:] == (60, len(names))
    assert len(timestamps) == tensor.shape[0]


def test_supervised_alignment() -> None:
    features, targets = build_supervised_dataset(synthetic_frame(900), cadence_s=60.0)
    assert features.index.equals(targets.index)
    assert "target_peak_abs_6h" in targets.columns


def test_feature_builder_is_warning_free() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
        make_forecast_features(synthetic_frame(2000), cadence_s=60.0)
