import numpy as np
import pandas as pd

from magnetometer.forecasting.release_gate import evaluate_forecast_release
from train_magnetometer_forecaster_production import _validate_training_frame


def _good_report():
    return {
        "1": {"samples": 1200, "regression_mae_nt": 10.0, "regression_rmse_nt": 15.0, "beats_persistence_mae": True, "storm": {"recall": 0.80, "precision": 0.90, "f1": 0.84, "false_alarm_rate": 0.005, "ece": 0.04, "brier_score": 0.03, "positive_rate": 0.10}},
        "3": {"samples": 1200, "regression_mae_nt": 10.0, "regression_rmse_nt": 15.0, "beats_persistence_mae": True, "storm": {"recall": 0.70, "precision": 0.80, "f1": 0.74, "false_alarm_rate": 0.005, "ece": 0.04, "brier_score": 0.03, "positive_rate": 0.10}},
        "6": {"samples": 1200, "regression_mae_nt": 10.0, "regression_rmse_nt": 15.0, "beats_persistence_mae": True, "storm": {"recall": 0.60, "precision": 0.75, "f1": 0.66, "false_alarm_rate": 0.005, "ece": 0.04, "brier_score": 0.03, "positive_rate": 0.10}},
    }


def test_release_gate_passes_valid_report():
    assert evaluate_forecast_release(_good_report())["passed"] is True


def test_release_gate_rejects_nan_metric():
    report = _good_report(); report["1"]["storm"]["ece"] = np.nan
    result = evaluate_forecast_release(report)
    assert result["passed"] is False
    assert result["horizons"]["1"]["checks"]["probability_ece"] is False


def test_release_gate_rejects_single_class_test_set():
    report = _good_report(); report["3"]["storm"]["positive_rate"] = 0.0
    result = evaluate_forecast_release(report)
    assert result["passed"] is False
    assert result["horizons"]["3"]["checks"]["both_test_classes_present"] is False


def test_training_frame_requires_utc_and_finite_residual():
    idx = pd.date_range("2025-01-01", periods=10, freq="min", tz="UTC")
    frame = pd.DataFrame({"residual": 1.0, "kp": 2.0, "dst": -5.0}, index=idx)
    _validate_training_frame(frame, "TEST")
    broken = frame.copy(); broken.iloc[0, 0] = np.nan
    try:
        _validate_training_frame(broken, "TEST")
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite residual should fail the data contract")
