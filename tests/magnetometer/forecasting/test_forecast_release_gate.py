from magnetometer.forecasting.release_gate import evaluate_forecast_release


def _metrics(
    samples=2000,
    precision=0.9,
    recall=0.8,
    f1=0.85,
    far=0.005,
    ece=0.03,
    mae_ok=True,
    positive_rate=0.10,
):
    return {
        "samples": samples,
        "regression_mae_nt": 10.0,
        "regression_rmse_nt": 15.0,
        "storm": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate": far,
            "ece": ece,
            "brier_score": 0.03,
            "positive_rate": positive_rate,
        },
        "beats_persistence_mae": mae_ok,
    }


def test_release_gate_passes_when_all_horizons_meet_criteria() -> None:
    report = {str(h): _metrics() for h in (1, 3, 6)}
    assert evaluate_forecast_release(report)["passed"] is True


def test_release_gate_blocks_weak_horizon() -> None:
    report = {str(h): _metrics() for h in (1, 3, 6)}
    report["6"] = _metrics(recall=0.2)
    gate = evaluate_forecast_release(report)
    assert gate["passed"] is False
    assert gate["horizons"]["6"]["checks"]["storm_recall"] is False


def test_release_gate_blocks_persistence_regression() -> None:
    report = {str(h): _metrics() for h in (1, 3, 6)}
    report["3"] = _metrics(mae_ok=False)
    gate = evaluate_forecast_release(report)
    assert gate["passed"] is False
    assert gate["horizons"]["3"]["checks"]["amplitude_mae_beats_persistence"] is False
