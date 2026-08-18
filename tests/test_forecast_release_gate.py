from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "magnetometer"))

from forecast_release_gate import evaluate_forecast_release  # noqa: E402


def _metrics(samples=2000, precision=0.9, recall=0.8, f1=0.85, far=0.005, ece=0.03, mae_ok=True):
    return {
        "samples": samples,
        "storm": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate": far,
            "ece": ece,
        },
        "beats_persistence_mae": mae_ok,
    }


def test_release_gate_passes_when_all_horizons_meet_criteria() -> None:
    report = {str(h): _metrics() for h in (1, 3, 6)}
    gate = evaluate_forecast_release(report)
    assert gate["passed"] is True


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
