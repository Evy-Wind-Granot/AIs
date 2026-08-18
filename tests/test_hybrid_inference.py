import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "magnetometer"))

from hybrid_inference import build_aligned_forecast_frame, hybrid_status_payload  # noqa: E402


def test_aligned_forecast_frame() -> None:
    idx = pd.date_range("2025-01-01", periods=120, freq="min", tz="UTC")
    residual = np.arange(120, dtype=float)
    # Keep the latest reference observation within the production 3-hour
    # alignment tolerance. The test should validate the alignment contract,
    # not rely on an out-of-tolerance value being forward-filled.
    kp_idx = pd.date_range("2024-12-31 22:00", periods=4, freq="1h", tz="UTC")
    kp = pd.Series([1.0, 2.0, 4.0, 6.0], index=kp_idx)
    dst = pd.Series([-5.0, -10.0, -30.0, -60.0], index=kp_idx)
    frame = build_aligned_forecast_frame(residual, idx, kp_series=kp, dst_series=dst)
    assert list(frame.columns) == ["residual", "kp", "dst"]
    assert frame.index.equals(idx)
    assert frame["kp"].iloc[-1] == 6.0
    assert frame["dst"].iloc[-1] == -60.0


def test_alignment_respects_tolerance() -> None:
    idx = pd.date_range("2025-01-01", periods=120, freq="min", tz="UTC")
    residual = np.zeros(120, dtype=float)
    too_old = pd.Series([6.0], index=pd.DatetimeIndex([pd.Timestamp("2024-12-31 00:00", tz="UTC")]))
    frame = build_aligned_forecast_frame(residual, idx, kp_series=too_old)
    assert pd.isna(frame["kp"].iloc[-1])


def test_hybrid_payload_shape() -> None:
    class FakeResult:
        model_version = "test"
        current_residual_nt = 5.0
        anomaly_delta = 0.6
        divergence = True
        horizons = {
            "1": {"forecast_tier": "active", "storm_probability": 0.4, "model_confidence": 0.6},
            "3": {"forecast_tier": "minor_storm", "storm_probability": 0.8, "model_confidence": 0.8},
            "6": {"forecast_tier": "major_storm", "storm_probability": 0.9, "model_confidence": 0.9},
        }

    class FakeForecaster:
        def predict(self, frame, *, cadence_s, current_rule_tier):
            assert not frame.empty
            return FakeResult()

    idx = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")
    frame = pd.DataFrame({"residual": np.zeros(20)}, index=idx)
    payload = hybrid_status_payload(
        frame,
        deterministic_tier="quiet",
        forecaster=FakeForecaster(),
    )
    assert payload["realtime"]["tier"] == "quiet"
    assert payload["hybrid"]["forecast_highest_tier"] == "major_storm"
    assert payload["hybrid"]["forecast_trend"] == "escalating"
    assert payload["hybrid"]["divergence"] is True
