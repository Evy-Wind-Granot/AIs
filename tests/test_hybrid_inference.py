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
    kp_idx = pd.date_range("2024-12-31", periods=3, freq="1h", tz="UTC")
    kp = pd.Series([1.0, 2.0, 6.0], index=kp_idx)
    dst = pd.Series([-5.0, -10.0, -60.0], index=kp_idx)
    frame = build_aligned_forecast_frame(residual, idx, kp_series=kp, dst_series=dst)
    assert list(frame.columns) == ["residual", "kp", "dst"]
    assert frame.index.equals(idx)
    assert frame["kp"].iloc[-1] == 6.0
    assert frame["dst"].iloc[-1] == -60.0


def test_hybrid_payload_shape() -> None:
    # Use a tiny fake model interface so this test remains independent of any
    # trained artifact; the production forecaster integration is covered by
    # test_forecaster.py.
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
