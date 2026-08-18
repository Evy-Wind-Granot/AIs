import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "magnetometer"))

from models.forecaster import ForecastConfig, GeomagneticForecaster  # noqa: E402


def synthetic_frame(n: int = 5000) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(123)
    t = np.arange(n, dtype=float)
    residual = 4.0 * np.sin(t / 31.0) + rng.normal(0, 1.0, n)
    # Place sustained events across train, calibration, and final-test partitions.
    # The 6h target needs 360 future samples, so the fixture must be long enough
    # to retain >=200 valid target rows in the final test partition.
    for start, stop, amp in (
        (700, 820, 70),
        (1800, 1950, 90),
        (3000, 3150, 55),
        (4000, 4150, 75),
        (4650, 4800, 85),
    ):
        residual[start:stop] += amp
    return pd.DataFrame({"residual": residual, "kp": 2.0, "dst": -10.0}, index=idx)


def test_forecaster_fit_predict_and_roundtrip(tmp_path: Path) -> None:
    frame = synthetic_frame()
    config = ForecastConfig(max_iter=80, min_samples_leaf=8, validation_fraction=0.20)
    model = GeomagneticForecaster(config=config)
    report = model.fit(frame, cadence_s=60.0)
    assert set(report["horizons"]) == {"1", "3", "6"}

    result = model.predict(frame.tail(720), cadence_s=60.0, current_rule_tier="quiet")
    assert set(result.horizons) == {"1", "3", "6"}
    for row in result.horizons.values():
        assert row["predicted_amplitude_nt"] is not None
        assert 0.0 <= float(row["storm_probability"]) <= 1.0
        assert isinstance(row["forecast_tier"], str)

    model_path = tmp_path / "forecaster"
    model.save_model(model_path)
    loaded = GeomagneticForecaster.load_model(model_path)
    loaded_result = loaded.predict(frame.tail(720), cadence_s=60.0)
    for horizon in result.horizons:
        assert np.isfinite(float(loaded_result.horizons[horizon]["predicted_amplitude_nt"]))
        assert abs(
            float(result.horizons[horizon]["storm_probability"])
            - float(loaded_result.horizons[horizon]["storm_probability"])
        ) < 1e-12


def test_horizon_predictions_are_not_future_features() -> None:
    frame = synthetic_frame()
    config = ForecastConfig(max_iter=50, min_samples_leaf=8)
    model = GeomagneticForecaster(config=config)
    model.fit(frame, cadence_s=60.0)
    baseline = model.predict(frame.tail(720), cadence_s=60.0)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("residual")] += 1000.0
    changed_result = model.predict(changed.tail(720), cadence_s=60.0)
    assert any(
        abs(
            float(baseline.horizons[h]["predicted_amplitude_nt"])
            - float(changed_result.horizons[h]["predicted_amplitude_nt"])
        ) > 0
        for h in baseline.horizons
    )
