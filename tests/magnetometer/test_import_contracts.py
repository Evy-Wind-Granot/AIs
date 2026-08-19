def test_detecting_import_contract():
    from magnetometer.detecting import flag_activity
    from magnetometer.detecting.detector_core import DetectorProfile

    assert callable(flag_activity)
    assert DetectorProfile.__name__ == "DetectorProfile"


def test_forecasting_import_contract():
    from magnetometer.forecasting.models.forecaster import GeomagneticForecaster
    from magnetometer.forecasting.hybrid_inference import build_aligned_forecast_frame

    assert GeomagneticForecaster.__name__ == "GeomagneticForecaster"
    assert callable(build_aligned_forecast_frame)
