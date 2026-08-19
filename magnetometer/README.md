# Magnetometer

The magnetometer subsystem is split into two responsibilities:

```text
magnetometer/
├── detecting/                 # real-time / historical activity detection
│   ├── detector_core.py       # canonical detector API
│   ├── diagnostic.py          # operational single-window diagnostics
│   ├── calibrate.py           # heuristic/profile calibration
│   ├── validation.py          # multi-year validation
│   ├── release_gate.py        # final detector certification gate
│   └── metrics.py             # detector scoring primitives
│
├── forecasting/               # future geomagnetic activity forecasting
│   ├── feature_engineering.py
│   ├── hybrid_inference.py
│   ├── release_gate.py
│   └── models/
│       ├── forecaster.py
│       ├── certified_forecaster.py
│       └── production_forecaster.py
│
└── legacy root modules        # compatibility during migration
```

## Canonical imports

New code should use:

```python
from magnetometer.detecting import flag_activity
from magnetometer.detecting.detector_core import DetectorProfile
from magnetometer.forecasting.feature_engineering import make_forecast_features
from magnetometer.forecasting.models.forecaster import GeomagneticForecaster
```

Do not add new imports against the legacy root-level module paths. They remain
only so existing operational commands and external users do not break while the
package migration is completed.

## Tests

Magnetometer tests live under the repository root:

```text
tests/magnetometer/
├── detecting/
│   └── test_detector_core.py
└── forecasting/
    ├── test_detector_hardening.py
    ├── test_forecast_features.py
    ├── test_forecast_release_gate.py
    ├── test_forecaster.py
    ├── test_hybrid_inference.py
    └── test_production_hardening.py
```

Run the complete magnetometer test suite from the repository root:

```bash
pytest -q tests/magnetometer
```

For the canonical detector tools, prefer module execution from the repository
root, for example:

```bash
python -m magnetometer.detecting.diagnostic --help
python -m magnetometer.detecting.calibrate --help
python -m magnetometer.detecting.validation --help
python -m magnetometer.detecting.release_gate --help
```
