# Cascadia Sentinel — Model Demos

The repository is organized by instrument, with the magnetometer further split by responsibility.

## Repository layout

```text
AIs/
├── magnetometer/
│   ├── detector/          # deterministic/QDC activity detection
│   └── forecasting/       # causal ML forecasting + hybrid inference
├── seisometer/            # PhaseNet / EQTransformer seismic processing
├── weather_tsfm_engine_v2_production_hybrid_fixed.py
├── run_all_demos.py
└── run_all_demos.sh
```

### Magnetometer

- `magnetometer/detector/` contains the deterministic magnetometer analysis and activity classification path.
- `magnetometer/forecasting/` contains causal feature engineering, the multi-horizon forecaster, and hybrid inference.

The forecasting system is intentionally separate from the detector: forecasting provides future disturbance/storm-risk predictions and does not replace deterministic detection.

### Seismometer

Seismic processing is under `seisometer/`, including the PhaseNet/EQTransformer demo and stored picking results.

### Compatibility

The historical root entry points `magnetometer_demo.py` and `seismic_demo.py` remain as thin wrappers so existing commands continue to work. New code should import/use the instrument packages directly.

## Quick start

```bash
python magnetometer_demo.py --self-test
python seismic_demo.py --self-test
python run_all_demos.py --self-test
```
