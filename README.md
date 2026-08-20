# Cascadia Sentinel — Model Demos

The repository is organized by instrument, with the magnetometer split into two independent subsystems.

## Repository layout

```text
AIs/
├── magnetometer/
│   ├── detector/          # deterministic/QDC activity detection
│   └── forecasting/       # causal ML forecasting + hybrid inference
├── seisometer/            # PhaseNet / EQTransformer seismic processing
├── weather/               # production weather time-series forecasting
├── run_all_demos.py       # canonical all-instrument launcher
└── run_all_demos.sh       # shell wrapper for run_all_demos.py
```

### Magnetometer

- `magnetometer/detector/` is the current deterministic/QDC activity-detection subsystem.
- `magnetometer/forecasting/` is the separate causal ML forecasting subsystem.

Forecasting is advisory and forward-looking; it does not replace the deterministic detector.

### Seisometer

`seisometer/` contains the seismic demo and stored PhaseNet/EQTransformer results.

### Weather

`weather/` contains the production TSFM engine and its benchmark entry point.

## Run everything

```bash
./run_all_demos.sh --self-test
# or
python run_all_demos.py --self-test
```

For public-data runs:

```bash
./run_all_demos.sh --real-data
```

There are no longer root-level duplicate instrument demo wrappers or the random-demo launcher.
