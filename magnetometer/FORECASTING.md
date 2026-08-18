# Geomagnetic Hybrid Forecasting

The magnetometer monitor keeps the deterministic Harmonic/QDC residual and activity classifier as the current-state authority. The forecasting layer predicts future disturbance magnitude and storm-threshold breach probability at +1h, +3h, and +6h.

## Model architecture

1. INTERMAGNET observed field is converted to the existing production residual: `observed - QDC/harmonic baseline`.
2. Causal rolling features are computed over 15m, 1h, 3h, and 6h windows.
3. Lagged residual/dB/dt context and aligned Kp/Dst are added.
4. Each horizon has a regression model for future peak absolute residual and a binary classifier for storm-threshold breach.
5. Deterministic current tier remains independent and is merged with the ML forecast through `hybrid_pipeline.py`.

The default backend is scikit-learn `HistGradientBoosting`; LightGBM can be selected when installed. Tree boosting is intentionally used first because it is fast at inference, handles nonlinear interactions well, and keeps the production dependency footprint small. LightGBM/XGBoost both support early-stopping workflows suitable for larger experiments; scikit-learn's `TimeSeriesSplit`/gap methodology is the recommended validation pattern for chronological evaluation.

## Offline self-test

```bash
python train_magnetometer_forecaster.py \
  --self-test \
  --model-path /tmp/magnetometer_forecaster
```

This trains all three horizons on deterministic synthetic data, evaluates them, saves the artifact, loads it back, and performs inference.

## Train on real data

```bash
python train_magnetometer_forecaster.py \
  --observatory VIC \
  --start-date 2024-01-01 \
  --days 180 \
  --backend sklearn \
  --model-path magnetometer/data/models/magnetometer_forecaster
```

The training report contains RMSE/MAE for the amplitude forecast and precision/recall/F1/false-alarm-rate for the storm-breach classifier at every horizon. It also reports a persistence baseline so the ML model can be required to beat a trivial forecast.

## Hybrid inference

With a trained model:

```bash
python magnetometer/hybrid_pipeline.py \
  --observatory VIC \
  --start-date 2026-08-18 \
  --days 2 \
  --model-path magnetometer/data/models/magnetometer_forecaster
```

The payload separates:

- `realtime`: deterministic current classification;
- `forecast`: +1h/+3h/+6h magnitude, storm probability, forecast tier, and confidence;
- `hybrid`: highest forecast tier, escalation trend, normalized anomaly delta, and divergence flag.

The ML layer is advisory. If the model artifact cannot be loaded or inference fails, the wrapper returns the deterministic status and marks the ML portion degraded rather than suppressing current monitoring.

## Persistence and compatibility

Model artifacts are stored locally as a Joblib bundle plus a JSON manifest containing the model version, feature schema, training range, dependency versions, and configuration. Joblib/pickle artifacts must only be loaded from trusted local sources. Runtime state helpers preserve unrelated keys in `.magnetometer_state.json` and update only forecast-specific fields.
