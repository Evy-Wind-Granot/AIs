# Short-Horizon Geomagnetic Forecasting

The production magnetometer pipeline has an optional ML forecasting layer on top of the deterministic harmonic/QDC residual pipeline.

## Production design

- Deterministic QDC/baseline processing remains the source of truth for the current real-time tier.
- Forecast context uses the causal residual (`Observed - QDC`) plus rolling statistics at 15m, 30m, 1h, 3h, 6h, and 12h scales.
- Kp and Dst are delayed by conservative release lags (3h for Kp, 1h for Dst) so finalized global-index values cannot leak into an earlier prediction timestamp.
- Missing Kp/Dst are represented explicitly with availability and age features; missing global indices are never fabricated.
- Forecast horizons are +1h, +3h, and +6h.
- Each horizon has a gradient-boosted regression model for the correction to causal persistence plus a binary minor-storm-or-higher classifier.
- A validation-only blend weight combines ML and persistence. A weight of zero is a safe persistence fallback for a horizon where ML adds no value.
- The design is intentionally gray-box: deterministic QDC processing supplies the state representation and the ML model learns residual disturbance dynamics.

## Leakage-safe training protocol

For the current 1h/3h/6h horizons with a 3-hour amplitude target window, the maximum target reach is 9 hours. The training script inserts a **9-hour purge gap** between train/validation and validation/test periods.

The target at time `t` is the peak-to-peak residual amplitude in:

```text
[t + horizon, t + horizon + 3h)
```

so no target sample overlaps the feature timestamp.

The production split is approximately:

```text
65% TRAIN | 9h PURGE | 15% VALIDATION | 9h PURGE | 20% TEST
```

The untouched test set is never used to select hyperparameters or blend weights.

## Production gate

A model is **not** published as a production artifact unless:

1. ML beats persistence at every configured horizon on the untouched chronological test set; and
2. validation does not show more than a 2% MAE regression against persistence at any horizon.

If the gate fails, training exits with status `3` and leaves the production artifact untouched.

`--save-candidate` can save a failed model under `models/artifacts/candidates/` for research, but the metadata remains `production_gate = failed` and the live pipeline refuses to load it.

## Train on real VIC data

For a production-quality run, use at least six months when the upstream data sources permit it:

```bash
python train_magnetometer_forecaster.py \
  --observatory VIC \
  --start-date 2024-01-01 \
  --days 180
```

The script:

1. Fetches INTERMAGNET minute data and Kp/Dst concurrently.
2. Runs the existing deterministic QDC/residual pipeline.
3. Builds causal ML features and strictly future targets.
4. Performs a purged chronological train/validation/test split.
5. Calibrates the ML/persistence blend using validation only.
6. Reports ML versus persistence MAE/RMSE and storm precision/recall/F1.
7. Refits only on train + validation.
8. Applies the untouched test-set production gate.
9. Writes `models/artifacts/<observatory>_forecaster.pkl` only after the gate passes.

If Dst is unavailable from Kyoto WDC, training continues without Dst and records that limitation in model metadata. It does not fabricate Dst values.

## Live/batch inference

Once a production-approved artifact exists, the normal pipeline automatically adds:

- `forecast.horizons`: +1h/+3h/+6h amplitude, probability, tier, confidence, data quality, and blend weight.
- `hybrid.real_time`: current deterministic tier and source.
- `hybrid.forecasted_status`: predicted tier by horizon.
- `hybrid.model_confidence`: confidence by horizon.
- `hybrid.divergence`: signed and absolute tier divergence.
- `hybrid.divergence.significant`: true when the forecast differs by at least two activity tiers.

Set `MAGNETOMETER_FORECAST_MODEL` to override the default artifact path.

If no approved artifact exists, inference remains deterministic. ML failure can never replace or alter the current deterministic classification.

## Tests

Offline tests include:

```bash
python test_magnetometer_ml.py
python -m unittest discover -s tests -p 'test_*.py'
```

They cover causal features, Kp/Dst release lags, missing-index signals, strict future targets, persistence-aware model inference, blend calibration, evaluation, and serialization.

Historical upstream-data regression remains opt-in:

```bash
RUN_REAL_DATA_TESTS=1 python -m unittest test_magnetometer_ml_real.py
```

It is intentionally not part of CI because INTERMAGNET, GFZ, and Kyoto WDC are external services.
