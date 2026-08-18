# Short-Horizon Geomagnetic Forecasting

The production magnetometer pipeline has an optional ML forecasting layer on top of the deterministic harmonic/QDC residual pipeline.

## Production design

- Deterministic QDC/baseline processing remains the source of truth for the current real-time tier.
- Forecast context uses the causal residual (`Observed - QDC`) plus rolling statistics at 15m, 30m, 1h, 3h, 6h, and 12h scales.
- Kp and Dst are delayed by conservative release lags (3h for Kp, 1h for Dst) so finalized global-index values cannot leak into an earlier prediction timestamp.
- Missing Kp/Dst are represented explicitly with availability and age features; missing global indices are never fabricated.
- Forecast horizons are +1h, +3h, and +6h.
- Each horizon has a gradient-boosted regression model for the correction to causal persistence plus a binary minor-storm-or-higher classifier.
- Regression uses supported HistGradientBoosting losses (`absolute_error` or `squared_error`). The old `huber` setting is accepted only as a compatibility alias for `absolute_error`.
- A validation-only blend weight combines ML and persistence. A weight of zero is a safe persistence fallback for a horizon where ML adds no value.
- The design is intentionally gray-box: deterministic QDC processing supplies the state representation and the ML model learns residual disturbance dynamics.

## Leakage-safe training protocol

For the current 1h/3h/6h horizons with a 3-hour amplitude target window, the maximum target reach is 9 hours. The training script inserts a **9-hour purge gap** between train/validation/test partitions.

The target at time `t` is the peak-to-peak residual amplitude in:

```text
[t + horizon, t + horizon + 3h)
```

so no target sample overlaps the feature timestamp.

The primary production split is approximately:

```text
65% TRAIN | 9h PURGE | 15% VALIDATION | 9h PURGE | 20% TEST
```

The test set is never used to select hyperparameters or blend weights. The final test period is additionally divided into contiguous unseen windows for a stability backtest.

## Validation-only model selection

With `--model-loss auto` (the default), the training run compares supported robust and conventional HistGradientBoosting regressors on the training/validation split only:

- absolute-error, multiple learning-rate/tree-regularization variants;
- squared-error, multiple learning-rate/tree-regularization variants.

The candidate score emphasizes the **weakest forecast horizon**, rather than allowing a strong +1h score to hide an unstable +6h result. The test period is completely untouched until the candidate and blend weights are frozen.

The production model disables estimator-internal random early stopping. This keeps the chronological validation protocol explicit and reproducible rather than introducing another randomly sampled validation subset inside the learner.

## Horizon-aware production gate

The model is deployed as a **hybrid production system**, not as an all-or-nothing three-horizon artifact.

The operational horizons are +1h and +3h. +6h remains an experimental forecast until it independently demonstrates stable improvement.

For each horizon the gate checks:

1. Validation MAE does not regress more than 2% versus persistence.
2. Validation RMSE does not regress more than 2% versus persistence.
3. Aggregate chronological test MAE beats persistence.
4. Aggregate chronological test RMSE does not regress more than 2%.
5. Mean and median walk-forward MAE improvement are positive.
6. At least 75% of unseen folds improve MAE.
7. Worst-fold MAE regression is no worse than 5%.
8. Worst-fold RMSE regression is no worse than 5% for +1h, and 10% for +3h. The +6h experimental horizon uses the stricter 5% threshold.

A production artifact is saved when **+1h passes**. The artifact records `approved_horizons_hours` and `horizon_deployment_status` so live inference can expose only horizons that actually earned production status.

This is deliberately stricter than a single train/test benchmark while avoiding the unsafe situation where a useful +1h/+3h forecaster is discarded because a materially harder +6h horizon is not yet reliable.

If +1h fails, training exits with status `3` and leaves the production artifact untouched.

`--save-candidate` can save a failed model under `models/artifacts/candidates/` for research, but the metadata remains `production_gate = failed` and the live pipeline refuses to load it.

## Train on real VIC data

For a production-quality run, use at least six months when the upstream data sources permit it:

```bash
python train_magnetometer_forecaster.py \
  --observatory VIC \
  --start-date 2024-01-01 \
  --days 180 \
  --model-loss auto
```

The script:

1. Fetches INTERMAGNET minute data and Kp/Dst concurrently.
2. Runs the existing deterministic QDC/residual pipeline.
3. Builds causal ML features and strictly future targets.
4. Performs a purged chronological train/validation/test split.
5. Selects the model family/hyperparameters using validation only.
6. Calibrates the ML/persistence blend using validation only.
7. Reports ML versus persistence MAE/RMSE and storm precision/recall/F1.
8. Refits only on train + validation using the selected configuration.
9. Evaluates the frozen model on multiple contiguous unseen test folds.
10. Applies the horizon-aware stability gate.
11. Writes `models/artifacts/<observatory>_forecaster.pkl` only when +1h has earned production status.

If Dst is unavailable from Kyoto WDC, training continues without Dst and records that limitation in model metadata. It does not fabricate Dst values.

## Live/batch inference

Once a production-approved artifact exists, the normal pipeline automatically adds:

- `forecast.horizons`: **production-approved** horizon amplitude, persistence amplitude, model delta, storm probability, tier, confidence, data quality, and blend weight.
- `forecast.experimental_horizons`: forecasts retained for analysis but not promoted to operational status.
- `forecast.model_health`: artifact schema, feature-count and deployment metadata.
- `hybrid.real_time`: current deterministic tier and source.
- `hybrid.forecasted_status`: predicted tier only for production-approved horizons.
- `hybrid.model_confidence`: confidence by approved horizon.
- `hybrid.divergence`: signed and absolute tier divergence plus an anomaly delta.
- `hybrid.divergence.significant`: true when the forecast differs by at least two activity tiers.

Set `MAGNETOMETER_FORECAST_MODEL` to override the default artifact path.

If no approved artifact exists, inference remains deterministic. ML failure can never replace or alter the current deterministic classification.

## Artifact safety

Forecast artifacts use schema version `3`. Loading requires both a fitted model and `production_gate = passed`; legacy or failed candidates are rejected by the production loader. Serialization is atomic so an interrupted write cannot replace an existing good artifact with a partial file.

## Tests

Offline tests include:

```bash
python test_magnetometer_ml.py
python -m unittest discover -s tests -p 'test_*.py'
```

They cover causal features, Kp/Dst release lags, missing-index signals, strict future targets, persistence-aware model inference, blend calibration, evaluation, production health metadata, serialization safety, hybrid integration, and horizon-aware gate policy.

Historical upstream-data regression remains opt-in:

```bash
RUN_REAL_DATA_TESTS=1 python -m unittest test_magnetometer_ml_real.py
```

It is intentionally not part of CI because INTERMAGNET, GFZ, and Kyoto WDC are external services.
