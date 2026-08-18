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

With `--model-loss auto` (the default), the training run compares supported robust and conventional HistGradientBoosting regressors on the training/validation split only.

The candidate score emphasizes the **weakest forecast horizon**, rather than allowing a strong +1h score to hide an unstable +6h result. The test period is completely untouched until the candidate and blend weights are frozen.

## Horizon-aware production gate

The model is deployed as a **hybrid production system**, not as an all-or-nothing three-horizon artifact.

The operational horizons are +1h and +3h. +6h remains an experimental forecast until it independently demonstrates stable improvement.

A production artifact is saved when **+1h passes**. The artifact records `approved_horizons_hours` and `horizon_deployment_status` so live inference can expose only horizons that actually earned production status.

If +1h fails, training exits with status `3` and leaves the production artifact untouched.

`--save-candidate` can save a failed model under `magnetometer/models/artifacts/candidates/` for research, but the metadata remains `production_gate = failed` and the live pipeline refuses to load it.

## Train on real VIC data

```bash
python magnetometer/models/train_magnetometer_forecaster.py \
  --observatory VIC \
  --start-date 2024-01-01 \
  --days 180 \
  --model-loss auto
```

The script fetches INTERMAGNET minute data and Kp/Dst concurrently, runs the deterministic pipeline, builds causal features, performs the purged chronological split, freezes the selected model/blend on validation, evaluates unseen test folds, and applies the production gate.

## Live/batch inference

Once a production-approved artifact exists, the normal pipeline automatically exposes approved forecast horizons plus experimental horizons separately. Set `MAGNETOMETER_FORECAST_MODEL` to override the default artifact path.

## Artifact safety

Forecast artifacts use schema version `3`. Loading requires both a fitted model and `production_gate = passed`; legacy or failed candidates are rejected by the production loader. Serialization is atomic so an interrupted write cannot replace an existing good artifact with a partial file.

## Tests

Offline model tests live in `magnetometer/models/tests/unit/` and `magnetometer/models/tests/integration/`. General pipeline tests live in `magnetometer/tests/`.

```bash
python -m unittest discover -s magnetometer/models/tests -p 'test_*.py'
python -m unittest discover -s magnetometer/tests -p 'test_*.py'
```

Historical upstream-data regression remains opt-in:

```bash
RUN_REAL_DATA_TESTS=1 python -m unittest magnetometer/models/tests/integration/test_magnetometer_ml_real.py
```

It is intentionally not part of CI because INTERMAGNET, GFZ, and Kyoto WDC are external services.
