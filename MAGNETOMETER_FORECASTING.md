# Short-Horizon Geomagnetic Forecasting

The production magnetometer pipeline has an optional ML forecasting layer on top of the deterministic harmonic/QDC residual pipeline.

## Production design

- Deterministic QDC/baseline processing remains the source of truth for the current real-time tier.
- Forecast context uses the causal residual (`Observed - QDC`) plus rolling statistics at 15m, 30m, 1h, 3h, 6h, and 12h scales.
- Kp and Dst are delayed by conservative release lags (3h for Kp, 1h for Dst) so finalized global-index values cannot leak into an earlier prediction timestamp.
- Forecast horizons are +1h, +3h, and +6h. The model learns from weeks-to-months of historical data rather than attempting an unreliable weeks-ahead local forecast.
- Each horizon has a gradient-boosted regression model for strictly future residual amplitude and a binary model for minor-storm-or-higher probability.
- The design is intentionally gray-box: deterministic physics/QDC processing supplies the state representation and the ML model learns the residual disturbance dynamics.
- Production evaluation is chronological, target-aware, and uses a persistence baseline. The final test period is never used for model selection.

This follows current space-weather forecasting practice: operational systems distinguish nowcasts from forecasts, probabilistic forecasts are increasingly emphasized, and hybrid physics/ML (gray-box) systems are a recognized approach. NOAA's operational geomagnetic products also provide deterministic and probabilistic forecasts over short horizons. 

## Leakage-safe training protocol

For the current 1h/3h/6h horizons with a 3-hour amplitude target window, the maximum target reach is 9 hours. The training script therefore inserts a **9-hour purge gap** between train/validation and validation/test periods. This prevents future target windows from crossing a partition boundary.

The target at time `t` is the peak-to-peak residual amplitude in:

```text
[t + horizon, t + horizon + 3h)
```

so no target sample overlaps the feature timestamp.

The production split is approximately:

```text
65% TRAIN | 9h PURGE | 15% VALIDATION | 9h PURGE | 20% TEST
```

This is deliberately different from a random train/test split.

## Production gate

A model is **not** published as a production artifact unless it beats the persistence baseline on every configured forecast horizon in the untouched chronological test set.

If the gate fails, training exits with status `3` and leaves the existing artifact untouched.

`--allow-nonbeating` exists only for research/candidate experiments. The live pipeline rejects artifacts whose metadata does not contain:

```text
production_gate = passed
```

Therefore an accidentally retained experimental model cannot silently become the live forecaster.

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
5. Reports ML versus persistence MAE/RMSE and storm precision/recall/F1.
6. Refits only on train + validation after the model-selection stage.
7. Applies the untouched test-set production gate.
8. Writes `models/artifacts/<observatory>_forecaster.pkl` only after the gate passes.

If Dst is unavailable from Kyoto WDC, training continues without Dst and records that limitation in the model metadata. It does not fabricate Dst values.

Model artifacts are intentionally ignored by Git.

## Live/batch inference

Once a production-approved artifact exists, the normal pipeline automatically adds:

- `forecast.horizons`: +1h/+3h/+6h amplitude, probability, tier, and confidence score.
- `hybrid.real_time`: current deterministic tier.
- `hybrid.forecasted_status`: predicted tier by horizon.
- `hybrid.model_confidence`: aggregate forecast confidence score.
- `hybrid.divergence`: signed and absolute tier divergence.
- `hybrid.divergence.significant`: true when the forecast differs by at least two activity tiers.

Set `MAGNETOMETER_FORECAST_MODEL` to override the default artifact path.

If no approved artifact exists, inference remains deterministic. ML failure can never replace or alter the current deterministic classification.

## Interpretation

The local residual-amplitude tiers are intentionally kept separate from NOAA's planetary G-scale. NOAA defines geomagnetic storm levels from Kp, with G1 beginning at Kp=5 and higher G levels at Kp=6–9. The local VIC residual forecast is therefore an observatory-specific disturbance forecast, not a claim that a local 100 nT residual is equivalent to a particular global G-scale level.

For operational use, treat the ML probability as a risk signal and validate its calibration on additional historical events before using it for automated protective actions. The deterministic classifier remains the current-state authority.

## Tests

The normal test suite is offline:

```bash
python test_magnetometer_ml.py
```

It includes causal-feature tests, Kp/Dst release-lag tests, strict future-target leakage tests, model inference, evaluation, and serialization tests.

There is also an opt-in historical upstream-data regression test:

```bash
RUN_REAL_DATA_TESTS=1 python -m unittest test_magnetometer_ml_real.py
```

It is intentionally not part of CI because INTERMAGNET, GFZ, and Kyoto WDC are external services.
