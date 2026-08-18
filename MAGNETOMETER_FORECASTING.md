# Short-Horizon Geomagnetic Forecasting

The production magnetometer pipeline now has an optional ML forecasting layer
on top of the deterministic harmonic/QDC residual pipeline.

## Design

- Deterministic QDC/baseline processing remains the source of truth for the
  current real-time tier.
- Forecast context uses the residual (`Observed - QDC`) plus causal rolling
  statistics at 15m, 30m, 1h, 3h, 6h, and 12h scales.
- Kp and Dst are included only after conservative release delays (3h for Kp,
  1h for Dst) so finalized global-index values cannot leak into an earlier
  prediction timestamp.
- Forecast horizons are +1h, +3h, and +6h.
- Each horizon has a gradient-boosted regression model for future residual
  amplitude and a binary model for minor-storm-or-higher probability.
- Historical evaluation is chronological rather than randomly shuffled.

The intended training history is weeks to months of real data. The default
training command uses 90 days, while the operational forecast horizon stays
short.

## Train on real VIC data

```bash
python train_magnetometer_forecaster.py \
  --observatory VIC \
  --start-date 2024-05-08 \
  --days 90
```

The script:

1. Fetches INTERMAGNET minute data and Kp/Dst concurrently.
2. Runs the existing deterministic QDC/residual pipeline.
3. Builds causal ML features and strictly future targets.
4. Holds out the newest 20% chronologically.
5. Reports RMSE, MAE, precision, recall, and F1 for +1h/+3h/+6h.
6. Fits a final model on the complete historical window.
7. Writes `models/artifacts/<observatory>_forecaster.pkl`.

Model artifacts are intentionally ignored by Git.

## Live/batch inference

Once the artifact exists, the normal pipeline automatically adds:

- `forecast.horizons`: +1h/+3h/+6h amplitude, probability, tier, and confidence.
- `hybrid.real_time`: current deterministic tier.
- `hybrid.forecasted_status`: predicted tier by horizon.
- `hybrid.model_confidence`: mean forecast confidence.
- `hybrid.divergence`: signed and absolute tier divergence.
- `hybrid.divergence.significant`: true when the forecast differs by at least
  two activity tiers.

Set `MAGNETOMETER_FORECAST_MODEL` to override the default artifact path.

If no trained artifact exists, inference remains deterministic and the result
reports `forecast.status = model_not_trained`; ML failure never changes the
current deterministic classification.

## Tests

The normal test suite is offline:

```bash
python test_magnetometer_ml.py
```

There is also an opt-in historical upstream-data regression test using a fixed
VIC period:

```bash
RUN_REAL_DATA_TESTS=1 python -m unittest test_magnetometer_ml_real.py
```

It is intentionally not part of CI because INTERMAGNET, GFZ, and Kyoto WDC are
external services. The test trains on the older 80% of the real window and
evaluates on the newest 20%, matching the production training methodology.
