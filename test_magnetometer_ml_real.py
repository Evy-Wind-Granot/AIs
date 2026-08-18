"""Opt-in regression test against real INTERMAGNET/GFZ/Kyoto data.

The normal CI suite stays offline. Run this test explicitly with
``RUN_REAL_DATA_TESTS=1`` when network access to the upstream services is
available.  It uses a fixed historical VIC window so results are reproducible
rather than depending on the current space-weather conditions.
"""
from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import magnetometer_demo as md
from magnetometer.acquisition import fetch_dst_kyoto, fetch_intermagnet_iaga2002, fetch_kp_gfz
from magnetometer.parsing import parse_iaga2002_to_dataframe
from models.forecaster import ForecastConfig, GeomagneticForecaster, build_training_data


@unittest.skipUnless(
    os.environ.get("RUN_REAL_DATA_TESTS") == "1",
    "set RUN_REAL_DATA_TESTS=1 to run historical upstream-data regression",
)
class RealDataForecastTests(unittest.TestCase):
    def test_vic_historical_forecast_pipeline(self) -> None:
        start = pd.Timestamp("2024-05-08", tz="UTC")
        days = 21
        warmup = 3
        fetch_start = (start - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")
        end = (start + pd.Timedelta(days=days)).strftime("%Y-%m-%d")

        with ThreadPoolExecutor(max_workers=3) as pool:
            mag_future = pool.submit(
                fetch_intermagnet_iaga2002, "VIC", fetch_start, days + warmup
            )
            kp_future = pool.submit(fetch_kp_gfz, fetch_start, end)
            mag = parse_iaga2002_to_dataframe(mag_future.result())
            self.assertIsNotNone(mag)
            months = sorted({(ts.year, ts.month) for ts in mag.index})
            dst_parts = [pool.submit(fetch_dst_kyoto, y, m) for y, m in months]
            dst = pd.concat([f.result() for f in dst_parts if f.result() is not None]).sort_index()
            kp = kp_future.result()

        self.assertGreater(len(mag), days * 1440 * 0.8)
        result = md.run_analysis(
            mag["x_nt"].to_numpy(),
            60.0,
            label="VIC real-data ML regression",
            start_time=mag.index.min().to_pydatetime(),
            analysis_start_time=start.to_pydatetime(),
            dst_series=dst,
            kp_series=kp,
            observatory="VIC",
        )
        self.assertEqual(result["status"], "ok")

        index = pd.date_range(start, periods=len(result["residual"]), freq="min", tz="UTC")
        residual = pd.Series(np.asarray(result["residual"], dtype=float), index=index)
        cfg = ForecastConfig(max_iter=100, min_samples_leaf=20)
        features, targets = build_training_data(residual, kp.reindex(index, method="ffill"), dst.reindex(index, method="ffill"), config=cfg)

        split = int(len(features) * 0.8)
        model = GeomagneticForecaster(cfg).fit(
            features.iloc[:split], {h: y.iloc[:split] for h, y in targets.items()}
        )
        metrics = model.evaluate(
            features.iloc[split:], {h: y.iloc[split:] for h, y in targets.items()}
        )
        for evaluation in metrics.values():
            self.assertGreater(evaluation.n_samples, 100)
            self.assertTrue(np.isfinite(evaluation.rmse_nt))
            self.assertTrue(np.isfinite(evaluation.mae_nt))


if __name__ == "__main__":
    unittest.main()
