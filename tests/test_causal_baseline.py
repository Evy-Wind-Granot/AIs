from __future__ import annotations

import numpy as np

from magnetometer.causal_baseline import compute_causal_qdc_baseline


def test_future_disturbance_cannot_change_past_baseline() -> None:
    rng = np.random.default_rng(7)
    base = 100.0 + 2.0 * np.sin(np.arange(6000) * 2.0 * np.pi / 1440.0)
    noisy = base + rng.normal(0.0, 0.2, base.size)
    changed = noisy.copy()
    changed[5000:5100] += 500.0

    baseline_a, residual_a = compute_causal_qdc_baseline(noisy, 60.0)
    baseline_b, residual_b = compute_causal_qdc_baseline(changed, 60.0)

    np.testing.assert_allclose(baseline_a[:4900], baseline_b[:4900], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(residual_a[:4900], residual_b[:4900], rtol=0.0, atol=1e-10)


def test_invalid_samples_do_not_produce_valid_residuals() -> None:
    x = np.ones(3000, dtype=float) * 100.0
    x[1500:1510] = np.nan

    baseline, residual = compute_causal_qdc_baseline(x, 60.0)

    assert baseline.shape == x.shape
    assert residual.shape == x.shape
    assert np.all(np.isnan(residual[1500:1510]))
