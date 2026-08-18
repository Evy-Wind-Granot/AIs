import unittest

import numpy as np

from magnetometer.core import _set_deterministic_hybrid


class HybridIntegrationTests(unittest.TestCase):
    def test_deterministic_fallback_has_stable_hybrid_schema(self):
        result = {"flags": np.asarray(["quiet", "active"], dtype=object)}
        _set_deterministic_hybrid(result)
        self.assertEqual(result["hybrid"]["real_time"]["tier"], "active")
        self.assertEqual(result["hybrid"]["real_time"]["source"], "deterministic_qdc")
        self.assertEqual(result["hybrid"]["forecasted_status"], {})
        self.assertFalse(result["hybrid"]["divergence"]["significant"])
        self.assertEqual(result["hybrid"]["divergence"]["anomaly_delta"], 0)

    def test_empty_flags_fall_back_to_unknown(self):
        result = {"flags": []}
        _set_deterministic_hybrid(result)
        self.assertEqual(result["hybrid"]["real_time"]["tier"], "unknown")
        self.assertEqual(result["hybrid"]["divergence"]["direction"], "none")


if __name__ == "__main__":
    unittest.main()
