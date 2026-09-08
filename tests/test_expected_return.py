from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_perp.expected_return import ExpectedReturnBundle, expected_direction, make_expected_return_dataset
from btc_perp.feature_schema import CORE_FEATURES


def make_data(rows: int = 180) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="30s", tz="UTC")
    close = 100 * np.exp(np.arange(rows) * 0.0002)
    return pd.DataFrame({"timestamp": timestamp, "open": np.r_[close[0], close[:-1]], "high": close * 1.001, "low": close * 0.999, "close": close, "volume": np.full(rows, 10.0)})


class ExpectedReturnTests(unittest.TestCase):
    def test_policy_keeps_flat_when_both_edges_are_below_threshold(self) -> None:
        self.assertEqual(expected_direction(0.0001, 0.0002, 0.0005), 0)
        self.assertEqual(expected_direction(0.001, 0.0002, 0.0005), 1)
        self.assertEqual(expected_direction(0.0002, 0.001, 0.0005), -1)

    def test_targets_are_cost_aware_and_features_keep_missing_values(self) -> None:
        x, targets = make_expected_return_dataset(make_data(), horizon_bars=10)
        self.assertEqual(tuple(x.columns), CORE_FEATURES)
        self.assertEqual(len(x), len(targets))
        self.assertIn("long_net_return", targets)
        self.assertIn("short_net_return", targets)
        self.assertTrue(x["trade_imbalance"].isna().all())

    def test_inference_preserves_nan(self) -> None:
        class FakeModel:
            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                self.frame = frame
                return np.array([0.001])

        model = FakeModel()
        bundle = ExpectedReturnBundle(model, "long", ("return_1", "funding_rate"), {})
        self.assertAlmostEqual(bundle.predict(pd.Series({"return_1": 0.01, "funding_rate": np.nan})), 0.001)
        self.assertTrue(pd.isna(model.frame.loc[0, "funding_rate"]))


if __name__ == "__main__":
    unittest.main()
