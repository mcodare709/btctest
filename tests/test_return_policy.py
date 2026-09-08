import unittest

import numpy as np

from btc_perp.return_policy import expected_net_returns, non_overlapping_actions
from scripts.walk_forward_gross_return import _side_metrics


class ReturnPolicyTests(unittest.TestCase):
    def test_expected_net_returns_are_derived_from_one_gross_forecast(self) -> None:
        result = expected_net_returns(np.array([0.002, -0.003]), 0.0012)
        np.testing.assert_allclose(result, [[0.0008, -0.0032], [-0.0042, 0.0018]])

    def test_policy_keeps_flat_below_net_threshold(self) -> None:
        expected = expected_net_returns(np.array([0.0013]), 0.0012)
        actions = non_overlapping_actions(expected, threshold=0.0005, horizon_bars=10)
        np.testing.assert_array_equal(actions, [0])

    def test_policy_prevents_overlapping_horizon_trades(self) -> None:
        expected = np.tile([[0.001, -0.002]], (13, 1))
        actions = non_overlapping_actions(expected, threshold=0.0005, horizon_bars=10)
        expected_actions = np.zeros(13, dtype=np.int8)
        expected_actions[[0, 11]] = 1
        np.testing.assert_array_equal(actions, expected_actions)

    def test_invalid_cost_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            expected_net_returns(np.array([0.0]), -0.001)

    def test_strategy_metrics_use_compounded_equity_and_drawdown(self) -> None:
        metrics = _side_metrics(np.array([0.10, -0.10]))
        self.assertAlmostEqual(metrics["compounded_net_return"], -0.01)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.10)


if __name__ == "__main__":
    unittest.main()
