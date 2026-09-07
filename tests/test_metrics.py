from __future__ import annotations

import unittest

import pandas as pd

from btc_perp.metrics import calculate_metrics


class MetricsTests(unittest.TestCase):
    def test_profit_factor_is_undefined_for_all_winners(self) -> None:
        index = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
        equity = pd.DataFrame(
            {"equity": [100.0, 101.0, 102.0], "near_liquidation": [False, False, False]},
            index=index,
        )
        trades = pd.DataFrame(
            {
                "side": [1, -1],
                "net_pnl": [1.0, 1.0],
                "trading_cost": [0.1, 0.1],
                "funding_cost": [0.0, 0.0],
                "exit_reason": ["signal_exit", "end_of_data"],
            }
        )
        summary = calculate_metrics(equity, trades, 100.0)
        self.assertIsNone(summary["profit_factor"])

    def test_short_sample_annualized_return_is_not_reported(self) -> None:
        index = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
        equity = pd.DataFrame(
            {"equity": [100.0, 101.0, 102.0], "near_liquidation": [False, False, False]},
            index=index,
        )
        summary = calculate_metrics(equity, pd.DataFrame(), 100.0)
        self.assertIsNone(summary["annualized_return"])


if __name__ == "__main__":
    unittest.main()
