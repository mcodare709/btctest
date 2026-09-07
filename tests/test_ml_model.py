from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_perp.config import BacktestConfig
from btc_perp.ml_model import MODEL_FEATURES, make_supervised_dataset
from btc_perp.signals import cost_aware_baseline_signal


def make_market_data(rows: int = 180) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="30s", tz="UTC")
    close = 100.0 * np.exp(np.arange(rows) * 0.0004)
    open_price = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": close * 1.0008,
            "low": open_price * 0.9992,
            "close": close,
            "volume": np.linspace(8.0, 12.0, rows),
            "bid_price": close * 0.9999,
            "ask_price": close * 1.0001,
            "bid_size": np.full(rows, 20.0),
            "ask_size": np.full(rows, 10.0),
            "taker_buy_volume": np.full(rows, 7.0),
            "taker_sell_volume": np.full(rows, 3.0),
        }
    )


class MLModelTests(unittest.TestCase):
    def test_supervised_labels_are_forward_only_and_cost_banded(self) -> None:
        x, y = make_supervised_dataset(make_market_data(), horizon_bars=10)
        self.assertEqual(tuple(x.columns), MODEL_FEATURES)
        self.assertEqual(len(x), len(y))
        self.assertGreaterEqual(len(x), 100)
        self.assertTrue(set(y.unique()).issubset({0, 1, 2}))
        self.assertLess(x.index.max(), make_market_data().set_index("timestamp").index.max())

    def test_cost_aware_baseline_rejects_low_edge(self) -> None:
        row = pd.Series(
            {
                "return_1": 0.0,
                "return_2": 0.0,
                "return_10": 0.0,
                "ema_gap_pct": 0.0,
                "atr_pct": 0.002,
                "realized_vol": 0.002,
                "spread_bps": 2.0,
                "trade_imbalance": 0.0,
                "order_book_imbalance": 0.0,
            }
        )
        probability, direction, edge_bps = cost_aware_baseline_signal(row, BacktestConfig())
        self.assertAlmostEqual(probability, 0.5, places=6)
        self.assertEqual(direction, 0)
        self.assertLessEqual(edge_bps, 0.0)


if __name__ == "__main__":
    unittest.main()
