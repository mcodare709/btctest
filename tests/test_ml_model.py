from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from btc_perp.config import BacktestConfig
from btc_perp.features import build_features
from btc_perp.ml_model import EXECUTION_PROTOCOL_VERSION, MODEL_FEATURES, CatBoostBundle, make_supervised_dataset
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
    def test_absent_optional_feeds_remain_missing_and_unavailable(self) -> None:
        frame = make_market_data().drop(
            columns=["bid_price", "ask_price", "bid_size", "ask_size", "taker_buy_volume", "taker_sell_volume"]
        )
        features = build_features(frame)
        self.assertTrue(features["order_book_imbalance"].isna().all())
        self.assertTrue(features["funding_rate"].isna().all())
        self.assertEqual(features["has_orderbook"].iloc[0], 0)
        self.assertEqual(features["has_funding"].iloc[0], 0)

        frame["funding_rate"] = np.nan
        frame.loc[1, "funding_rate"] = 0.0001
        partial = build_features(frame)
        self.assertEqual(partial["has_funding"].iloc[0], 0)
        self.assertEqual(partial["has_funding"].iloc[1], 1)

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

    def test_model_load_rejects_timeframe_mismatch(self) -> None:
        class FakeClassifier:
            def load_model(self, path: str) -> None:
                self.loaded_path = path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            path.write_text("placeholder", encoding="utf-8")
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
                        "timeframe": "30s",
                        "horizon_bars": 10,
                        "feature_names": list(MODEL_FEATURES),
                    }
                ),
                encoding="utf-8",
            )
            with patch("btc_perp.ml_model._catboost", return_value=FakeClassifier):
                with self.assertRaisesRegex(ValueError, "timeframe"):
                    CatBoostBundle.load(path, expected_timeframe="5min", expected_horizon_bars=10)

if __name__ == "__main__":
    unittest.main()
