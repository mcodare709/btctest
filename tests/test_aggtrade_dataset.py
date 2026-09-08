from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from btc_perp.aggtrade_dataset import build_aggtrade_30s_dataset


class AggTradeDatasetTests(unittest.TestCase):
    def test_aggregates_raw_events_to_real_30_second_bars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "archive"
            source = root / "aggTrades"
            source.mkdir(parents=True)
            pd.DataFrame(
                {
                    "agg_trade_id": [1, 2, 3],
                    "price": [100.0, 101.0, 102.0],
                    "quantity": [2.0, 3.0, 4.0],
                    "first_trade_id": [1, 2, 3],
                    "last_trade_id": [1, 2, 3],
                    "transact_time": [0, 10_000, 35_000],
                    "is_buyer_maker": [False, True, False],
                }
            ).to_csv(source / "BTCUSDT-aggTrades-2026-01.zip", index=False, compression="zip")

            metadata = build_aggtrade_30s_dataset(root, Path(directory) / "out")
            bars = pd.read_csv(Path(directory) / "out" / "btc_usdt_perp_30s_aggtrades.csv.gz")

            self.assertEqual(metadata["rows"], 2)
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars.loc[0, "open"], 100.0)
            self.assertEqual(bars.loc[0, "close"], 101.0)
            self.assertEqual(bars.loc[0, "trade_count"], 2)
            self.assertEqual(bars.loc[0, "taker_buy_volume"], 2.0)


if __name__ == "__main__":
    unittest.main()
