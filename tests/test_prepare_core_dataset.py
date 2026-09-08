from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_core_dataset import prepare_core_dataset


class PrepareCoreDatasetTests(unittest.TestCase):
    def test_prepare_keeps_nan_and_does_not_add_optional_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv.gz"
            output = root / "core.csv.gz"
            frame = pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC"),
                "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0], "volume": [0.0, np.nan, 2.0],
                "return_1": [0.0, np.nan, 0.1], "return_2": [np.nan, np.nan, 0.1],
                "return_10": [np.nan, np.nan, np.nan], "ema_gap_pct": [0.0, 0.0, 0.1],
                "atr_pct": [0.1, 0.1, 0.1], "realized_vol": [0.1, 0.1, 0.1], "volume_z": [0.0, np.nan, 1.0],
                "trade_imbalance": [np.nan, np.nan, np.nan], "cvd_change": [np.nan, np.nan, np.nan],
                "trade_imbalance_mean_10": [np.nan, np.nan, np.nan], "trade_imbalance_acceleration": [np.nan, np.nan, np.nan],
                "cvd_rolling_change": [np.nan, np.nan, np.nan], "taker_buy_ratio": [np.nan, np.nan, np.nan],
                "taker_sell_ratio": [np.nan, np.nan, np.nan], "mark_index_basis_bps": [0.0, 0.0, 0.0],
                "premium_index": [0.0, 0.0, 0.0], "open_interest": [1.0, 1.0, 1.0],
            })
            frame.to_csv(source, index=False, compression="gzip")
            metadata = prepare_core_dataset(source, output, chunksize=2)
            result = pd.read_csv(output)
            self.assertEqual(metadata["rows"], 3)
            self.assertIn("return_1", result.columns)
            self.assertNotIn("open_interest", result.columns)
            self.assertTrue(pd.isna(result.loc[1, "volume"]))
            self.assertEqual(metadata["missing_value_policy"], "preserve_nan")


if __name__ == "__main__":
    unittest.main()
