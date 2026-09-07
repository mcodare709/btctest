from __future__ import annotations

import unittest

import pandas as pd

from btc_perp.binance_history import merge_event_funding


class BinanceHistoryTests(unittest.TestCase):
    def test_funding_stays_event_only_after_merge(self) -> None:
        bars = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")})
        funding = pd.DataFrame({"timestamp": [bars.loc[1, "timestamp"]], "funding_rate": [0.001]})
        merged = merge_event_funding(bars, funding)
        self.assertEqual(int(merged["funding_rate"].notna().sum()), 1)
        self.assertEqual(float(merged.loc[1, "funding_rate"]), 0.001)
