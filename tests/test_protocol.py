from __future__ import annotations

import unittest

import pandas as pd

from btc_perp.protocol import infer_timeframe, next_open_horizon_return, scheduled_exit_index


class ProtocolTests(unittest.TestCase):
    def test_next_open_return_uses_entry_and_exit_opens(self) -> None:
        frame = pd.DataFrame(
            {"open": [100.0, 101.0, 103.0, 106.0], "close": [100.5, 102.0, 104.0, 107.0]},
            index=pd.date_range("2026-01-01", periods=4, freq="30s", tz="UTC"),
        )
        result = next_open_horizon_return(frame, horizon_bars=2)
        self.assertAlmostEqual(result.iloc[0], 106.0 / 101.0 - 1.0)
        self.assertTrue(pd.isna(result.iloc[1]))

    def test_scheduled_exit_index_matches_horizon(self) -> None:
        self.assertEqual(scheduled_exit_index(7, 10), 17)
        with self.assertRaisesRegex(ValueError, "positive"):
            scheduled_exit_index(7, 0)
    def test_infer_timeframe_rejects_irregular_bars(self) -> None:
        index = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:30Z"), pd.Timestamp("2026-01-01T00:02:00Z")])
        with self.assertRaisesRegex(ValueError, "regular"):
            infer_timeframe(pd.DataFrame({"open": [1, 1, 1]}, index=index))
