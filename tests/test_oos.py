from __future__ import annotations

import unittest

from btc_perp.config import BacktestConfig
from btc_perp.oos import run_purged_oos_backtest
from tests.test_backtest import make_market_data


class OOSBacktestTests(unittest.TestCase):
    def test_reports_only_oos_folds(self) -> None:
        frame = make_market_data(rows=180)
        report = run_purged_oos_backtest(
            frame, config=BacktestConfig(), train_bars=60, test_bars=30, purge_bars=10
        )
        self.assertGreater(report["fold_count"], 0)
        for fold in report["folds"]:
            self.assertIn("summary", fold)
            self.assertIn("buy_and_hold", fold["benchmarks"])
            self.assertEqual(fold["split"]["purge_end"], fold["split"]["test_start"])
