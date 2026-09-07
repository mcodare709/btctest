from __future__ import annotations

import unittest

from btc_perp.config import BacktestConfig
from btc_perp.oos import run_parameter_sensitivity, run_purged_oos_backtest
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
    def test_sensitivity_uses_oos_reports_for_every_variant(self) -> None:
        frame = make_market_data(rows=180)
        result = run_parameter_sensitivity(
            frame,
            config=BacktestConfig(),
            parameter_sets={
                "lower_edge": {"min_expected_edge_bps": 1.0},
                "higher_edge": {"min_expected_edge_bps": 3.0},
            },
            train_bars=60,
            test_bars=30,
            purge_bars=10,
        )
        self.assertEqual(set(result), {"lower_edge", "higher_edge"})
        self.assertGreater(result["lower_edge"]["fold_count"], 0)
        self.assertIn("oos_report", result["higher_edge"])
