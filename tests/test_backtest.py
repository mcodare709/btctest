from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_perp.config import BacktestConfig
from btc_perp.data import resample_market_data, validate_market_data
from btc_perp.engine import run_backtest
from btc_perp.risk import size_position


def make_market_data(rows: int = 120, trend: float = 0.0004) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="30s", tz="UTC")
    close = 100.0 * np.exp(np.arange(rows) * trend)
    open_price = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": close * 1.0008,
            "low": open_price * 0.9992,
            "close": close,
            "volume": np.full(rows, 10.0),
            "funding_rate": [0.001 if index == 80 else np.nan for index in range(rows)],
            "bid_price": close * 0.9999,
            "ask_price": close * 1.0001,
            "bid_size": np.full(rows, 20.0),
            "ask_size": np.full(rows, 10.0),
            "taker_buy_volume": np.full(rows, 7.0),
            "taker_sell_volume": np.full(rows, 3.0),
        }
    )


class BacktestTests(unittest.TestCase):
    def test_validation_and_resample_keep_event_funding_sparse(self) -> None:
        raw = make_market_data(4)
        validated = validate_market_data(raw)
        resampled = resample_market_data(validated, "1min")
        self.assertEqual(len(resampled), 3)
        self.assertEqual(int(resampled["funding_rate"].notna().sum()), 0)

        raw = make_market_data(120)
        resampled = resample_market_data(validate_market_data(raw), "1min")
        self.assertEqual(int(resampled["funding_rate"].notna().sum()), 1)

    def test_forward_filled_funding_is_rejected(self) -> None:
        raw = make_market_data(5)
        raw.loc[1, "funding_rate"] = 0.001
        raw.loc[2, "funding_rate"] = 0.001
        with self.assertRaisesRegex(ValueError, "event-only"):
            validate_market_data(raw)

    def test_position_size_matches_fixed_risk_example(self) -> None:
        config = BacktestConfig(leverage=5.0, fee_rate=0.0, slippage_bps=0.0, default_spread_bps=0.0)
        size = size_position(100.0, 100.0, 0.02, config)
        self.assertAlmostEqual(size.notional, 50.0)
        self.assertAlmostEqual(size.quantity, 0.5)
        self.assertAlmostEqual(size.risk_amount, 1.0)

    def test_backtest_has_realized_costs_and_metrics(self) -> None:
        config = BacktestConfig(
            max_holding_bars=15,
            fee_rate=0.0004,
            slippage_bps=1.0,
            default_spread_bps=2.0,
        )
        result = run_backtest(make_market_data(), config=config, timeframe="30s")
        self.assertEqual(len(result.equity_curve), 120)
        self.assertIn("sharpe_ratio", result.summary)
        self.assertTrue(np.isfinite(result.equity_curve["equity"]).all())
        self.assertGreater(result.summary["trade_count"], 0)
        self.assertGreater(result.summary["trading_cost"], 0.0)
        self.assertGreater(result.summary["funding_cost"], 0.0)
        self.assertTrue((result.trades["exit_reason"] != "liquidation").all())

    def test_next_open_sizing_ignores_entry_bar_hlc(self) -> None:
        config = BacktestConfig(max_holding_bars=15)
        base = make_market_data().drop(
            columns=["bid_price", "ask_price", "bid_size", "ask_size", "taker_buy_volume", "taker_sell_volume"]
        )
        perturbed = base.copy()
        perturbed.loc[30, "high"] *= 2.0
        first = run_backtest(base, config=config, timeframe="30s").trades.iloc[0]
        second = run_backtest(perturbed, config=config, timeframe="30s").trades.iloc[0]
        self.assertEqual(first["entry_time"], second["entry_time"])
        self.assertAlmostEqual(first["quantity"], second["quantity"])

    def test_gap_beyond_liquidation_is_recorded(self) -> None:
        raw = make_market_data()
        raw.loc[35, ["open", "high", "low", "close"]] = [50.0, 51.0, 30.0, 50.0]
        config = BacktestConfig(max_holding_bars=100, leverage=100.0, maintenance_margin_rate=0.1)
        result = run_backtest(raw, config=config, timeframe="30s")
        self.assertEqual(result.summary["liquidation_count"], 1)
        self.assertTrue(result.summary["near_liquidation"])
        liquidation = result.trades[result.trades["exit_reason"] == "liquidation"]
        self.assertEqual(len(liquidation), 1)
        self.assertGreater(liquidation.iloc[0]["liquidation_fee"], 0.0)

    def test_flat_market_does_not_trade(self) -> None:
        result = run_backtest(make_market_data(trend=0.0), timeframe="30s")
        self.assertEqual(result.summary["trade_count"], 0)
        self.assertAlmostEqual(result.summary["final_equity"], 100.0)


if __name__ == "__main__":
    unittest.main()
