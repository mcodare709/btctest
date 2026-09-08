"""Backtest paired expected-return models with the canonical execution engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.config import BacktestConfig
from btc_perp.data import load_market_csv
from btc_perp.engine import run_backtest
from btc_perp.expected_return_runtime import ExpectedReturnModels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model-prefix", required=True, type=Path)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--horizon-bars", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    models = ExpectedReturnModels.load(args.model_prefix, expected_timeframe=args.timeframe, expected_horizon_bars=args.horizon_bars)
    result = run_backtest(
        load_market_csv(args.data),
        config=BacktestConfig(max_holding_bars=args.horizon_bars),
        timeframe=args.timeframe,
        signal_fn=models.signal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(args.output_dir / "equity_curve.csv")
    result.trades.to_csv(args.output_dir / "trades.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
