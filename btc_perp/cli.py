"""Command line entry points for the BTC perpetual research baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BacktestConfig, DEFAULT_EXPERIMENTS
from .data import load_market_csv, resample_market_data
from .engine import run_backtest
from .ml_model import CatBoostBundle, train_catboost
from .oos import run_catboost_purged_oos_evaluation, run_purged_oos_backtest


def _write_result(result, output_dir: Path, name: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    summary = dict(result.summary)
    if name:
        summary["experiment"] = name
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest BTCUSDT perpetual baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--data", required=True, type=Path)
    backtest.add_argument("--timeframe", default="30s")
    backtest.add_argument("--horizon-bars", type=int, default=10)
    backtest.add_argument("--output-dir", required=True, type=Path)
    backtest.add_argument("--model", type=Path, help="optional trained CatBoost .cbm artifact")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--data", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)

    oos = subparsers.add_parser("evaluate-oos")
    oos.add_argument("--data", required=True, type=Path)
    oos.add_argument("--output-dir", required=True, type=Path)
    oos.add_argument("--timeframe", default="30s")
    oos.add_argument("--train-bars", required=True, type=int)
    oos.add_argument("--test-bars", required=True, type=int)
    oos.add_argument("--purge-bars", required=True, type=int)
    oos.add_argument("--warmup-bars", default=40, type=int)
    oos.add_argument("--horizon-bars", default=10, type=int)

    model_oos = subparsers.add_parser("evaluate-oos-model")
    model_oos.add_argument("--data", required=True, type=Path)
    model_oos.add_argument("--output-dir", required=True, type=Path)
    model_oos.add_argument("--timeframe", default="30s")
    model_oos.add_argument("--train-bars", required=True, type=int)
    model_oos.add_argument("--test-bars", required=True, type=int)
    model_oos.add_argument("--purge-bars", required=True, type=int)
    model_oos.add_argument("--horizon-bars", default=10, type=int)
    model_oos.add_argument("--iterations", default=400, type=int)
    train = subparsers.add_parser("train-model")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--horizon-bars", type=int, default=10)
    train.add_argument("--timeframe", default="30s")
    train.add_argument("--data-source", default="unknown")
    train.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    market_data = load_market_csv(args.data)

    if args.command == "train-model":
        print(json.dumps(train_catboost(market_data, args.output, horizon_bars=args.horizon_bars, timeframe=args.timeframe, data_source=args.data_source), indent=2))
        return

    if args.command == "backtest":
        config = BacktestConfig(max_holding_bars=args.horizon_bars)
        bundle = CatBoostBundle.load(args.model) if args.model else None
        result = run_backtest(
            market_data,
            config=config,
            timeframe=args.timeframe,
            signal_fn=bundle.signal if bundle else None,
        )
        _write_result(result, args.output_dir)
        return

    if args.command == "evaluate-oos-model":
        evaluation_data = resample_market_data(market_data, args.timeframe)
        report = run_catboost_purged_oos_evaluation(
            evaluation_data,
            horizon_bars=args.horizon_bars,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            purge_bars=args.purge_bars,
            iterations=args.iterations,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "catboost_oos_report.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote CatBoost OOS-only report: {output}")
        return
    if args.command == "evaluate-oos":
        config = BacktestConfig(max_holding_bars=args.horizon_bars)
        evaluation_data = resample_market_data(market_data, args.timeframe)
        report = run_purged_oos_backtest(
            evaluation_data,
            config=config,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            purge_bars=args.purge_bars,
            warmup_bars=args.warmup_bars,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "oos_report.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Wrote OOS-only report: {output}")
        return

    comparison = []
    for experiment in DEFAULT_EXPERIMENTS:
        config = BacktestConfig(
            max_holding_bars=experiment.horizon_bars,
            update_every_bars=experiment.update_every_bars,
        )
        result = run_backtest(market_data, config=config, timeframe=experiment.timeframe)
        experiment_dir = args.output_dir / experiment.name
        _write_result(result, experiment_dir, experiment.name)
        comparison.append(dict(result.summary, experiment=experiment.name, timeframe=experiment.timeframe))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
