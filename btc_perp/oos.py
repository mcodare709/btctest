"""OOS-only walk-forward backtests and simple deterministic benchmarks."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import validate_market_data
from .engine import run_backtest
from .evaluation import WalkForwardFold, purged_walk_forward_splits
from .metrics import calculate_metrics
from .signals import cost_aware_baseline_signal


def benchmark_returns(frame: pd.DataFrame, seed: int = 42) -> dict[str, float]:
    """Return OOS asset, random-sign, and one-bar momentum benchmark returns.

    These are deliberately simple return benchmarks, not claims of executable
    profitability; strategy comparisons must use the cost-aware engine result.
    """

    close = frame["close"].astype(float)
    returns = close.pct_change().dropna()
    if returns.empty:
        return {"buy_and_hold": 0.0, "random_sign": 0.0, "simple_momentum": 0.0}
    rng = np.random.default_rng(seed)
    random_sign = rng.choice(np.array([-1.0, 1.0]), size=len(returns))
    momentum_sign = np.sign(returns.shift(1).fillna(0.0)).to_numpy()
    return {
        "buy_and_hold": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "random_sign": float(np.prod(1.0 + returns.to_numpy() * random_sign) - 1.0),
        "simple_momentum": float(np.prod(1.0 + returns.to_numpy() * momentum_sign) - 1.0),
    }


def run_purged_oos_backtest(
    market_data: pd.DataFrame,
    *,
    config: BacktestConfig,
    train_bars: int,
    test_bars: int,
    purge_bars: int,
    warmup_bars: int = 40,
) -> dict[str, Any]:
    """Run baseline only on each fold's OOS region and report no in-sample PnL."""

    if not isinstance(market_data.index, pd.DatetimeIndex):
        market_data = validate_market_data(market_data)
    folds = purged_walk_forward_splits(
        len(market_data), train_bars=train_bars, test_bars=test_bars, purge_bars=purge_bars
    )
    if not folds:
        raise ValueError("not enough rows for one purged walk-forward fold")
    reports: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        reports.append(_run_fold(market_data, fold, fold_index, config, warmup_bars))
    return {"folds": reports, "fold_count": len(reports)}


def _run_fold(
    market_data: pd.DataFrame,
    fold: WalkForwardFold,
    fold_index: int,
    config: BacktestConfig,
    warmup_bars: int,
) -> dict[str, Any]:
    context_start = max(0, fold.test_start - warmup_bars)
    frame = market_data.iloc[context_start : fold.test_end].copy()
    oos_start = market_data.index[fold.test_start]

    def oos_signal(row: pd.Series, signal_config: BacktestConfig) -> tuple[float, int]:
        if row.name < oos_start:
            return 0.5, 0
        probability, direction, _ = cost_aware_baseline_signal(row, signal_config)
        return probability, direction

    result = run_backtest(frame, config=config, signal_fn=oos_signal)
    equity = result.equity_curve.loc[result.equity_curve.index >= oos_start]
    trades = result.trades.loc[result.trades["entry_time"] >= oos_start].copy() if not result.trades.empty else result.trades
    summary = calculate_metrics(equity, trades, config.initial_equity, config.annualization_days)
    return {
        "fold": fold_index,
        "split": asdict(fold),
        "oos_start": oos_start.isoformat(),
        "oos_end": market_data.index[fold.test_end - 1].isoformat(),
        "summary": summary,
        "benchmarks": benchmark_returns(market_data.iloc[fold.test_start : fold.test_end], seed=42 + fold_index),
    }

def run_parameter_sensitivity(
    market_data: pd.DataFrame,
    *,
    config: BacktestConfig,
    parameter_sets: Mapping[str, Mapping[str, Any]],
    train_bars: int,
    test_bars: int,
    purge_bars: int,
    warmup_bars: int = 40,
) -> dict[str, dict[str, Any]]:
    """Evaluate named BacktestConfig variants using the same purged OOS folds.

    This does not select a winner. It exposes whether the observed OOS result
    remains similar across nearby, pre-declared assumptions.
    """

    allowed_fields = set(BacktestConfig.__dataclass_fields__)
    results: dict[str, dict[str, Any]] = {}
    for name, overrides in parameter_sets.items():
        unknown = sorted(set(overrides) - allowed_fields)
        if unknown:
            raise ValueError(f"{name}: unknown BacktestConfig fields: {unknown}")
        report = run_purged_oos_backtest(
            market_data,
            config=replace(config, **dict(overrides)),
            train_bars=train_bars,
            test_bars=test_bars,
            purge_bars=purge_bars,
            warmup_bars=warmup_bars,
        )
        summaries = [fold["summary"] for fold in report["folds"]]
        total_returns = np.array([float(summary["total_return"]) for summary in summaries])
        drawdowns = np.array([float(summary["max_drawdown"]) for summary in summaries])
        results[name] = {
            "overrides": dict(overrides),
            "fold_count": report["fold_count"],
            "mean_fold_return": float(total_returns.mean()),
            "median_fold_return": float(np.median(total_returns)),
            "profitable_fold_fraction": float((total_returns > 0).mean()),
            "mean_fold_max_drawdown": float(drawdowns.mean()),
            "oos_report": report,
        }
    return results
