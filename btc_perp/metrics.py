"""Performance metrics with explicit handling for sparse/short samples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float,
    annualization_days: float = 365.0,
) -> dict[str, float | int | bool | None]:
    """Calculate requested account, risk, trade, and cost metrics."""

    if equity_curve.empty:
        return {
            "initial_equity": initial_equity,
            "final_equity": initial_equity,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "trading_cost": 0.0,
            "funding_cost": 0.0,
            "liquidation_count": 0,
            "near_liquidation": False,
        }

    equity = equity_curve["equity"].astype(float)
    final_equity = float(equity.iloc[-1])
    total_return = _safe_ratio(final_equity, initial_equity) - 1.0
    elapsed_days = max((equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400, 1 / 1440)
    if elapsed_days < 1.0:
        annualized_return: float | None = None
    elif final_equity <= 0:
        annualized_return = -1.0
    else:
        log_return = math.log(final_equity / initial_equity) * annualization_days / elapsed_days
        annualized_return = math.exp(log_return) - 1.0 if log_return < 700 else None

    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    periods_per_year = max(len(returns) / elapsed_days * annualization_days, 1.0)
    return_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = _safe_ratio(float(returns.mean()) * math.sqrt(periods_per_year), return_std)
    sortino = _safe_ratio(float(returns.mean()) * math.sqrt(periods_per_year), downside_std)

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    if trades.empty:
        trade_count = 0
        win_rate = 0.0
        profit_factor = 0.0
        long_pnl = short_pnl = trading_cost = funding_cost = 0.0
        liquidation_count = 0
    else:
        pnl = trades["net_pnl"].astype(float)
        trade_count = len(trades)
        win_rate = float((pnl > 0).mean())
        gains = float(pnl[pnl > 0].sum())
        losses = float(-pnl[pnl < 0].sum())
        profit_factor = None if gains > 0 and losses == 0 else _safe_ratio(gains, losses)
        long_pnl = float(trades.loc[trades["side"] == 1, "net_pnl"].sum())
        short_pnl = float(trades.loc[trades["side"] == -1, "net_pnl"].sum())
        trading_cost = float(trades["trading_cost"].sum())
        funding_cost = float(trades["funding_cost"].sum())
        liquidation_count = int((trades["exit_reason"] == "liquidation").sum())

    return {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "annualized_return": None if annualized_return is None else float(annualized_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": trade_count,
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "trading_cost": trading_cost,
        "funding_cost": funding_cost,
        "liquidation_count": liquidation_count,
        "near_liquidation": bool(equity_curve["near_liquidation"].any()) if "near_liquidation" in equity_curve else False,
    }
