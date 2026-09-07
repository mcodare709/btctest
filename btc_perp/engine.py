"""Event-driven perpetual-futures backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import resample_market_data, validate_market_data
from .features import build_features
from .metrics import calculate_metrics
from .risk import size_position
from .signals import cost_aware_baseline_signal


@dataclass
class _Position:
    side: int
    quantity: float
    entry_ref_price: float
    entry_fill_price: float
    entry_time: pd.Timestamp
    entry_index: int
    stop_price: float
    entry_fee: float
    entry_slippage: float
    entry_spread: float
    funding_cost: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, float | int | bool | None]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


TRADE_COLUMNS = [
    "side",
    "entry_time",
    "exit_time",
    "entry_ref_price",
    "entry_fill_price",
    "exit_ref_price",
    "exit_fill_price",
    "quantity",
    "bars_held",
    "gross_pnl",
    "fee",
    "slippage_cost",
    "spread_cost",
    "liquidation_fee",
    "trading_cost",
    "funding_cost",
    "net_pnl",
    "exit_reason",
]


def _numeric(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _execution_costs(
    reference_price: float,
    quantity: float,
    side: int,
    row: pd.Series,
    config: BacktestConfig,
    is_entry: bool,
) -> tuple[float, float, float, float, float]:
    """Return fill price, fee, slippage, spread, and total trading cost."""

    spread_bps = max(_numeric(row, "spread_bps", config.default_spread_bps), 0.0)
    spread_rate = spread_bps / 20_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    adverse_sign = side if is_entry else -side
    fill_price = reference_price * (1.0 + adverse_sign * (slippage_rate + spread_rate))
    slippage_cost = quantity * reference_price * slippage_rate
    spread_cost = quantity * reference_price * spread_rate
    fee = quantity * fill_price * config.fee_rate
    return fill_price, fee, slippage_cost, spread_cost, fee + slippage_cost + spread_cost


def _unrealized(position: _Position | None, close_price: float) -> float:
    if position is None:
        return 0.0
    return position.quantity * (close_price - position.entry_ref_price) * position.side


def run_backtest(
    market_data: pd.DataFrame,
    config: BacktestConfig | None = None,
    timeframe: str | None = None,
    signal_fn: Callable[[pd.Series, BacktestConfig], tuple[float, int]] | None = None,
) -> BacktestResult:
    """Run a cost-aware, next-bar-open backtest on market data."""

    config = config or BacktestConfig()
    frame = validate_market_data(market_data.reset_index() if isinstance(market_data.index, pd.DatetimeIndex) else market_data)
    if timeframe:
        frame = resample_market_data(frame, timeframe)
    features = build_features(frame)
    if features.empty:
        raise ValueError("market_data contains no rows")

    cash = config.initial_equity
    position: _Position | None = None
    pending: dict[str, Any] | None = None
    trade_records: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []

    def close_position(
        current_position: _Position,
        reference_price: float,
        row: pd.Series,
        timestamp: pd.Timestamp,
        index: int,
        reason: str,
    ) -> float:
        nonlocal cash
        fill, fee, slippage, spread, _ = _execution_costs(
            reference_price,
            current_position.quantity,
            current_position.side,
            row,
            config,
            is_entry=False,
        )
        gross_pnl = current_position.quantity * (reference_price - current_position.entry_ref_price) * current_position.side
        liquidation_fee = current_position.quantity * reference_price * config.liquidation_fee_rate if reason == "liquidation" else 0.0
        cash += gross_pnl - fee - slippage - spread - liquidation_fee
        if reason == "liquidation":
            # A perpetual account cannot owe more than the simulated balance.
            cash = max(cash, 0.0)
        entry_cost = current_position.entry_fee + current_position.entry_slippage + current_position.entry_spread
        trading_cost = entry_cost + fee + slippage + spread + liquidation_fee
        net_pnl = gross_pnl - trading_cost - current_position.funding_cost
        trade_records.append(
            {
                "side": current_position.side,
                "entry_time": current_position.entry_time,
                "exit_time": timestamp,
                "entry_ref_price": current_position.entry_ref_price,
                "entry_fill_price": current_position.entry_fill_price,
                "exit_ref_price": reference_price,
                "exit_fill_price": fill,
                "quantity": current_position.quantity,
                "bars_held": index - current_position.entry_index,
                "gross_pnl": gross_pnl,
                "fee": current_position.entry_fee + fee,
                "slippage_cost": current_position.entry_slippage + slippage,
                "spread_cost": current_position.entry_spread + spread,
                "liquidation_fee": liquidation_fee,
                "trading_cost": trading_cost,
                "funding_cost": current_position.funding_cost,
                "net_pnl": net_pnl,
                "exit_reason": reason,
            }
        )
        return fill

    def open_position(side: int, reference_price: float, row: pd.Series, timestamp: pd.Timestamp, index: int, stop_distance_pct: float) -> _Position | None:
        nonlocal cash
        spread_bps = _numeric(row, "spread_bps", config.default_spread_bps)
        sizing = size_position(cash, reference_price, stop_distance_pct, config, estimated_spread_bps=spread_bps)
        if sizing.quantity <= 0:
            return None
        fill, fee, slippage, spread, _ = _execution_costs(reference_price, sizing.quantity, side, row, config, is_entry=True)
        cash -= fee + slippage + spread
        stop_distance = max(stop_distance_pct, config.minimum_stop_pct)
        stop_price = reference_price * (1.0 - side * stop_distance)
        return _Position(
            side=side,
            quantity=sizing.quantity,
            entry_ref_price=reference_price,
            entry_fill_price=fill,
            entry_time=timestamp,
            entry_index=index,
            stop_price=stop_price,
            entry_fee=fee,
            entry_slippage=slippage,
            entry_spread=spread,
        )

    for index, (timestamp, row) in enumerate(features.iterrows()):
        open_price = _numeric(row, "open")
        close_price = _numeric(row, "close")

        # Funding events are defined at the bar open. Only an existing position
        # is charged, and the known open/mark is used instead of future close.
        if position is not None and not pd.isna(row.get("funding_rate", np.nan)):
            funding_payment = position.quantity * open_price * _numeric(row, "funding_rate") * position.side
            cash -= funding_payment
            position.funding_cost += funding_payment

        # Time-based exit is known before this bar's open.
        if position is not None and index - position.entry_index >= config.max_holding_bars:
            close_position(position, open_price, row, timestamp, index, "time_exit")
            position = None

        liquidated_this_bar = False
        near_liquidation = False
        if position is not None:
            open_equity = cash + _unrealized(position, open_price)
            open_margin = abs(position.quantity * open_price) * config.maintenance_margin_rate
            if open_equity <= open_margin or open_equity <= 0:
                close_position(position, open_price, row, timestamp, index, "liquidation")
                position = None
                pending = None
                liquidated_this_bar = True
                near_liquidation = True

        # Execute the signal generated on the previous close at this open.
        if pending is not None and not liquidated_this_bar:
            target = int(pending["direction"])
            stop_distance_pct = float(pending["stop_distance_pct"])
            if position is not None and target != position.side:
                close_position(position, open_price, row, timestamp, index, "signal_exit" if target == 0 else "signal_reversal")
                position = None
            if position is None and target != 0:
                position = open_position(target, open_price, row, timestamp, index, stop_distance_pct)
            pending = None

        # Intrabar liquidation is checked at the adverse extreme before stop.
        # OHLC cannot reveal the exact path, so liquidation wins ties.
        if position is not None:
            adverse_price = _numeric(row, "low") if position.side == 1 else _numeric(row, "high")
            adverse_equity = cash + _unrealized(position, adverse_price)
            adverse_margin = abs(position.quantity * adverse_price) * config.maintenance_margin_rate
            liquidation_hit = adverse_equity <= adverse_margin or adverse_equity <= 0
            stop_hit = (position.side == 1 and _numeric(row, "low") <= position.stop_price) or (
                position.side == -1 and _numeric(row, "high") >= position.stop_price
            )
            if liquidation_hit:
                close_position(position, adverse_price, row, timestamp, index, "liquidation")
                position = None
                near_liquidation = True
            elif stop_hit:
                if position.side == 1:
                    stop_reference = min(open_price, position.stop_price)
                else:
                    stop_reference = max(open_price, position.stop_price)
                close_position(position, stop_reference, row, timestamp, index, "stop_loss")
                position = None

        equity_before_liquidation = cash + _unrealized(position, close_price)
        margin = abs(position.quantity * close_price) * config.maintenance_margin_rate if position is not None else 0.0
        near_liquidation = near_liquidation or (position is not None and equity_before_liquidation <= margin * 1.25)
        if position is not None and (equity_before_liquidation <= margin or equity_before_liquidation <= 0):
            close_position(position, close_price, row, timestamp, index, "liquidation")
            position = None
            equity_before_liquidation = cash
            near_liquidation = True

        # Force close at the final observed price so summary equity is realized.
        if index == len(features) - 1 and position is not None:
            close_position(position, close_price, row, timestamp, index, "end_of_data")
            position = None
            equity_before_liquidation = cash

        equity_records.append(
            {
                "timestamp": timestamp,
                "equity": cash + _unrealized(position, close_price),
                "cash": cash,
                "close": close_price,
                "position_side": 0 if position is None else position.side,
                "position_notional": 0.0 if position is None else position.quantity * close_price,
                "near_liquidation": bool(near_liquidation),
            }
        )

        # Signals are computed at this close and cannot execute until next bar.
        if index % config.update_every_bars == 0 and index < len(features) - 1:
            if signal_fn is not None:
                probability, direction = signal_fn(row, config)
            else:
                probability, direction, _ = cost_aware_baseline_signal(row, config)
            stop_distance_pct = max(
                config.minimum_stop_pct,
                config.stop_atr_multiplier * _numeric(row, "atr_pct", config.minimum_stop_pct),
            )
            pending = {"probability": probability, "direction": direction, "stop_distance_pct": stop_distance_pct}

    equity_curve = pd.DataFrame(equity_records).set_index("timestamp")
    trades = pd.DataFrame(trade_records, columns=TRADE_COLUMNS)
    summary = calculate_metrics(equity_curve, trades, config.initial_equity, config.annualization_days)
    return BacktestResult(summary=summary, equity_curve=equity_curve, trades=trades)
