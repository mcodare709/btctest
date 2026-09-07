"""Position sizing under fixed-risk and leverage constraints."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BacktestConfig


@dataclass(frozen=True)
class PositionSize:
    notional: float
    quantity: float
    stop_distance_pct: float
    risk_amount: float


def size_position(
    equity: float,
    entry_price: float,
    stop_distance_pct: float,
    config: BacktestConfig,
    estimated_spread_bps: float | None = None,
) -> PositionSize:
    """Size notional using frozen stop distance plus estimated round-trip cost.

    The stop distance must be calculated at signal time and passed unchanged at
    execution time. Costs are an estimate; gap-through-stop losses can still
    exceed the budget.
    """

    if equity <= 0 or entry_price <= 0:
        return PositionSize(0.0, 0.0, config.minimum_stop_pct, 0.0)
    stop_distance_pct = max(config.minimum_stop_pct, stop_distance_pct)
    spread_bps = config.default_spread_bps if estimated_spread_bps is None else max(estimated_spread_bps, 0.0)
    round_trip_cost_rate = (
        2.0 * config.fee_rate
        + 2.0 * config.slippage_bps / 10_000.0
        + spread_bps / 10_000.0
    )
    risk_rate = stop_distance_pct + round_trip_cost_rate
    risk_amount = equity * config.risk_per_trade
    risk_notional = risk_amount / risk_rate
    leverage_cap = equity * config.leverage
    notional = min(risk_notional, leverage_cap)
    return PositionSize(
        notional=notional,
        quantity=notional / entry_price,
        stop_distance_pct=stop_distance_pct,
        risk_amount=notional * risk_rate,
    )
