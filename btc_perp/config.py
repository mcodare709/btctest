"""Configuration objects for reproducible backtests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    """Execution, cost, and risk assumptions.

    ``funding_rate`` is interpreted as a funding rate per funding event at the
    bar open. The data validator requires event-only values, so an 8-hour rate
    cannot silently be charged on every 30-second bar after forward-fill.
    """

    initial_equity: float = 100.0
    risk_per_trade: float = 0.01
    leverage: float = 5.0
    maintenance_margin_rate: float = 0.005
    stop_atr_multiplier: float = 2.0
    minimum_stop_pct: float = 0.002
    max_holding_bars: int = 20
    update_every_bars: int = 1
    long_probability_threshold: float = 0.65
    short_probability_threshold: float = 0.35
    cost_aware_filter: bool = True
    min_expected_edge_bps: float = 2.0
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    default_spread_bps: float = 2.0
    liquidation_fee_rate: float = 0.001
    annualization_days: float = 365.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not 0 < self.risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1)")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if not 0 <= self.maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate must be in [0, 1)")
        if self.stop_atr_multiplier <= 0 or self.minimum_stop_pct <= 0:
            raise ValueError("stop parameters must be positive")
        if self.max_holding_bars <= 0 or self.update_every_bars <= 0:
            raise ValueError("bar counts must be positive")
        if not 0 < self.short_probability_threshold < self.long_probability_threshold < 1:
            raise ValueError("probability thresholds must satisfy 0 < short < long < 1")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        if self.fee_rate < 0 or self.slippage_bps < 0 or self.default_spread_bps < 0 or self.liquidation_fee_rate < 0:
            raise ValueError("cost parameters cannot be negative")


@dataclass(frozen=True)
class ExperimentConfig:
    """One time-scale experiment in the requested comparison."""

    name: str
    timeframe: str
    horizon_bars: int
    update_every_bars: int = 1


DEFAULT_EXPERIMENTS = (
    ExperimentConfig("30s_5m", "30s", horizon_bars=10),
    ExperimentConfig("1m_15m", "1min", horizon_bars=15),
    ExperimentConfig("5m_1h", "5min", horizon_bars=12),
)
