"""Interpretable baseline signal."""

from __future__ import annotations

import math

import pandas as pd

from .config import BacktestConfig


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _feature_value(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value is None or pd.isna(value):
        return default
    return float(value)


def baseline_signal(row: pd.Series, config: BacktestConfig) -> tuple[float, int]:
    """Return heuristic ``(prob_up, direction)`` from current-row features.

    This is a screening baseline, not a calibrated probability model. It
    combines momentum, EMA regime, trade imbalance, and book imbalance. Missing
    warm-up features result in no position.
    """

    required = ("return_1", "return_2", "return_10", "ema_gap_pct", "atr_pct")
    if any(pd.isna(row.get(column)) for column in required):
        return 0.5, 0

    score = 0.0
    score += 3.0 * float(row["return_1"]) / max(float(row["atr_pct"]), 1e-8)
    score += 2.0 * float(row["return_2"]) / max(float(row["atr_pct"]), 1e-8)
    score += 1.0 * float(row["return_10"]) / max(float(row["atr_pct"]), 1e-8)
    score += 2.0 * float(row["ema_gap_pct"]) / max(float(row["atr_pct"]), 1e-8)
    score += 1.0 * _feature_value(row, "trade_imbalance")
    score += 0.5 * _feature_value(row, "order_book_imbalance")
    probability = _sigmoid(score)

    if probability > config.long_probability_threshold:
        return probability, 1
    if probability < config.short_probability_threshold:
        return probability, -1
    return probability, 0


def cost_aware_baseline_signal(row: pd.Series, config: BacktestConfig) -> tuple[float, int, float]:
    """Return ``(prob_up, direction, expected_edge_bps)`` after cost filtering.

    The baseline probability is not calibrated. This deliberately conservative
    gate estimates gross edge from confidence times current volatility, then
    subtracts a round-trip fee, slippage, and spread estimate. It is a safety
    bridge until a real walk-forward CatBoost model is trained on market data.
    """

    probability, raw_direction = baseline_signal(row, config)
    if not config.cost_aware_filter or raw_direction == 0:
        return probability, raw_direction, 0.0

    spread_bps = max(_feature_value(row, "spread_bps", config.default_spread_bps), 0.0)
    round_trip_cost_bps = 2.0 * (
        config.fee_rate * 10_000.0 + config.slippage_bps
    ) + spread_bps
    volatility = max(
        _feature_value(row, "realized_vol"),
        _feature_value(row, "atr_pct"),
        config.minimum_stop_pct,
    )
    confidence = abs(2.0 * probability - 1.0)
    gross_edge_bps = confidence * volatility * 10_000.0
    expected_edge_bps = gross_edge_bps - round_trip_cost_bps
    direction = raw_direction if expected_edge_bps >= config.min_expected_edge_bps else 0
    return probability, direction, expected_edge_bps
