"""Cost-aware expected-return policy helpers."""

from __future__ import annotations

import numpy as np


def expected_net_returns(
    gross_prediction: np.ndarray,
    round_trip_cost: float | np.ndarray,
) -> np.ndarray:
    """Derive long/short expected net returns from one gross-return forecast."""

    gross = np.asarray(gross_prediction, dtype=float).reshape(-1)
    cost = np.broadcast_to(np.asarray(round_trip_cost, dtype=float), gross.shape)
    if not np.isfinite(gross).all() or not np.isfinite(cost).all():
        raise ValueError("predictions and costs must be finite")
    if (cost < 0).any():
        raise ValueError("round-trip costs cannot be negative")
    return np.column_stack((gross - cost, -gross - cost))


def non_overlapping_actions(
    expected_net: np.ndarray,
    *,
    threshold: float,
    horizon_bars: int,
) -> np.ndarray:
    """Select LONG=1/SHORT=-1/FLAT=0 with at most one open position."""

    values = np.asarray(expected_net, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("expected_net must have shape (rows, 2)")
    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    actions = np.zeros(len(values), dtype=np.int8)
    next_allowed = 0
    for index, (long_expected, short_expected) in enumerate(values):
        if index < next_allowed:
            continue
        if long_expected > threshold and long_expected > short_expected:
            actions[index] = 1
        elif short_expected > threshold and short_expected > long_expected:
            actions[index] = -1
        if actions[index] != 0:
            next_allowed = index + horizon_bars + 1
    return actions
