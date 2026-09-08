"""Canonical bar-time semantics shared by research and execution code.

At row ``t`` the feature vector is observed only after that bar closes. A
signal therefore enters at ``open[t + 1]``. Holding ``N`` bars exits at
``open[t + 1 + N]`` before that bar's intrabar path is evaluated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EXECUTION_PROTOCOL_VERSION = "next-open-v1"


def scheduled_exit_index(entry_index: int, horizon_bars: int) -> int:
    """Return the bar index whose open executes a fixed-horizon time exit."""

    if entry_index < 0:
        raise ValueError("entry_index must be non-negative")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    return entry_index + horizon_bars


def next_open_horizon_return(frame: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """Return the executable gross return for a close-time signal.

    Values are indexed by signal bar. The return is from next-bar open to the
    open after ``horizon_bars`` elapsed holding bars; the final rows are NaN.
    """

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if "open" not in frame:
        raise ValueError("market data requires open for next-open execution")
    entry_open = frame["open"].shift(-1)
    exit_open = frame["open"].shift(-(horizon_bars + 1))
    return exit_open / entry_open - 1.0


def infer_timeframe(frame: pd.DataFrame) -> str:
    """Return the regular UTC bar interval used for a model artifact."""

    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 2:
        raise ValueError("at least two timestamp-indexed rows are required to infer timeframe")
    deltas = frame.index.to_series().diff().dropna()
    interval = deltas.mode().iloc[0]
    if (deltas != interval).any():
        raise ValueError("model training data must have a regular bar timeframe")
    seconds = int(interval.total_seconds())
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}min"

def net_horizon_returns(
    frame: pd.DataFrame,
    horizon_bars: int,
    *,
    fee_rate: float,
    slippage_bps: float,
    default_spread_bps: float,
) -> pd.DataFrame:
    """Return executable long/short net returns under the bar backtest cost model.

    Signal t enters at open[t+1] and exits at open[t+1+horizon]. Funding is
    charged only for known event rows strictly after entry through exit open,
    matching the engine's event ordering.  Stop/liquidation path labels remain
    a future extension requiring intrabar event data.
    """

    gross_long = next_open_horizon_return(frame, horizon_bars)
    entry_open = frame["open"].shift(-1)
    exit_open = frame["open"].shift(-(horizon_bars + 1))
    if {"bid_price", "ask_price"}.issubset(frame.columns):
        spread = (frame["ask_price"] - frame["bid_price"]) / frame["close"] * 10_000.0
    else:
        spread = pd.Series(default_spread_bps, index=frame.index, dtype=float)
    entry_spread = spread.shift(-1).fillna(default_spread_bps)
    exit_spread = spread.shift(-(horizon_bars + 1)).fillna(default_spread_bps)
    round_trip_cost = 2.0 * (fee_rate + slippage_bps / 10_000.0) + (entry_spread + exit_spread) / 20_000.0
    funding = pd.Series(0.0, index=frame.index)
    if "funding_rate" in frame.columns:
        rates = frame["funding_rate"].fillna(0.0).astype(float)
        for offset in range(2, horizon_bars + 2):
            funding += rates.shift(-offset).fillna(0.0)
    return pd.DataFrame(
        {
            "long_net_return": gross_long - round_trip_cost - funding,
            "short_net_return": -gross_long - round_trip_cost + funding,
            "gross_long_return": gross_long,
            "round_trip_cost": round_trip_cost,
            "funding_long": funding,
            "entry_open": entry_open,
            "exit_open": exit_open,
        },
        index=frame.index,
    ).replace([np.inf, -np.inf], np.nan)
