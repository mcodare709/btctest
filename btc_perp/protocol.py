"""Canonical bar-time semantics shared by research and execution code.

At row ``t`` the feature vector is observed only after that bar closes. A
signal therefore enters at ``open[t + 1]``. Holding ``N`` bars exits at
``open[t + 1 + N]`` before that bar's intrabar path is evaluated.
"""

from __future__ import annotations

import pandas as pd


EXECUTION_PROTOCOL_VERSION = "next-open-v1"


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
