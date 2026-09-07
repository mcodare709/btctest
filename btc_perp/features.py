"""Leakage-aware feature engineering for short-horizon BTC perpetual data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features using current and past rows only.

    The backtest executes a signal on the next bar's open. NaNs from warm-up
    windows remain present and are handled by the strategy as no-trade rows.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns for features: {sorted(missing)}")

    result = frame.copy()
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["volume"].astype(float)

    result["return_1"] = close.pct_change()
    result["return_2"] = close.pct_change(2)
    result["return_10"] = close.pct_change(10)
    result["ema_fast"] = _ema(close, 10)
    result["ema_slow"] = _ema(close, 30)
    result["ema_gap_pct"] = result["ema_fast"] / result["ema_slow"] - 1.0

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["atr_pct"] = true_range.rolling(14, min_periods=14).mean() / close
    result["realized_vol"] = result["return_1"].rolling(20, min_periods=20).std() * np.sqrt(20)
    volume_mean = volume.rolling(30, min_periods=30).mean()
    volume_std = volume.rolling(30, min_periods=30).std()
    result["volume_z"] = (volume - volume_mean) / volume_std.replace(0, np.nan)

    if {"bid_size", "ask_size"}.issubset(result.columns):
        denominator = (result["bid_size"] + result["ask_size"]).replace(0, np.nan)
        result["order_book_imbalance"] = (result["bid_size"] - result["ask_size"]) / denominator
    else:
        result["order_book_imbalance"] = 0.0

    if {"taker_buy_volume", "taker_sell_volume"}.issubset(result.columns):
        denominator = (result["taker_buy_volume"] + result["taker_sell_volume"]).replace(0, np.nan)
        result["trade_imbalance"] = (result["taker_buy_volume"] - result["taker_sell_volume"]) / denominator
        result["cvd"] = (result["taker_buy_volume"] - result["taker_sell_volume"]).cumsum()
        result["cvd_change"] = result["cvd"].diff()
    else:
        result["trade_imbalance"] = 0.0
        result["cvd"] = 0.0
        result["cvd_change"] = 0.0

    if {"bid_price", "ask_price", "close"}.issubset(result.columns):
        result["spread_bps"] = (result["ask_price"] - result["bid_price"]) / close * 10_000
    else:
        result["spread_bps"] = np.nan

    if "open_interest" in result.columns:
        result["oi_change"] = result["open_interest"].pct_change()
    else:
        result["oi_change"] = 0.0

    if "liquidation_volume" in result.columns:
        result["liquidation_ratio"] = result["liquidation_volume"] / volume.replace(0, np.nan)
    else:
        result["liquidation_ratio"] = 0.0

    if "long_short_ratio" in result.columns:
        result["long_short_log_ratio"] = np.log(result["long_short_ratio"].clip(lower=1e-12))
    else:
        result["long_short_log_ratio"] = 0.0

    if "funding_rate" not in result.columns:
        result["funding_rate"] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)
