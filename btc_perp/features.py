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
        result["order_book_imbalance"] = np.nan

    if {"taker_buy_volume", "taker_sell_volume"}.issubset(result.columns):
        denominator = (result["taker_buy_volume"] + result["taker_sell_volume"]).replace(0, np.nan)
        result["trade_imbalance"] = (result["taker_buy_volume"] - result["taker_sell_volume"]) / denominator
        result["cvd"] = (result["taker_buy_volume"] - result["taker_sell_volume"]).cumsum()
        result["cvd_change"] = result["cvd"].diff()
    else:
        result["trade_imbalance"] = np.nan
        result["cvd"] = np.nan
        result["cvd_change"] = np.nan

    if {"bid_price", "ask_price", "close"}.issubset(result.columns):
        result["spread_bps"] = (result["ask_price"] - result["bid_price"]) / close * 10_000
    else:
        result["spread_bps"] = np.nan

    if "open_interest" in result.columns:
        result["oi_change"] = result["open_interest"].pct_change()
    else:
        result["oi_change"] = np.nan

    if "liquidation_volume" in result.columns:
        result["liquidation_ratio"] = result["liquidation_volume"] / volume.replace(0, np.nan)
    else:
        result["liquidation_ratio"] = np.nan

    if "long_short_ratio" in result.columns:
        result["long_short_log_ratio"] = np.log(result["long_short_ratio"].clip(lower=1e-12))
    else:
        result["long_short_log_ratio"] = np.nan

    if "funding_rate" not in result.columns:
        result["funding_rate"] = np.nan
    # Per-row indicators distinguish an absent observation from a real zero.
    result["has_orderbook"] = (
        result[["bid_price", "ask_price", "bid_size", "ask_size"]].notna().all(axis=1).astype(int)
        if {"bid_price", "ask_price", "bid_size", "ask_size"}.issubset(result.columns)
        else 0
    )
    result["has_open_interest"] = result["open_interest"].notna().astype(int) if "open_interest" in result.columns else 0
    result["has_liquidation"] = result["liquidation_volume"].notna().astype(int) if "liquidation_volume" in result.columns else 0
    result["has_long_short_ratio"] = result["long_short_ratio"].notna().astype(int) if "long_short_ratio" in result.columns else 0
    result["has_funding"] = result["funding_rate"].notna().astype(int)
    result["latest_known_funding_rate"] = result["funding_rate"].ffill()
    result["funding_change"] = result["latest_known_funding_rate"].diff()
    result["funding_zscore"] = (result["latest_known_funding_rate"] - result["latest_known_funding_rate"].rolling(100, min_periods=20).mean()) / result["latest_known_funding_rate"].rolling(100, min_periods=20).std().replace(0, np.nan)
    result["mark_index_basis_bps"] = ((result["mark_price"] / result["index_price"] - 1.0) * 10_000.0 if {"mark_price", "index_price"}.issubset(result.columns) else np.nan)
    result["premium_index"] = result["premium_index"] if "premium_index" in result.columns else result["mark_index_basis_bps"]
    result["predicted_funding_rate"] = result["predicted_funding_rate"] if "predicted_funding_rate" in result.columns else np.nan
    if "open_interest" in result.columns:
        result["oi_zscore"] = (result["open_interest"] - result["open_interest"].rolling(100, min_periods=20).mean()) / result["open_interest"].rolling(100, min_periods=20).std().replace(0, np.nan)
        result["oi_acceleration"] = result["oi_change"].diff()
        result["price_oi_interaction"] = result["return_1"] * result["oi_change"]
    else:
        result[["oi_zscore", "oi_acceleration", "price_oi_interaction"]] = np.nan
    if {"bid_price", "ask_price", "bid_size", "ask_size"}.issubset(result.columns):
        result["microprice"] = (result["ask_price"] * result["bid_size"] + result["bid_price"] * result["ask_size"]) / (result["bid_size"] + result["ask_size"]).replace(0, np.nan)
        result["weighted_mid_bps"] = (result["microprice"] / result["close"] - 1.0) * 10_000.0
        result["spread_change_bps"] = result["spread_bps"].diff()
        result["order_book_imbalance_change"] = result["order_book_imbalance"].diff()
    else:
        result[["microprice", "weighted_mid_bps", "spread_change_bps", "order_book_imbalance_change"]] = np.nan
    for depth in (5, 10):
        bid, ask = f"bid_depth_{depth}", f"ask_depth_{depth}"
        result[f"depth_imbalance_{depth}"] = (result[bid] - result[ask]) / (result[bid] + result[ask]).replace(0, np.nan) if {bid, ask}.issubset(result.columns) else np.nan
    result["trade_imbalance_mean_10"] = result["trade_imbalance"].rolling(10, min_periods=5).mean()
    result["trade_imbalance_acceleration"] = result["trade_imbalance"].diff()
    result["cvd_rolling_change"] = result["cvd"].diff(10)
    result["taker_buy_ratio"] = result["taker_buy_volume"] / volume.replace(0, np.nan) if "taker_buy_volume" in result.columns else np.nan
    result["taker_sell_ratio"] = result["taker_sell_volume"] / volume.replace(0, np.nan) if "taker_sell_volume" in result.columns else np.nan
    return result.replace([np.inf, -np.inf], np.nan)
