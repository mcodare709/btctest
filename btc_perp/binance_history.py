"""Public Binance USD-M Futures history ingestion with explicit feed limits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com"


def _get(path: str, params: dict[str, Any]) -> list[Any]:
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    """Fetch recent USD-M futures OHLCV plus taker-buy base volume."""
    rows = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote", "ignore"])
    frame = frame[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]]
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["taker_sell_volume"] = frame["volume"] - frame["taker_buy_volume"]
    return frame


def fetch_funding(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    rows = _get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return pd.DataFrame({"timestamp": pd.to_datetime(frame["fundingTime"], unit="ms", utc=True), "funding_rate": pd.to_numeric(frame["fundingRate"], errors="coerce")})


def merge_event_funding(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Attach funding only to the first known bar at/after each event."""
    result = bars.sort_values("timestamp").copy()
    result["funding_rate"] = pd.NA
    for event in funding.itertuples(index=False):
        eligible = result.index[result["timestamp"] >= event.timestamp]
        if len(eligible):
            result.loc[eligible[0], "funding_rate"] = event.funding_rate
    return result


def save_recent_dataset(path: str | Path, symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1000) -> Path:
    bars = fetch_klines(symbol, interval, limit)
    data = merge_event_funding(bars, fetch_funding(symbol))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return output
