"""Market-data loading, validation, and timeframe aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import infer_timeframe

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS = (
    "funding_rate",
    "funding_event",
    "open_interest",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "taker_buy_volume",
    "taker_sell_volume",
    "liquidation_volume",
    "long_short_ratio",
)


def _parse_timestamp(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps or Unix seconds/milliseconds into UTC."""

    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        unit = "ms" if numeric.dropna().abs().median() > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return clean, sorted market data or raise a useful validation error."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    result = frame.copy()
    result["timestamp"] = _parse_timestamp(result["timestamp"])
    if result["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values")

    numeric_columns = [column for column in result.columns if column != "timestamp"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if "funding_rate" in result.columns:
        funding_present = result["funding_rate"].notna()
        if "funding_event" in result.columns:
            event_flag = result["funding_event"].fillna(0).astype(bool)
            if (funding_present & ~event_flag).any():
                raise ValueError("funding_rate is present on a row where funding_event is false")
        consecutive_events = funding_present & funding_present.shift(1, fill_value=False)
        if consecutive_events.any():
            raise ValueError("funding_rate must be event-only; consecutive non-null rows look forward-filled")

    if result[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise ValueError("required OHLCV columns contain NaN or non-numeric values")
    if (result["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high is below an OHLC value")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low is above an OHLC value")

    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    result = result.set_index("timestamp")
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    return result


def load_market_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV market-data file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return validate_market_data(pd.read_csv(source))


def resample_market_data(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Downsample regular bars only, with left-closed/start-labeled buckets.

    Upsampling invents price paths and is prohibited. A 1m source can produce
    5m bars, but never a synthetic 30s experiment.
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = validate_market_data(frame.reset_index())
    if not timeframe:
        return frame.copy()
    source_timeframe = infer_timeframe(frame)
    source_seconds = int(pd.Timedelta(source_timeframe).total_seconds())
    target_seconds = int(pd.Timedelta(timeframe).total_seconds())
    if target_seconds < source_seconds:
        raise ValueError(f"upsampling from {source_timeframe} to {timeframe} is prohibited")
    if target_seconds == source_seconds:
        return frame.copy()
    if target_seconds % source_seconds:
        raise ValueError(f"target timeframe {timeframe!r} must be an integer multiple of {source_timeframe!r}")

    agg: dict[str, str] = {
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }
    for column in ("quote_volume", "trade_count", "taker_buy_volume", "taker_sell_volume", "taker_buy_quote", "taker_sell_quote", "liquidation_volume"):
        if column in frame:
            agg[column] = "sum"
    for column in ("open_interest", "long_short_ratio", "bid_price", "ask_price", "bid_size", "ask_size", "mark_price", "index_price", "premium_index"):
        if column in frame:
            agg[column] = "last"
    if "funding_rate" in frame:
        agg["funding_rate"] = "last"
    if "funding_event" in frame:
        agg["funding_event"] = "max"

    result = frame.resample(timeframe, label="left", closed="left").agg(agg).dropna(subset=["open", "high", "low", "close"])
    if result.empty:
        raise ValueError(f"resampling to {timeframe!r} produced no rows")
    return result

def available_columns(frame: pd.DataFrame) -> list[str]:
    """Return optional data columns that are actually present."""

    return [column for column in OPTIONAL_COLUMNS if column in frame.columns]
