"""Build genuine 30-second BTCUSDT bars from Binance aggTrade events."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

AGGTRADE_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
)


def _read_aggtrades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="zip")
    aliases = {"transact_time": "timestamp", "T": "timestamp", "a": "agg_trade_id", "p": "price", "q": "quantity", "f": "first_trade_id", "l": "last_trade_id", "m": "is_buyer_maker"}
    frame = frame.rename(columns=aliases)
    if not set(AGGTRADE_COLUMNS).issubset(frame.columns):
        frame = pd.read_csv(path, compression="zip", header=None, names=AGGTRADE_COLUMNS)
    frame = frame.loc[:, list(AGGTRADE_COLUMNS)].copy()
    for name in ("price", "quantity", "timestamp"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["price", "quantity", "timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["is_buyer_maker"] = frame["is_buyer_maker"].astype(str).str.lower().eq("true")
    return frame.sort_values("timestamp")


def build_aggtrade_30s_dataset(archive_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Aggregate raw events directly; never upsample 1m bars to 30s."""

    root = Path(archive_root)
    files = sorted((root / "aggTrades").glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"no aggTrade archives found in {root / 'aggTrades'}")
    parts: list[pd.DataFrame] = []
    for path in files:
        trades = _read_aggtrades(path)
        trades["bucket"] = trades["timestamp"].dt.floor("30s")
        trades["quote"] = trades["price"] * trades["quantity"]
        trades["taker_buy_volume"] = trades["quantity"].where(~trades["is_buyer_maker"], 0.0)
        trades["taker_buy_quote"] = trades["quote"].where(~trades["is_buyer_maker"], 0.0)
        parts.append(
            trades.groupby("bucket", sort=True).agg(
                open=("price", "first"),
                high=("price", "max"),
                low=("price", "min"),
                close=("price", "last"),
                volume=("quantity", "sum"),
                quote_volume=("quote", "sum"),
                trade_count=("agg_trade_id", "count"),
                taker_buy_volume=("taker_buy_volume", "sum"),
                taker_buy_quote=("taker_buy_quote", "sum"),
            )
        )
    bars = pd.concat(parts).groupby(level=0, sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"), trade_count=("trade_count", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"), taker_buy_quote=("taker_buy_quote", "sum"),
    )
    bars["taker_sell_volume"] = bars["volume"] - bars["taker_buy_volume"]
    bars["taker_sell_quote"] = bars["quote_volume"] - bars["taker_buy_quote"]
    bars.index.name = "timestamp"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "btc_usdt_perp_30s_aggtrades.csv.gz"
    bars.reset_index().to_csv(data_path, index=False, compression="gzip")
    metadata = {
        "source": "Binance USD-M aggTrades",
        "aggregation": "direct event aggregation to 30-second UTC buckets",
        "rows": len(bars),
        "start": bars.index[0].isoformat(),
        "end": bars.index[-1].isoformat(),
        "columns": list(bars.columns),
    }
    (output / "btc_usdt_perp_30s_aggtrades_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
