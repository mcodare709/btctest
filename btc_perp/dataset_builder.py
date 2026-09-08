"""Clean official Binance archive ZIPs into canonical raw and feature datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .features import build_features

KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote", "ignore",
]


def _read_archives(root: Path, kind: str, value_name: str) -> pd.DataFrame:
    files = sorted((root / kind / "1m").glob("*.zip"))
    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file, header=None, names=KLINE_COLUMNS, compression="zip")
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frame = frame[["timestamp", "close"]].rename(columns={"close": value_name})
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp", value_name])
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp", keep="last")


def build_archive_dataset(archive_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Write UTC-clean raw bars and the canonical no-look-ahead feature table."""
    root = Path(archive_root)
    output = Path(output_dir)
    kline_files = sorted((root / "klines" / "1m").glob("*.zip"))
    if not kline_files:
        raise FileNotFoundError(f"no Kline ZIPs in {root}")
    bars = pd.concat([pd.read_csv(file, header=None, names=KLINE_COLUMNS, compression="zip") for file in kline_files], ignore_index=True)
    bars = bars[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]]
    bars["timestamp"] = pd.to_numeric(bars["timestamp"], errors="coerce")
    bars = bars.dropna(subset=["timestamp"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], unit="ms", utc=True)
    for column in bars.columns.drop("timestamp"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars["taker_sell_volume"] = bars["volume"] - bars["taker_buy_volume"]
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    for kind, name in (("markPriceKlines", "mark_price"), ("indexPriceKlines", "index_price"), ("premiumIndexKlines", "premium_index")):
        extra = _read_archives(root, kind, name)
        if not extra.empty:
            extra["timestamp"] = pd.to_datetime(extra["timestamp"], unit="ms", utc=True)
            extra[name] = pd.to_numeric(extra[name], errors="coerce")
            bars = bars.merge(extra, on="timestamp", how="left")
    features = build_features(bars.set_index("timestamp")).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "btc_usdt_perp_1m_raw.csv.gz"
    feature_path = output / "btc_usdt_perp_1m_features.csv.gz"
    bars.to_csv(raw_path, index=False, compression="gzip")
    features.to_csv(feature_path, index=False, compression="gzip")
    metadata = {
        "rows": len(bars), "start": bars["timestamp"].iloc[0].isoformat(), "end": bars["timestamp"].iloc[-1].isoformat(),
        "raw_columns": list(bars.columns), "feature_columns": list(features.columns),
        "missing_features": features.isna().sum().loc[lambda values: values > 0].to_dict(),
    }
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
