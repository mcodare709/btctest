"""Validate a canonical dataset and record feature availability without loading it all at once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btc_perp.ml_model import MODEL_FEATURES


def inspect_dataset(path: Path, output: Path, chunksize: int = 200_000) -> dict[str, object]:
    rows = 0
    missing: dict[str, int] = {}
    minimum: dict[str, float] = {}
    maximum: dict[str, float] = {}
    active: list[str] = []
    timestamps: list[pd.Series] = []
    invalid_ohlc = 0

    for chunk in pd.read_csv(path, compression="gzip", chunksize=chunksize):
        rows += len(chunk)
        for column, count in chunk.isna().sum().items():
            missing[column] = missing.get(column, 0) + int(count)
        for column in chunk.columns:
            values = pd.to_numeric(chunk[column], errors="coerce").dropna()
            if not values.empty:
                minimum[column] = min(minimum.get(column, float("inf")), float(values.min()))
                maximum[column] = max(maximum.get(column, float("-inf")), float(values.max()))
        invalid_ohlc += int(((chunk["low"] > chunk["high"]) | (chunk["open"] < chunk["low"]) | (chunk["open"] > chunk["high"]) | (chunk["close"] < chunk["low"]) | (chunk["close"] > chunk["high"])).sum())
        timestamps.append(pd.to_datetime(chunk["timestamp"], utc=True))

    timestamp = pd.concat(timestamps, ignore_index=True)
    availability = {column: 1.0 - count / rows for column, count in missing.items()}
    constant_features = sorted(column for column in minimum if minimum[column] == maximum[column])
    active = [
        column for column in MODEL_FEATURES
        if availability.get(column, 0.0) >= 0.95 and column not in constant_features
    ]
    report = {
        "path": str(path),
        "rows": rows,
        "start": timestamp.iloc[0].isoformat(),
        "end": timestamp.iloc[-1].isoformat(),
        "duplicate_timestamps": int(timestamp.duplicated().sum()),
        "missing_minutes": int(timestamp.diff().dt.total_seconds().div(60).sub(1).clip(lower=0).sum()),
        "invalid_ohlc": invalid_ohlc,
        "columns": len(availability),
        "availability": availability,
        "constant_features": constant_features,
        "trainable_features_at_95pct": active,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/btc_usdt_perp_1m_features.csv.gz")
    parser.add_argument("--output", default="data/processed/model_feature_availability.json")
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(Path(args.input), Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
