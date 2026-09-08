"""Measure row loss under different NaN deletion policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from btc_perp.feature_schema import CORE_FEATURES


def measure(path: Path, chunksize: int = 200_000) -> dict[str, int | float]:
    columns = ["timestamp", "open", "high", "low", "close", "volume", *CORE_FEATURES]
    total = any_feature_nan = any_required_nan = 0
    for chunk in pd.read_csv(path, compression="gzip", usecols=columns, chunksize=chunksize):
        required = chunk[["timestamp", "open", "high", "low", "close", "volume"]].isna().any(axis=1)
        feature_nan = chunk[list(CORE_FEATURES)].isna().any(axis=1)
        total += len(chunk)
        any_feature_nan += int(feature_nan.sum())
        any_required_nan += int(required.sum())
    return {
        "rows": total,
        "rows_deleted_by_dropna_core_features": any_feature_nan,
        "pct_deleted_by_dropna_core_features": 100.0 * any_feature_nan / total,
        "rows_deleted_by_dropna_required_ohlcv": any_required_nan,
        "pct_deleted_by_dropna_required_ohlcv": 100.0 * any_required_nan / total,
        "rows_kept_with_core_nan": total - any_feature_nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    args = parser.parse_args()
    print(json.dumps(measure(args.input), indent=2))


if __name__ == "__main__":
    main()
