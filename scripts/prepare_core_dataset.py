"""Prepare the long-history Core model dataset without filling missing values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from btc_perp.feature_schema import CORE_FEATURES


BASE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def prepare_core_dataset(input_path: Path, output_path: Path, chunksize: int = 200_000) -> dict[str, object]:
    source_columns = pd.read_csv(input_path, compression="gzip", nrows=0).columns.tolist()
    selected = [column for column in (*BASE_COLUMNS, *CORE_FEATURES) if column in source_columns]
    missing_schema = [column for column in CORE_FEATURES if column not in source_columns]
    if missing_schema:
        raise ValueError(f"core schema columns missing from source: {missing_schema}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    rows = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    missing_counts = {column: 0 for column in selected}
    zero_counts = {column: 0 for column in selected}

    for number, chunk in enumerate(pd.read_csv(input_path, compression="gzip", usecols=selected, chunksize=chunksize)):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
        chunk = chunk.sort_values("timestamp")
        rows += len(chunk)
        if first_timestamp is None and len(chunk):
            first_timestamp = chunk["timestamp"].iloc[0].isoformat()
        if len(chunk):
            last_timestamp = chunk["timestamp"].iloc[-1].isoformat()
        for column in selected:
            values = pd.to_numeric(chunk[column], errors="coerce")
            missing_counts[column] += int(values.isna().sum())
            zero_counts[column] += int((values == 0).sum())
        chunk.to_csv(output_path, mode="w" if number == 0 else "a", header=number == 0, index=False, compression="gzip")

    metadata = {
        "schema": "core",
        "schema_version": "named-v1",
        "source": str(input_path),
        "output": str(output_path),
        "rows": rows,
        "start": first_timestamp,
        "end": last_timestamp,
        "columns": selected,
        "missing_rate": {column: missing_counts[column] / rows for column in selected},
        "zero_rate": {column: zero_counts[column] / rows for column in selected},
        "missing_value_policy": "preserve_nan",
        "synthetic_augmentation": False,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_usdt_perp_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    print(json.dumps(prepare_core_dataset(args.input, args.output, args.chunksize), indent=2))


if __name__ == "__main__":
    main()
