"""Train separate long/short expected net-return CatBoost models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.data import load_market_csv
from btc_perp.expected_return import train_expected_return_models
from btc_perp.feature_schema import feature_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--schema", default="core", choices=("core", "derivatives", "microstructure"))
    parser.add_argument("--horizon-bars", default=10, type=int)
    parser.add_argument("--timeframe")
    parser.add_argument("--data-source", default="unknown")
    args = parser.parse_args()
    metadata = train_expected_return_models(
        load_market_csv(args.data),
        args.output_prefix,
        horizon_bars=args.horizon_bars,
        timeframe=args.timeframe,
        data_source=args.data_source,
        feature_names=feature_schema(args.schema),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
