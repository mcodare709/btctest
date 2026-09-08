"""CLI entry point for deterministic Binance archive cleaning and features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.dataset_builder import build_archive_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="data/binance")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    print(json.dumps(build_archive_dataset(args.archive_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
