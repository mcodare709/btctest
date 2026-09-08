"""Download recent Binance USDⓈ-M derivative feeds with explicit HTTP checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
FEEDS: tuple[tuple[str, str, dict[str, object], str], ...] = (
    (
        "open_interest_5m",
        "/futures/data/openInterestHist",
        {"symbol": "BTCUSDT", "period": "5m", "limit": 500},
        "timestamp",
    ),
    (
        "long_short_ratio_5m",
        "/futures/data/globalLongShortAccountRatio",
        {"symbol": "BTCUSDT", "period": "5m", "limit": 500},
        "timestamp",
    ),
    (
        "funding_events",
        "/fapi/v1/fundingRate",
        {"symbol": "BTCUSDT", "limit": 1000},
        "fundingTime",
    ),
)


def download_derivatives(
    output_dir: Path,
    *,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
) -> dict[str, dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, Any]] = {}

    with requests.Session() as session:
        for name, path, params, time_column in FEEDS:
            url = f"{base_url.rstrip('/')}{path}"
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"{name} endpoint returned a non-list payload")
            frame = pd.DataFrame(payload)
            if time_column not in frame.columns:
                raise ValueError(f"{name} response is missing timestamp column {time_column!r}")
            frame.to_csv(output_dir / f"{name}.csv", index=False)
            timestamp = pd.to_datetime(frame[time_column], unit="ms", utc=True, errors="raise")
            metadata[name] = {
                "rows": len(frame),
                "start": timestamp.min().isoformat(),
                "end": timestamp.max().isoformat(),
                "columns": list(frame.columns),
                "source": url,
            }

    (output_dir / "coverage.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/derivatives"))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    print(
        json.dumps(
            download_derivatives(
                args.output_dir,
                base_url=args.base_url,
                timeout=args.timeout,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
