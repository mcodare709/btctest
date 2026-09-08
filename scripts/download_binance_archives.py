"""Download official Binance USD-M BTCUSDT monthly archive data."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import requests

BASE = "https://data.binance.vision/data/futures/um/monthly"


def months(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current.strftime("%Y-%m")
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def download(kind: str, interval: str, target: Path, start: date, end: date) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    saved = []
    for month in months(start, end):
        if kind in {"aggTrades", "trades"}:
            name = f"BTCUSDT-{kind}-{month}.zip"
            url = f"{BASE}/{kind}/BTCUSDT/{name}"
            output = target / kind / name
        else:
            name = f"BTCUSDT-{interval}-{month}.zip"
            url = f"{BASE}/{kind}/BTCUSDT/{interval}/{name}"
            output = target / kind / interval / name
        if output.exists():
            saved.append(output)
            continue
        response = requests.get(url, timeout=120)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(response.content)
        saved.append(output)
        print(f"downloaded {output}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/binance"))
    parser.add_argument("--start", default="2019-09")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m"))
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--kinds", nargs="+", default=["klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"])
    args = parser.parse_args()
    start = date.fromisoformat(args.start + "-01")
    end = date.fromisoformat(args.end + "-01")
    for kind in args.kinds:
        files = download(kind, args.interval, args.output_dir, start, end)
        print(f"{kind}: {len(files)} files")


if __name__ == "__main__":
    main()
