"""Deterministic smoke-test data generator; not a research dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int = 6_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="30s", tz="UTC")
    regime = np.where((np.arange(rows) // 600) % 2 == 0, 0.00002, -0.000015)
    shocks = rng.normal(0.0, 0.00035, rows)
    returns = regime + shocks
    close = 90_000 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.00005, 0.00035, rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.00005, 0.00035, rows))
    volume = rng.lognormal(mean=3.0, sigma=0.4, size=rows)
    taker_buy = volume * np.clip(0.5 + returns * 800, 0.05, 0.95)
    taker_sell = volume - taker_buy
    spread = close * 0.0002
    funding = np.full(rows, np.nan)
    funding[::960] = 0.0001

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding,
            "open_interest": 10_000 + np.cumsum(rng.normal(0, 5, rows)),
            "bid_price": close - spread / 2,
            "ask_price": close + spread / 2,
            "bid_size": rng.lognormal(3, 0.5, rows),
            "ask_size": rng.lognormal(3, 0.5, rows),
            "taker_buy_volume": taker_buy,
            "taker_sell_volume": taker_sell,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=6_000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(rows=args.rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
