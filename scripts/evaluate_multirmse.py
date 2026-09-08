"""Evaluate a saved MultiRMSE artifact on its chronological validation tail."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.protocol import net_horizon_returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/multirmse_core_smoke.cbm"))
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    args = parser.parse_args()

    from catboost import CatBoostRegressor

    columns = ["timestamp", "open", "close", *CORE_FEATURES]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns, nrows=args.rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    targets = net_horizon_returns(frame, args.horizon_bars, fee_rate=0.0004, slippage_bps=1.0, default_spread_bps=2.0)[["long_net_return", "short_net_return"]]
    valid = targets.notna().all(axis=1)
    x = frame.loc[valid, list(CORE_FEATURES)].astype(np.float32)
    y = targets.loc[valid].to_numpy(dtype=np.float32)
    split = int(len(x) * 0.8)
    model = CatBoostRegressor()
    model.load_model(str(args.model))
    prediction = np.asarray(model.predict(x.iloc[split:]), dtype=float)
    actual = y[split:]
    errors = prediction - actual
    threshold = args.threshold_bps / 10_000.0
    actions = np.where((prediction[:, 0] > threshold) & (prediction[:, 0] > prediction[:, 1]), 1, np.where((prediction[:, 1] > threshold) & (prediction[:, 1] > prediction[:, 0]), -1, 0))
    print(json.dumps({
        "validation_rows": len(actual),
        "long_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "short_rmse": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "long_mae": float(np.mean(np.abs(errors[:, 0]))),
        "short_mae": float(np.mean(np.abs(errors[:, 1]))),
        "mean_realized_long_return": float(actual[:, 0].mean()),
        "mean_realized_short_return": float(actual[:, 1].mean()),
        "action_distribution": {"long": int((actions == 1).sum()), "flat": int((actions == 0).sum()), "short": int((actions == -1).sum())},
        "threshold_bps": args.threshold_bps,
    }, indent=2))


if __name__ == "__main__":
    main()
