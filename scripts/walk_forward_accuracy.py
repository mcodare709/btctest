"""Walk-forward accuracy and win-rate report for return policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btc_perp.evaluation import chronological_train_validation_indices
from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.protocol import net_horizon_returns
from scripts.walk_forward_return_models import fit_predictions


def policy_metrics(prediction: np.ndarray, actual: np.ndarray, threshold_bps: float) -> dict[str, object]:
    threshold = threshold_bps / 10_000.0
    predicted = np.where((prediction[:, 0] > threshold) & (prediction[:, 0] > prediction[:, 1]), 1, np.where((prediction[:, 1] > threshold) & (prediction[:, 1] > prediction[:, 0]), -1, 0))
    realized_best = np.where((actual[:, 0] > threshold) & (actual[:, 0] > actual[:, 1]), 1, np.where((actual[:, 1] > threshold) & (actual[:, 1] > actual[:, 0]), -1, 0))
    traded = predicted != 0
    long = predicted == 1
    short = predicted == -1
    realized = np.where(long, actual[:, 0], np.where(short, actual[:, 1], 0.0))
    return {
        "decision_accuracy_all_rows": float((predicted == realized_best).mean()),
        "direction_accuracy_on_trades": float((predicted[traded] == realized_best[traded]).mean()) if traded.any() else 0.0,
        "win_rate": float((realized[traded] > 0).mean()) if traded.any() else 0.0,
        "long_count": int(long.sum()),
        "short_count": int(short.sum()),
        "long_win_rate": float((actual[long, 0] > 0).mean()) if long.any() else 0.0,
        "short_win_rate": float((actual[short, 1] > 0).mean()) if short.any() else 0.0,
        "flat_count": int((predicted == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/core_return_accuracy.json"))
    parser.add_argument("--train-bars", type=int, default=1_500_000)
    parser.add_argument("--test-bars", type=int, default=300_000)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--fold-step", type=int, default=1_500_000)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--thread-count", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    columns = ["timestamp", "open", "close", *CORE_FEATURES]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    targets = net_horizon_returns(frame, args.horizon_bars, fee_rate=0.0004, slippage_bps=1.0, default_spread_bps=2.0)[["long_net_return", "short_net_return"]]
    valid = targets.notna().all(axis=1)
    x = frame.loc[valid, list(CORE_FEATURES)].astype(np.float32)
    y = targets.loc[valid].to_numpy(dtype=np.float32)
    folds = []
    for fold_index in range(args.fold_count):
        train_start = fold_index * args.fold_step
        train_end = train_start + args.train_bars
        purge_bars = args.horizon_bars + 1
        test_start = train_end + purge_bars
        test_end = test_start + args.test_bars
        if test_end > len(x):
            break
        x_train, y_train = x.iloc[train_start:train_end], y[train_start:train_end]
        x_test, y_test = x.iloc[test_start:test_end], y[test_start:test_end]
        fit_end, validation_start = chronological_train_validation_indices(
            len(x_train),
            horizon_bars=args.horizon_bars,
            validation_fraction=args.validation_fraction,
        )
        predictions = fit_predictions(
            x_train,
            y_train,
            x_test,
            horizon_bars=args.horizon_bars,
            validation_fraction=args.validation_fraction,
            iterations=args.iterations,
            depth=args.depth,
            thread_count=args.thread_count,
        )
        folds.append({
            "fold": fold_index,
            "test_start": x.index[test_start].isoformat(),
            "test_end": x.index[test_end - 1].isoformat(),
            "train_rows": len(x_train),
            "fit_rows": fit_end,
            "validation_rows": len(x_train) - validation_start,
            "purged_train_validation_rows": validation_start - fit_end,
            "purge_train_test_rows": purge_bars,
            "joint_multirmse": policy_metrics(predictions["joint_multirmse"], y_test, args.threshold_bps),
            "separate_regressors": policy_metrics(predictions["separate_regressors"], y_test, args.threshold_bps),
        })
    report = {"threshold_bps": args.threshold_bps, "validation_fraction": args.validation_fraction, "fold_count": len(folds), "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
