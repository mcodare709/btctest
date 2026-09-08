"""Purged rolling OOS comparison for joint and independent return models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.evaluation import chronological_train_validation_indices
from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.protocol import net_horizon_returns


def metrics(prediction: np.ndarray, actual: np.ndarray, threshold_bps: float) -> dict[str, object]:
    error = prediction - actual
    threshold = threshold_bps / 10_000.0
    actions = np.where((prediction[:, 0] > threshold) & (prediction[:, 0] > prediction[:, 1]), 1, np.where((prediction[:, 1] > threshold) & (prediction[:, 1] > prediction[:, 0]), -1, 0))
    realized = np.where(actions == 1, actual[:, 0], np.where(actions == -1, actual[:, 1], 0.0))
    traded = actions != 0
    return {
        "long_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "short_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "long_mae": float(np.mean(np.abs(error[:, 0]))),
        "short_mae": float(np.mean(np.abs(error[:, 1]))),
        "long_count": int((actions == 1).sum()),
        "short_count": int((actions == -1).sum()),
        "flat_count": int((actions == 0).sum()),
        "trade_count": int(traded.sum()),
        "mean_trade_net_return": float(realized[traded].mean()) if traded.any() else 0.0,
        "total_trade_net_return": float(realized[traded].sum()) if traded.any() else 0.0,
    }


def fit_predictions(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    *,
    horizon_bars: int,
    validation_fraction: float,
    iterations: int,
    depth: int,
    thread_count: int,
) -> dict[str, np.ndarray]:
    from catboost import CatBoostRegressor

    fit_end, validation_start = chronological_train_validation_indices(
        len(x_train),
        horizon_bars=horizon_bars,
        validation_fraction=validation_fraction,
    )
    x_fit = x_train.iloc[:fit_end]
    y_fit = y_train[:fit_end]
    x_validation = x_train.iloc[validation_start:]
    y_validation = y_train[validation_start:]
    common = dict(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=thread_count,
        use_best_model=True,
    )
    joint = CatBoostRegressor(loss_function="MultiRMSE", eval_metric="MultiRMSE", **common)
    joint.fit(x_fit, y_fit, eval_set=(x_validation, y_validation), verbose=False)
    independent: list[np.ndarray] = []
    for index in (0, 1):
        model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **common)
        model.fit(x_fit, y_fit[:, index], eval_set=(x_validation, y_validation[:, index]), verbose=False)
        independent.append(np.asarray(model.predict(x_test), dtype=float))
    return {
        "joint_multirmse": np.asarray(joint.predict(x_test), dtype=float),
        "separate_regressors": np.column_stack(independent),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/core_return_walk_forward.json"))
    parser.add_argument("--train-bars", type=int, default=1_500_000)
    parser.add_argument("--test-bars", type=int, default=300_000)
    parser.add_argument("--fold-count", type=int, default=3)
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
    folds: list[dict[str, object]] = []
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
            "train_start": x.index[train_start].isoformat(),
            "train_end": x.index[train_end - 1].isoformat(),
            "test_start": x.index[test_start].isoformat(),
            "test_end": x.index[test_end - 1].isoformat(),
            "train_rows": len(x_train),
            "fit_rows": fit_end,
            "validation_rows": len(x_train) - validation_start,
            "purged_train_validation_rows": validation_start - fit_end,
            "test_rows": len(x_test),
            "purge_train_test_rows": purge_bars,
            "purge_rows": purge_bars,
            "joint_multirmse": metrics(predictions["joint_multirmse"], y_test, args.threshold_bps),
            "separate_regressors": metrics(predictions["separate_regressors"], y_test, args.threshold_bps),
        })
    if not folds:
        raise ValueError("fold configuration does not fit available rows")
    report = {"rows": len(x), "fold_count": len(folds), "train_bars": args.train_bars, "test_bars": args.test_bars, "purge_bars": args.horizon_bars + 1, "threshold_bps": args.threshold_bps, "validation_fraction": args.validation_fraction, "iterations": args.iterations, "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
