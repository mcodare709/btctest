"""Compare joint MultiRMSE with two independent regressors on one OOS split."""

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


def _metrics(prediction: np.ndarray, actual: np.ndarray, threshold_bps: float) -> dict[str, object]:
    error = prediction - actual
    threshold = threshold_bps / 10_000.0
    actions = np.where((prediction[:, 0] > threshold) & (prediction[:, 0] > prediction[:, 1]), 1, np.where((prediction[:, 1] > threshold) & (prediction[:, 1] > prediction[:, 0]), -1, 0))
    selected = np.where(actions == 1, actual[:, 0], np.where(actions == -1, actual[:, 1], 0.0))
    traded = actions != 0
    return {
        "long_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "short_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "long_mae": float(np.mean(np.abs(error[:, 0]))),
        "short_mae": float(np.mean(np.abs(error[:, 1]))),
        "action_distribution": {"long": int((actions == 1).sum()), "flat": int((actions == 0).sum()), "short": int((actions == -1).sum())},
        "selected_trade_count": int(traded.sum()),
        "selected_mean_realized_return": float(selected[traded].mean()) if traded.any() else 0.0,
        "selected_total_realized_return": float(selected[traded].sum()) if traded.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/core_return_model_comparison.json"))
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--thread-count", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
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
    validation_end = int(len(x) * 0.8)
    purge_bars = args.horizon_bars + 1
    test_start = validation_end + purge_bars
    if test_start >= len(x):
        raise ValueError("split does not leave fit, validation, purge, and test rows")
    fit_end, validation_start = chronological_train_validation_indices(
        validation_end,
        horizon_bars=args.horizon_bars,
        validation_fraction=args.validation_fraction,
    )
    x_fit, y_fit = x.iloc[:fit_end], y[:fit_end]
    x_validation, y_validation = x.iloc[validation_start:validation_end], y[validation_start:validation_end]
    x_test, y_test = x.iloc[test_start:], y[test_start:]

    common = dict(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=args.thread_count,
        use_best_model=True,
    )
    joint = CatBoostRegressor(loss_function="MultiRMSE", eval_metric="MultiRMSE", **common)
    joint.fit(x_fit, y_fit, eval_set=(x_validation, y_validation), verbose=False)
    joint_prediction = np.asarray(joint.predict(x_test), dtype=float)

    separate_prediction = []
    for index, side in enumerate(("long", "short")):
        model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **common)
        model.fit(x_fit, y_fit[:, index], eval_set=(x_validation, y_validation[:, index]), verbose=False)
        separate_prediction.append(np.asarray(model.predict(x_test), dtype=float))
    separate_prediction = np.column_stack(separate_prediction)

    report = {
        "rows_loaded": len(frame), "rows_valid": len(x), "fit_rows": len(x_fit), "validation_rows": len(x_validation), "test_rows": len(x_test), "purge_fit_validation_rows": validation_start - fit_end, "purge_validation_test_rows": purge_bars, "features": list(CORE_FEATURES), "iterations": args.iterations, "validation_fraction": args.validation_fraction, "threshold_bps": args.threshold_bps, "joint_multirmse": _metrics(joint_prediction, y_test, args.threshold_bps), "separate_regressors": _metrics(separate_prediction, y_test, args.threshold_bps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
