"""Train and evaluate the long-history technical expected-return expert.

The script keeps the final chronological block outside the fitting rows, purges
the label horizon at the boundary, and uses that block for early stopping and
candidate-policy validation before saving separate long/short models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.evaluation import chronological_train_validation_indices
from btc_perp.expected_return import EXPECTED_RETURN_SCHEMA_VERSION
from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.protocol import EXECUTION_PROTOCOL_VERSION, net_horizon_returns


def _policy_metrics(prediction: np.ndarray, actual: np.ndarray, threshold_bps: float) -> dict[str, object]:
    threshold = threshold_bps / 10_000.0
    actions = np.where(
        (prediction[:, 0] > threshold) & (prediction[:, 0] > prediction[:, 1]),
        1,
        np.where(
            (prediction[:, 1] > threshold) & (prediction[:, 1] > prediction[:, 0]),
            -1,
            0,
        ),
    )
    realized = np.where(actions == 1, actual[:, 0], np.where(actions == -1, actual[:, 1], 0.0))
    traded = actions != 0
    pnl = realized[traded]
    gross_profit = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    return {
        "threshold_bps": threshold_bps,
        "actions": {
            "long": int((actions == 1).sum()),
            "short": int((actions == -1).sum()),
            "flat": int((actions == 0).sum()),
        },
        "trade_count": int(traded.sum()),
        "win_rate": float((pnl > 0).mean()) if pnl.size else None,
        "mean_net_return": float(pnl.mean()) if pnl.size else None,
        "median_net_return": float(np.median(pnl)) if pnl.size else None,
        "total_uncompounded_net_return": float(pnl.sum()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/models/technical_expert"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--thread-count", type=int, default=10)
    parser.add_argument("--rows", type=int, default=None)
    args = parser.parse_args()

    from catboost import CatBoostRegressor

    columns = ["timestamp", "open", "close", *CORE_FEATURES]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns, nrows=args.rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    targets = net_horizon_returns(
        frame,
        args.horizon_bars,
        fee_rate=0.0004,
        slippage_bps=1.0,
        default_spread_bps=2.0,
    )[["long_net_return", "short_net_return"]]
    valid = targets.notna().all(axis=1)
    x = frame.loc[valid, list(CORE_FEATURES)].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    y = targets.loc[valid].to_numpy(dtype=np.float32)

    train_end, validation_start = chronological_train_validation_indices(
        len(x),
        horizon_bars=args.horizon_bars,
        validation_fraction=args.validation_fraction,
    )
    purge_rows = validation_start - train_end
    if train_end < 100:
        raise ValueError("not enough rows for chronological train/purge/validation split")
    x_train, y_train = x.iloc[:train_end], y[:train_end]
    x_validation, y_validation = x.iloc[validation_start:], y[validation_start:]

    common = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=42,
        verbose=50,
        allow_writing_files=False,
        thread_count=args.thread_count,
    )
    predictions: list[np.ndarray] = []
    model_details: dict[str, object] = {}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for index, side in enumerate(("long", "short")):
        model = CatBoostRegressor(**common)
        model.fit(
            x_train,
            y_train[:, index],
            eval_set=(x_validation, y_validation[:, index]),
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        prediction = np.asarray(model.predict(x_validation), dtype=float)
        predictions.append(prediction)
        model_path = args.output_prefix.with_name(f"{args.output_prefix.name}_{side}.cbm")
        model.save_model(str(model_path))
        error = prediction - y_validation[:, index]
        model_details[side] = {
            "artifact": str(model_path),
            "best_iteration": int(model.get_best_iteration()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "prediction_mean": float(prediction.mean()),
            "target_mean": float(y_validation[:, index].mean()),
        }

    prediction_matrix = np.column_stack(predictions)
    report = {
        "schema_version": EXPECTED_RETURN_SCHEMA_VERSION,
        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
        "model_role": "technical_expert",
        "model_family": "separate_catboost_regressors",
        "feature_names": list(CORE_FEATURES),
        "feature_schema_version": "core-v1",
        "source_timeframe": "1min",
        "horizon_bars": args.horizon_bars,
        "horizon_seconds": args.horizon_bars * 60,
        "data_source": "Binance BTCUSDT perpetual 1m processed core dataset",
        "cost_assumptions": {"fee_rate": 0.0004, "slippage_bps": 1.0, "default_spread_bps": 2.0},
        "rows_loaded": len(frame),
        "rows_valid": len(x),
        "train_rows": len(x_train),
        "purged_rows": purge_rows,
        "validation_rows": len(x_validation),
        "training_start": x_train.index[0].isoformat(),
        "training_end": x_train.index[-1].isoformat(),
        "validation_start": x_validation.index[0].isoformat(),
        "validation_end": x_validation.index[-1].isoformat(),
        "hyperparameters": {
            "iterations_cap": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "random_seed": 42,
        },
        "models": model_details,
        "threshold_sweep_validation": [
            _policy_metrics(prediction_matrix, y_validation, threshold)
            for threshold in (2.0, 3.0, 5.0, 8.0, 10.0)
        ],
        "limitations": [
            "The chronological holdout is used for early stopping and is validation, not untouched final OOS data.",
            "Funding and observed bid/ask spread are unavailable in the long-history core dataset.",
            "Threshold selection on this holdout would overfit; freeze it only after walk-forward validation.",
        ],
    }
    report_path = args.output_prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
