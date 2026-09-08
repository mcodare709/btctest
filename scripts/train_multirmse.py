"""Train a CatBoost MultiRMSE long/short smoke or full model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.evaluation import chronological_train_validation_indices
from btc_perp.expected_return import _catboost_regressor
from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.protocol import EXECUTION_PROTOCOL_VERSION, net_horizon_returns


def train(input_path: Path, output_path: Path, *, rows: int | None, horizon_bars: int, iterations: int, depth: int, learning_rate: float, task_type: str, thread_count: int) -> dict[str, object]:
    columns = ["timestamp", "open", "close", *CORE_FEATURES]
    frame = pd.read_csv(input_path, compression="gzip", usecols=columns, nrows=rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    targets = net_horizon_returns(frame, horizon_bars, fee_rate=0.0004, slippage_bps=1.0, default_spread_bps=2.0)[["long_net_return", "short_net_return"]]
    valid = targets.notna().all(axis=1)
    x = frame.loc[valid, list(CORE_FEATURES)].astype(np.float32)
    y = targets.loc[valid].to_numpy(dtype=np.float32)
    if len(x) < 100:
        raise ValueError("not enough valid rows for MultiRMSE training")
    train_end, validation_start = chronological_train_validation_indices(
        len(x),
        horizon_bars=horizon_bars,
    )
    model = _catboost_regressor()(loss_function="MultiRMSE", eval_metric="MultiRMSE", iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=42, verbose=False, allow_writing_files=False, task_type=task_type, thread_count=thread_count if task_type == "CPU" else None)
    model.fit(x.iloc[:train_end], y[:train_end], eval_set=(x.iloc[validation_start:], y[validation_start:]), verbose=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    metadata = {"model": "CatBoostRegressor", "loss_function": "MultiRMSE", "targets": ["long_net_return", "short_net_return"], "feature_names": list(CORE_FEATURES), "execution_protocol": EXECUTION_PROTOCOL_VERSION, "horizon_bars": horizon_bars, "rows_loaded": len(frame), "rows_valid": len(x), "train_rows": train_end, "validation_rows": len(x) - validation_start, "purged_rows": validation_start - train_end, "task_type": task_type, "iterations": iterations, "depth": depth, "learning_rate": learning_rate, "best_iteration": model.get_best_iteration(), "best_score": model.get_best_score()}
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/multirmse_core_smoke.cbm"))
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--thread-count", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(train(input_path=args.input, output_path=args.output, rows=args.rows, horizon_bars=args.horizon_bars, iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate, task_type=args.task_type, thread_count=args.thread_count), indent=2, default=str))


if __name__ == "__main__":
    main()
