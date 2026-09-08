"""Fixed-parameter purged walk-forward evaluation of the Core technical expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.protocol import EXECUTION_PROTOCOL_VERSION, net_horizon_returns
from btc_perp.return_policy import expected_net_returns, non_overlapping_actions


def _side_metrics(pnl: np.ndarray) -> dict[str, float | int | None]:
    gains = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    losses = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    if pnl.size:
        equity = np.cumprod(1.0 + pnl)
        peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))
        drawdowns = 1.0 - np.concatenate(([1.0], equity)) / peaks
        compounded_return = float(equity[-1] - 1.0)
        max_drawdown = float(drawdowns.max())
    else:
        compounded_return = 0.0
        max_drawdown = 0.0
    return {
        "count": int(pnl.size),
        "win_rate": float((pnl > 0).mean()) if pnl.size else None,
        "mean_net_return": float(pnl.mean()) if pnl.size else None,
        "median_net_return": float(np.median(pnl)) if pnl.size else None,
        "total_uncompounded_net_return": float(pnl.sum()),
        "compounded_net_return": compounded_return,
        "max_drawdown": max_drawdown,
        "profit_factor": gains / losses if losses > 0 else None,
    }


def _policy_metrics(actions: np.ndarray, actual_net: np.ndarray) -> dict[str, object]:
    long_mask = actions == 1
    short_mask = actions == -1
    long_pnl = actual_net[long_mask, 0]
    short_pnl = actual_net[short_mask, 1]
    chosen = np.where(long_mask, actual_net[:, 0], np.where(short_mask, actual_net[:, 1], np.nan))
    all_pnl = chosen[actions != 0]
    return {
        "all": _side_metrics(all_pnl),
        "long": _side_metrics(long_pnl),
        "short": _side_metrics(short_pnl),
        "flat_rows": int((actions == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/core_gross_return_walk_forward.json"))
    parser.add_argument("--train-bars", type=int, default=1_000_000)
    parser.add_argument("--test-bars", type=int, default=250_000)
    parser.add_argument("--step-bars", type=int, default=250_000)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=51)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--thread-count", type=int, default=10)
    args = parser.parse_args()

    from catboost import CatBoostRegressor

    columns = ["timestamp", "open", "close", *CORE_FEATURES]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    target_frame = net_horizon_returns(
        frame,
        args.horizon_bars,
        fee_rate=0.0004,
        slippage_bps=1.0,
        default_spread_bps=2.0,
    )
    valid = target_frame[["gross_long_return", "long_net_return", "short_net_return"]].notna().all(axis=1)
    x = frame.loc[valid, list(CORE_FEATURES)].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    gross_target = target_frame.loc[valid, "gross_long_return"].to_numpy(dtype=np.float32)
    actual_net = target_frame.loc[valid, ["long_net_return", "short_net_return"]].to_numpy(dtype=np.float32)

    purge_bars = args.horizon_bars + 1
    fixed_cost = 2.0 * (0.0004 + 1.0 / 10_000.0) + 2.0 / 10_000.0
    fold_reports: list[dict[str, object]] = []
    aggregate_pnl: list[tuple[int, float]] = []
    for fold_index in range(args.fold_count):
        train_start = fold_index * args.step_bars
        train_end = train_start + args.train_bars
        test_start = train_end + purge_bars
        test_end = test_start + args.test_bars
        if test_end > len(x):
            break
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=args.thread_count,
        )
        model.fit(x.iloc[train_start:train_end], gross_target[train_start:train_end], verbose=False)
        gross_prediction = np.asarray(model.predict(x.iloc[test_start:test_end]), dtype=float)
        expected_net = expected_net_returns(gross_prediction, fixed_cost)
        actions = non_overlapping_actions(
            expected_net,
            threshold=args.threshold_bps / 10_000.0,
            horizon_bars=args.horizon_bars,
        )
        fold_actual = actual_net[test_start:test_end]
        chosen = np.where(actions == 1, fold_actual[:, 0], np.where(actions == -1, fold_actual[:, 1], np.nan))
        aggregate_pnl.extend((int(action), float(pnl)) for action, pnl in zip(actions, chosen) if action != 0)
        fold_reports.append(
            {
                "fold": fold_index,
                "train_start": x.index[train_start].isoformat(),
                "train_end": x.index[train_end - 1].isoformat(),
                "test_start": x.index[test_start].isoformat(),
                "test_end": x.index[test_end - 1].isoformat(),
                "gross_rmse": float(np.sqrt(np.mean((gross_prediction - gross_target[test_start:test_end]) ** 2))),
                "gross_mae": float(np.mean(np.abs(gross_prediction - gross_target[test_start:test_end]))),
                "policy": _policy_metrics(actions, fold_actual),
            }
        )
        print(f"fold {fold_index + 1}: {fold_reports[-1]['test_start']} -> {fold_reports[-1]['test_end']}, trades={fold_reports[-1]['policy']['all']['count']}", flush=True)

    if not fold_reports:
        raise ValueError("fold configuration does not fit available rows")
    aggregate_actions = np.asarray([item[0] for item in aggregate_pnl], dtype=np.int8)
    aggregate_values = np.asarray([item[1] for item in aggregate_pnl], dtype=float)
    aggregate_long = aggregate_values[aggregate_actions == 1]
    aggregate_short = aggregate_values[aggregate_actions == -1]
    fold_totals = np.asarray(
        [fold["policy"]["all"]["total_uncompounded_net_return"] for fold in fold_reports],
        dtype=float,
    )
    report = {
        "experiment": "fixed-core-gross-return-walk-forward-v1",
        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
        "model": "CatBoostRegressor",
        "target": "gross_long_return",
        "feature_names": list(CORE_FEATURES),
        "source_timeframe": "1min",
        "horizon_bars": args.horizon_bars,
        "purge_bars": purge_bars,
        "threshold_bps": args.threshold_bps,
        "fixed_round_trip_cost_bps": fixed_cost * 10_000.0,
        "non_overlapping_positions": True,
        "hyperparameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "random_seed": 42,
        },
        "split": {
            "kind": "rolling_purged_walk_forward",
            "train_bars": args.train_bars,
            "test_bars": args.test_bars,
            "step_bars": args.step_bars,
            "fold_count": len(fold_reports),
        },
        "aggregate_oos": {
            "all": _side_metrics(aggregate_values),
            "long": _side_metrics(aggregate_long),
            "short": _side_metrics(aggregate_short),
            "fold_stability": {
                "positive_folds": int((fold_totals > 0).sum()),
                "negative_folds": int((fold_totals < 0).sum()),
                "median_fold_total_net_return": float(np.median(fold_totals)),
                "best_fold_total_net_return": float(fold_totals.max()),
                "worst_fold_total_net_return": float(fold_totals.min()),
            },
        },
        "folds": fold_reports,
        "limitations": [
            "Core history uses a fixed 12 bps round-trip cost because observed spread and funding are unavailable.",
            "Fold-level models are research artifacts and are not saved for live inference.",
            "Statistical confidence remains limited when trade counts are small or concentrated in a few folds.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
