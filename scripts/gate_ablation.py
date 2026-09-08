"""Purged walk-forward ablation for CatBoost return-gate density and quality."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.feature_schema import CORE_FEATURES
from btc_perp.gate_ablation import (
    calibrated_gate_threshold,
    evaluate_non_overlapping_gate,
    side_metrics,
    validate_fold_layout,
)
from btc_perp.protocol import EXECUTION_PROTOCOL_VERSION, net_horizon_returns
from btc_perp.return_policy import expected_net_returns


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or min(values) < 0:
        raise argparse.ArgumentTypeError("expected comma-separated non-negative numbers")
    return values


def _candidate_id(safety_bps: float, top_fraction: float | None) -> str:
    quantile = "absolute" if top_fraction is None else f"top_{top_fraction * 100:g}pct"
    return f"safety_{safety_bps:g}bps__{quantile}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/core_gate_ablation.json"))
    parser.add_argument("--horizons", type=_csv_ints, default=(10, 30, 60, 120))
    parser.add_argument("--safety-margins-bps", type=_csv_floats, default=(0.0, 2.0, 5.0))
    parser.add_argument("--top-fractions", type=_csv_floats, default=(0.02, 0.01, 0.005))
    parser.add_argument("--train-bars", type=int, default=600_000)
    parser.add_argument("--calibration-bars", type=int, default=120_000)
    parser.add_argument("--test-bars", type=int, default=120_000)
    parser.add_argument("--step-bars", type=int, default=120_000)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=51)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--thread-count", type=int, default=10)
    args = parser.parse_args()

    from catboost import CatBoostRegressor

    max_horizon = max(args.horizons)
    purge_bars = max_horizon + 1
    required_rows = (
        (args.fold_count - 1) * args.step_bars
        + args.train_bars
        + purge_bars
        + args.test_bars
    )
    validate_fold_layout(
        required_rows,
        train_bars=args.train_bars,
        calibration_bars=args.calibration_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        purge_bars=purge_bars,
        fold_count=args.fold_count,
    )
    columns = ["timestamp", "open", "close", *CORE_FEATURES]
    raw = pd.read_csv(args.input, compression="gzip", usecols=columns)
    if len(raw) < required_rows + max_horizon + 1:
        raise ValueError("input does not contain enough rows for requested folds")
    raw = raw.tail(required_rows + max_horizon + 1).copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("timestamp").sort_index()
    evaluation_frame = raw.iloc[:required_rows]
    x = evaluation_frame[list(CORE_FEATURES)].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    fixed_cost = 2.0 * (0.0004 + 1.0 / 10_000.0) + 2.0 / 10_000.0
    top_fractions: tuple[float | None, ...] = (None, *args.top_fractions)
    candidates = [
        (safety, top_fraction, _candidate_id(safety, top_fraction))
        for safety in args.safety_margins_bps
        for top_fraction in top_fractions
    ]
    results: list[dict[str, object]] = []

    for horizon in args.horizons:
        targets = net_horizon_returns(
            raw,
            horizon,
            fee_rate=0.0004,
            slippage_bps=1.0,
            default_spread_bps=2.0,
        ).loc[evaluation_frame.index]
        if not targets[["gross_long_return", "long_net_return", "short_net_return"]].notna().all(axis=1).all():
            raise ValueError(f"horizon {horizon} produced missing evaluation targets")
        gross_target = targets["gross_long_return"].to_numpy(dtype=np.float32)
        actual_net = targets[["long_net_return", "short_net_return"]].to_numpy(dtype=float)
        accumulators = {
            candidate_id: {"pnl": [], "sides": [], "fold_totals": [], "folds": []}
            for _, _, candidate_id in candidates
        }
        for fold_index in range(args.fold_count):
            train_start = fold_index * args.step_bars
            train_end = train_start + args.train_bars
            calibration_start = train_end - args.calibration_bars
            fit_end = calibration_start - purge_bars
            test_start = train_end + purge_bars
            test_end = test_start + args.test_bars
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
            model.fit(x.iloc[train_start:fit_end], gross_target[train_start:fit_end], verbose=False)
            calibration_gross = np.asarray(model.predict(x.iloc[calibration_start:train_end]), dtype=float)
            test_gross = np.asarray(model.predict(x.iloc[test_start:test_end]), dtype=float)
            calibration_expected = expected_net_returns(calibration_gross, fixed_cost)
            test_expected = expected_net_returns(test_gross, fixed_cost)
            fold_actual = actual_net[test_start:test_end]
            for safety, top_fraction, candidate_id in candidates:
                threshold = calibrated_gate_threshold(
                    calibration_expected,
                    safety_margin_bps=safety,
                    top_fraction=top_fraction,
                )
                actions, metrics = evaluate_non_overlapping_gate(
                    test_expected,
                    fold_actual,
                    threshold=threshold,
                    horizon_bars=horizon,
                )
                pnl = np.where(
                    actions == 1,
                    fold_actual[:, 0],
                    np.where(actions == -1, fold_actual[:, 1], np.nan),
                )[actions != 0]
                accumulator = accumulators[candidate_id]
                accumulator["pnl"].extend(pnl.tolist())
                accumulator["sides"].extend(actions[actions != 0].tolist())
                accumulator["fold_totals"].append(float(pnl.sum()))
                accumulator["folds"].append(
                    {
                        "fold": fold_index,
                        "threshold_bps": threshold * 10_000.0,
                        "test_start": x.index[test_start].isoformat(),
                        "test_end": x.index[test_end - 1].isoformat(),
                        "metrics": metrics,
                    }
                )
            print(f"horizon={horizon} fold={fold_index + 1}/{args.fold_count}", flush=True)

        for safety, top_fraction, candidate_id in candidates:
            accumulator = accumulators[candidate_id]
            pnl = np.asarray(accumulator["pnl"], dtype=float)
            sides = np.asarray(accumulator["sides"], dtype=np.int8)
            fold_totals = np.asarray(accumulator["fold_totals"], dtype=float)
            all_metrics = side_metrics(pnl)
            positive_folds = int((fold_totals > 0).sum())
            profit_factor = all_metrics["profit_factor"]
            trades_per_30_days = float(len(pnl) / (args.fold_count * args.test_bars / 1_440.0) * 30.0)
            accepted = bool(
                trades_per_30_days >= 30.0
                and int((sides == 1).sum()) >= 10
                and int((sides == -1).sum()) >= 10
                and all_metrics["mean_net_return"] is not None
                and all_metrics["mean_net_return"] > 0
                and profit_factor is not None
                and profit_factor > 1.0
                and positive_folds >= math.ceil(args.fold_count * 0.6)
            )
            results.append(
                {
                    "candidate_id": candidate_id,
                    "horizon_bars": horizon,
                    "safety_margin_bps": safety,
                    "top_fraction": top_fraction,
                    "aggregate_oos": {
                        "all": all_metrics,
                        "long": side_metrics(pnl[sides == 1]),
                        "short": side_metrics(pnl[sides == -1]),
                        "trades_per_30_days": trades_per_30_days,
                        "positive_folds": positive_folds,
                        "negative_folds": int((fold_totals < 0).sum()),
                    },
                    "acceptance_passed": accepted,
                    "folds": accumulator["folds"],
                }
            )

    leaderboard = sorted(
        results,
        key=lambda item: (
            bool(item["acceptance_passed"]),
            int(item["aggregate_oos"]["all"]["count"]) > 0,
            float(item["aggregate_oos"]["all"]["total_uncompounded_net_return"]),
        ),
        reverse=True,
    )
    report = {
        "experiment": "core-catboost-gate-horizon-safety-quantile-ablation-v1",
        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
        "model": "CatBoostRegressor gross return -> symmetric expected net returns",
        "feature_names": list(CORE_FEATURES),
        "source_timeframe": "1min",
        "cost_assumptions": {
            "fee_rate": 0.0004,
            "slippage_bps_each_side": 1.0,
            "spread_bps_round_trip": 2.0,
            "fixed_round_trip_cost_bps": fixed_cost * 10_000.0,
        },
        "split": {
            "kind": "rolling_fit_purge_calibration_purge_test",
            "train_bars": args.train_bars,
            "fit_bars": args.train_bars - args.calibration_bars - purge_bars,
            "calibration_bars": args.calibration_bars,
            "test_bars": args.test_bars,
            "step_bars": args.step_bars,
            "purge_bars": purge_bars,
            "fold_count": args.fold_count,
            "oos_start": leaderboard[0]["folds"][0]["test_start"],
            "oos_end": leaderboard[0]["folds"][-1]["test_end"],
        },
        "grid": {
            "horizons": list(args.horizons),
            "safety_margins_bps": list(args.safety_margins_bps),
            "top_fractions": [None, *args.top_fractions],
        },
        "acceptance": {
            "minimum_trades_per_30_days": 30.0,
            "minimum_each_side": 10,
            "mean_net_return_gt": 0.0,
            "profit_factor_gt": 1.0,
            "minimum_positive_fold_fraction": 0.6,
        },
        "accepted_candidate_count": int(sum(bool(item["acceptance_passed"]) for item in results)),
        "leaderboard": leaderboard,
        "limitations": [
            "Funding and observed spread are unavailable; fixed 12 bps round-trip cost is used.",
            "Grid comparison on the same OOS folds has multiple-testing risk; a selected candidate needs a new untouched holdout.",
            "One random seed and one CatBoost hyperparameter set are used to isolate gate design effects.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accepted_candidate_count": report["accepted_candidate_count"],
        "top_10": [
            {
                "candidate_id": item["candidate_id"],
                "horizon_bars": item["horizon_bars"],
                "trades": item["aggregate_oos"]["all"]["count"],
                "trades_per_30_days": item["aggregate_oos"]["trades_per_30_days"],
                "mean_net_return": item["aggregate_oos"]["all"]["mean_net_return"],
                "profit_factor": item["aggregate_oos"]["all"]["profit_factor"],
                "positive_folds": item["aggregate_oos"]["positive_folds"],
                "accepted": item["acceptance_passed"],
            }
            for item in leaderboard[:10]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
