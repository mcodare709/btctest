"""OOS-only walk-forward backtests and simple deterministic benchmarks."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from .calibration import PriorOOSIsotonic
from .config import BacktestConfig
from .data import validate_market_data
from .engine import run_backtest
from .evaluation import calibration_metrics, expected_return_calibration, WalkForwardFold, purged_walk_forward_splits
from .metrics import calculate_metrics
from .ml_model import _catboost, make_supervised_dataset
from .protocol import net_horizon_returns
from .signals import cost_aware_baseline_signal


def engine_cost_aware_benchmarks(
    frame: pd.DataFrame,
    *,
    config: BacktestConfig,
    oos_start: pd.Timestamp,
    seed: int,
) -> dict[str, dict[str, float | int | bool | None]]:
    """Replay deterministic comparison rules through the same cost-aware engine."""

    benchmark_config = replace(config, max_holding_bars=max(len(frame) + 1, config.max_holding_bars), cost_aware_filter=False)
    rng = np.random.default_rng(seed)
    random_directions = pd.Series(rng.choice(np.array([-1, 1]), size=len(frame)), index=frame.index)

    def buy_and_hold(_: pd.Series, __: BacktestConfig) -> tuple[float, int]:
        return 1.0, 1

    def random_sign(row: pd.Series, _: BacktestConfig) -> tuple[float, int]:
        return 0.5, int(random_directions.loc[row.name])

    def momentum(row: pd.Series, _: BacktestConfig) -> tuple[float, int]:
        value = row.get("return_1", np.nan)
        return 0.5, 1 if value > 0 else -1 if value < 0 else 0

    reports: dict[str, dict[str, float | int | bool | None]] = {}
    for name, signal in {"buy_and_hold": buy_and_hold, "random_sign": random_sign, "simple_momentum": momentum}.items():
        result = run_backtest(frame, config=benchmark_config, signal_fn=signal)
        equity = result.equity_curve.loc[result.equity_curve.index >= oos_start]
        trades = result.trades.loc[result.trades["entry_time"] >= oos_start] if not result.trades.empty else result.trades
        reports[name] = calculate_metrics(equity, trades, config.initial_equity, config.annualization_days)
    return reports


def realized_class_net_return(labels: np.ndarray, long_net: np.ndarray, short_net: np.ndarray) -> np.ndarray:
    """Map class targets to executable returns; Flat is explicitly no-trade."""

    labels = np.asarray(labels, dtype=int)
    return np.select([labels == 0, labels == 2], [short_net, long_net], default=0.0)

def benchmark_returns(frame: pd.DataFrame, seed: int = 42) -> dict[str, float]:
    """Return OOS asset, random-sign, and one-bar momentum benchmark returns.

    These are deliberately simple return benchmarks, not claims of executable
    profitability; strategy comparisons must use the cost-aware engine result.
    """

    close = frame["close"].astype(float)
    returns = close.pct_change().dropna()
    if returns.empty:
        return {"buy_and_hold": 0.0, "random_sign": 0.0, "simple_momentum": 0.0}
    rng = np.random.default_rng(seed)
    random_sign = rng.choice(np.array([-1.0, 1.0]), size=len(returns))
    momentum_sign = np.sign(returns.shift(1).fillna(0.0)).to_numpy()
    return {
        "buy_and_hold": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "random_sign": float(np.prod(1.0 + returns.to_numpy() * random_sign) - 1.0),
        "simple_momentum": float(np.prod(1.0 + returns.to_numpy() * momentum_sign) - 1.0),
    }


def run_purged_oos_backtest(
    market_data: pd.DataFrame,
    *,
    config: BacktestConfig,
    train_bars: int,
    test_bars: int,
    purge_bars: int,
    warmup_bars: int = 40,
) -> dict[str, Any]:
    """Run baseline only on each fold's OOS region and report no in-sample PnL."""

    if not isinstance(market_data.index, pd.DatetimeIndex):
        market_data = validate_market_data(market_data)
    folds = purged_walk_forward_splits(
        len(market_data), train_bars=train_bars, test_bars=test_bars, purge_bars=purge_bars
    )
    if not folds:
        raise ValueError("not enough rows for one purged walk-forward fold")
    reports: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        reports.append(_run_fold(market_data, fold, fold_index, config, warmup_bars))
    return {"folds": reports, "fold_count": len(reports)}


def _run_fold(
    market_data: pd.DataFrame,
    fold: WalkForwardFold,
    fold_index: int,
    config: BacktestConfig,
    warmup_bars: int,
) -> dict[str, Any]:
    context_start = max(0, fold.test_start - warmup_bars)
    frame = market_data.iloc[context_start : fold.test_end].copy()
    oos_start = market_data.index[fold.test_start]

    def oos_signal(row: pd.Series, signal_config: BacktestConfig) -> tuple[float, int]:
        if row.name < oos_start:
            return 0.5, 0
        probability, direction, _ = cost_aware_baseline_signal(row, signal_config)
        return probability, direction

    result = run_backtest(frame, config=config, signal_fn=oos_signal)
    equity = result.equity_curve.loc[result.equity_curve.index >= oos_start]
    trades = result.trades.loc[result.trades["entry_time"] >= oos_start].copy() if not result.trades.empty else result.trades
    summary = calculate_metrics(equity, trades, config.initial_equity, config.annualization_days)
    return {
        "fold": fold_index,
        "split": asdict(fold),
        "oos_start": oos_start.isoformat(),
        "oos_end": market_data.index[fold.test_end - 1].isoformat(),
        "summary": summary,
        "benchmarks": engine_cost_aware_benchmarks(frame, config=config, oos_start=oos_start, seed=42 + fold_index),
    }

def run_parameter_sensitivity(
    market_data: pd.DataFrame,
    *,
    config: BacktestConfig,
    parameter_sets: Mapping[str, Mapping[str, Any]],
    train_bars: int,
    test_bars: int,
    purge_bars: int,
    warmup_bars: int = 40,
) -> dict[str, dict[str, Any]]:
    """Evaluate named BacktestConfig variants using the same purged OOS folds.

    This does not select a winner. It exposes whether the observed OOS result
    remains similar across nearby, pre-declared assumptions.
    """

    allowed_fields = set(BacktestConfig.__dataclass_fields__)
    results: dict[str, dict[str, Any]] = {}
    for name, overrides in parameter_sets.items():
        unknown = sorted(set(overrides) - allowed_fields)
        if unknown:
            raise ValueError(f"{name}: unknown BacktestConfig fields: {unknown}")
        report = run_purged_oos_backtest(
            market_data,
            config=replace(config, **dict(overrides)),
            train_bars=train_bars,
            test_bars=test_bars,
            purge_bars=purge_bars,
            warmup_bars=warmup_bars,
        )
        summaries = [fold["summary"] for fold in report["folds"]]
        total_returns = np.array([float(summary["total_return"]) for summary in summaries])
        drawdowns = np.array([float(summary["max_drawdown"]) for summary in summaries])
        results[name] = {
            "overrides": dict(overrides),
            "fold_count": report["fold_count"],
            "mean_fold_return": float(total_returns.mean()),
            "median_fold_return": float(np.median(total_returns)),
            "profitable_fold_fraction": float((total_returns > 0).mean()),
            "mean_fold_max_drawdown": float(drawdowns.mean()),
            "oos_report": report,
        }
    return results



def classification_report(y_true: np.ndarray, probabilities: np.ndarray, class_ids: np.ndarray) -> dict[str, Any]:
    """Standard multiclass OOS metrics plus confidence reliability buckets."""
    truth = np.asarray(y_true, dtype=int)
    predicted = class_ids[np.argmax(probabilities, axis=1)]
    labels = np.array([0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    confidence = probabilities.max(axis=1)
    buckets = []
    for low in np.arange(0.5, 1.0, 0.1):
        high = min(low + 0.1, 1.0)
        mask = (confidence >= low) & ((confidence < high) if high < 1.0 else (confidence <= high))
        if mask.any():
            buckets.append({"range": f"{low:.2f}-{high:.2f}", "count": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()), "accuracy": float((predicted[mask] == truth[mask]).mean())})
    return {
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).astype(int).tolist(),
        "class_distribution": {str(label): int((truth == label).sum()) for label in labels},
        "prediction_distribution": {str(label): int((predicted == label).sum()) for label in labels},
        "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)},
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "confidence_buckets": buckets,
    }
def run_catboost_purged_oos_evaluation(
    market_data: pd.DataFrame,
    *,
    horizon_bars: int,
    train_bars: int,
    test_bars: int,
    purge_bars: int,
    iterations: int = 100,
    depth: int = 6,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    """Train one classifier per fold and report only future OOS predictions.

    Each fitted model sees only a fold's training segment. The purge must be
    at least the label horizon, so training labels cannot observe OOS prices.
    """

    if purge_bars < horizon_bars:
        raise ValueError("purge_bars must be at least horizon_bars")
    frame = market_data if isinstance(market_data.index, pd.DatetimeIndex) else validate_market_data(market_data)
    folds = purged_walk_forward_splits(len(frame), train_bars=train_bars, test_bars=test_bars, purge_bars=purge_bars)
    if not folds:
        raise ValueError("not enough rows for one purged walk-forward fold")
    all_x, all_y = make_supervised_dataset(frame, horizon_bars=horizon_bars)
    all_net_returns = net_horizon_returns(frame, horizon_bars, fee_rate=0.0004, slippage_bps=1.0, default_spread_bps=2.0)
    classifier = _catboost()
    reports: list[dict[str, Any]] = []
    calibrator = PriorOOSIsotonic()
    for number, fold in enumerate(folds):
        train_frame = frame.iloc[fold.train_start:fold.train_end]
        train_x, train_y = make_supervised_dataset(train_frame, horizon_bars=horizon_bars, feature_names=all_x.columns)
        test_start = frame.index[fold.test_start]
        test_end = frame.index[fold.test_end - 1]
        test_x = all_x.loc[(all_x.index >= test_start) & (all_x.index <= test_end)]
        test_y = all_y.loc[test_x.index]
        if len(train_x) < 100 or train_y.nunique() < 3 or test_x.empty:
            reports.append({"fold": number, "status": "skipped", "reason": "insufficient rows or target classes", "split": asdict(fold)})
            continue
        model = classifier(loss_function="MultiClass", iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=42, verbose=False, allow_writing_files=False)
        model.fit(train_x, train_y, verbose=False)
        probabilities = np.asarray(model.predict_proba(test_x), dtype=float)
        class_ids = np.asarray(model.classes_, dtype=int)
        calibration_rows = calibrator.rows
        calibrated = calibrator.transform(probabilities, class_ids)
        probability_by_class = {label: calibrated[:, position] for position, label in enumerate(class_ids)}
        up_probability = probability_by_class.get(2, np.zeros(len(test_x)))
        test_net = all_net_returns.loc[test_x.index]
        train_net = all_net_returns.loc[train_x.index]
        long_net = test_net["long_net_return"].to_numpy(dtype=float)
        short_net = test_net["short_net_return"].to_numpy(dtype=float)
        realized_return = realized_class_net_return(test_y.to_numpy(), long_net, short_net)
        class_mean_return = {
            label: float(train_net["short_net_return"][train_y == label].mean()) if label == 0 else
            float(train_net["long_net_return"][train_y == label].mean()) if label == 2 else 0.0
            for label in class_ids
        }
        expected_return = sum(probability_by_class.get(label, 0.0) * class_mean_return[label] for label in class_ids)
        context_start = max(0, fold.test_start - 40)
        replay_frame = frame.iloc[context_start:fold.test_end]
        def model_oos_signal(row: pd.Series, signal_config: BacktestConfig) -> tuple[float, int]:
            if row.name < test_start:
                return 0.5, 0
            row_x = pd.DataFrame([{name: row.get(name, np.nan) for name in test_x.columns}])
            raw = np.asarray(model.predict_proba(row_x), dtype=float)
            calibrated_row = calibrator.transform(raw, class_ids)[0]
            probability = {label: calibrated_row[position] for position, label in enumerate(class_ids)}
            direction = 1 if probability.get(2, 0.0) >= signal_config.long_probability_threshold and probability.get(2, 0.0) > probability.get(0, 0.0) else -1 if probability.get(0, 0.0) >= 1.0 - signal_config.short_probability_threshold and probability.get(0, 0.0) > probability.get(2, 0.0) else 0
            return float(probability.get(2, 0.0)), direction
        replay = run_backtest(replay_frame, config=BacktestConfig(max_holding_bars=horizon_bars), signal_fn=model_oos_signal)
        oos_equity = replay.equity_curve.loc[replay.equity_curve.index >= test_start]
        oos_trades = replay.trades.loc[replay.trades["entry_time"] >= test_start] if not replay.trades.empty else replay.trades
        trading_summary = calculate_metrics(oos_equity, oos_trades, 100.0)
        reports.append({
            "fold": number,
            "status": "ok",
            "split": asdict(fold),
            "train_rows": len(train_x),
            "oos_rows": len(test_x),
            "classification": classification_report(test_y.to_numpy(), calibrated, class_ids),
            "calibration_source_oos_rows": calibration_rows,
            "probability_calibration": calibration_metrics(up_probability, (test_y.to_numpy() == 2).astype(float)),
            "feature_importance": {name: float(value) for name, value in zip(test_x.columns, model.get_feature_importance())},
            "expected_return_calibration": expected_return_calibration(expected_return, realized_return),
            "trading": trading_summary,
        })
        calibrator.observe(probabilities, test_y.to_numpy())
    return {"folds": reports, "fold_count": len(reports), "horizon_bars": horizon_bars, "purge_bars": purge_bars}
