"""Leak-free threshold calibration and trading metrics for gate ablations."""

from __future__ import annotations

import numpy as np

from .return_policy import non_overlapping_actions


def calibrated_gate_threshold(
    calibration_expected_net: np.ndarray,
    *,
    safety_margin_bps: float,
    top_fraction: float | None,
) -> float:
    values = np.asarray(calibration_expected_net, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("calibration_expected_net must be finite with shape (rows, 2)")
    if len(values) == 0:
        raise ValueError("calibration_expected_net cannot be empty")
    if safety_margin_bps < 0:
        raise ValueError("safety_margin_bps cannot be negative")
    safety = safety_margin_bps / 10_000.0
    if top_fraction is None:
        return safety
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    quantile = float(np.quantile(np.max(values, axis=1), 1.0 - top_fraction))
    return max(0.0, safety, quantile)


def side_metrics(pnl: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(pnl, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("pnl must be finite")
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if values.size:
        equity = np.cumprod(1.0 + values)
        path = np.concatenate(([1.0], equity))
        drawdown = 1.0 - path / np.maximum.accumulate(path)
        compounded = float(equity[-1] - 1.0)
        max_drawdown = float(drawdown.max())
    else:
        compounded = 0.0
        max_drawdown = 0.0
    return {
        "count": int(values.size),
        "win_rate": float((values > 0).mean()) if values.size else None,
        "mean_net_return": float(values.mean()) if values.size else None,
        "median_net_return": float(np.median(values)) if values.size else None,
        "total_uncompounded_net_return": float(values.sum()),
        "compounded_net_return": compounded,
        "max_drawdown": max_drawdown,
        "profit_factor": gains / losses if losses > 0 else None,
    }


def evaluate_non_overlapping_gate(
    expected_net: np.ndarray,
    actual_net: np.ndarray,
    *,
    threshold: float,
    horizon_bars: int,
) -> tuple[np.ndarray, dict[str, object]]:
    expected = np.asarray(expected_net, dtype=float)
    actual = np.asarray(actual_net, dtype=float)
    if expected.shape != actual.shape or expected.ndim != 2 or expected.shape[1] != 2:
        raise ValueError("expected_net and actual_net must share shape (rows, 2)")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("expected_net and actual_net must be finite")
    actions = non_overlapping_actions(expected, threshold=threshold, horizon_bars=horizon_bars)
    long_pnl = actual[actions == 1, 0]
    short_pnl = actual[actions == -1, 1]
    all_pnl = np.concatenate((long_pnl, short_pnl))
    ordered_pnl = np.where(actions == 1, actual[:, 0], np.where(actions == -1, actual[:, 1], np.nan))
    all_pnl = ordered_pnl[actions != 0]
    return actions, {
        "all": side_metrics(all_pnl),
        "long": side_metrics(long_pnl),
        "short": side_metrics(short_pnl),
        "flat_rows": int((actions == 0).sum()),
    }


def validate_fold_layout(
    rows: int,
    *,
    train_bars: int,
    calibration_bars: int,
    test_bars: int,
    step_bars: int,
    purge_bars: int,
    fold_count: int,
) -> None:
    if min(rows, train_bars, calibration_bars, test_bars, step_bars, purge_bars, fold_count) <= 0:
        raise ValueError("fold parameters must be positive")
    fit_bars = train_bars - calibration_bars - purge_bars
    if fit_bars < 100:
        raise ValueError("training window leaves fewer than 100 fit rows")
    final_test_end = (fold_count - 1) * step_bars + train_bars + purge_bars + test_bars
    if final_test_end > rows:
        raise ValueError("fold layout does not fit available rows")
