"""Leakage-aware OOS evaluation primitives for strategy research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: int
    train_end: int
    purge_start: int
    purge_end: int
    test_start: int
    test_end: int

def chronological_train_validation_indices(
    rows: int,
    *,
    horizon_bars: int,
    validation_fraction: float = 0.2,
) -> tuple[int, int]:
    """Return fit end and validation start with an h+1 purge gap."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_start = int(rows * (1.0 - validation_fraction))
    purge_bars = horizon_bars + 1
    fit_end = validation_start - purge_bars
    if fit_end <= 0 or validation_start >= rows:
        raise ValueError("split does not leave fit, purge, and validation rows")
    return fit_end, validation_start


def purged_walk_forward_splits(
    rows: int, *, train_bars: int, test_bars: int, purge_bars: int, step_bars: int | None = None
) -> list[WalkForwardFold]:
    """Return chronological folds with an explicit no-trade purge gap.

    The purge must be at least the label horizon so no training target can
    observe an OOS test price.
    """

    if min(rows, train_bars, test_bars) <= 0 or purge_bars < 0:
        raise ValueError("rows, train_bars, and test_bars must be positive; purge_bars cannot be negative")
    step = step_bars or test_bars
    if step <= 0:
        raise ValueError("step_bars must be positive")
    folds: list[WalkForwardFold] = []
    train_start = 0
    while True:
        train_end = train_start + train_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_bars
        if test_end > rows:
            break
        folds.append(WalkForwardFold(train_start, train_end, train_end, test_start, test_start, test_end))
        train_start += step
    return folds


def calibration_metrics(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Return Brier score and expected calibration error for binary forecasts."""

    probability = np.asarray(probabilities, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    if probability.ndim != 1 or outcome.ndim != 1 or len(probability) != len(outcome) or len(probability) == 0:
        raise ValueError("probabilities and outcomes must be equal-length non-empty vectors")
    if bins <= 0 or np.any((probability < 0) | (probability > 1)) or np.any((outcome < 0) | (outcome > 1)):
        raise ValueError("probabilities/outcomes must be in [0, 1] and bins positive")
    brier = float(np.mean((probability - outcome) ** 2))
    bucket = np.minimum((probability * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            ece += float(mask.mean() * abs(probability[mask].mean() - outcome[mask].mean()))
    return {"brier_score": brier, "expected_calibration_error": ece}

def expected_return_calibration(
    expected_returns: np.ndarray, realized_returns: np.ndarray, bins: int = 10
) -> dict[str, float]:
    """Measure whether OOS expected returns agree with realized next-open returns."""

    expected = np.asarray(expected_returns, dtype=float)
    realized = np.asarray(realized_returns, dtype=float)
    if expected.ndim != 1 or realized.ndim != 1 or len(expected) != len(realized) or len(expected) == 0:
        raise ValueError("expected_returns and realized_returns must be equal-length non-empty vectors")
    if bins <= 0 or not np.isfinite(expected).all() or not np.isfinite(realized).all():
        raise ValueError("returns must be finite and bins positive")
    error = expected - realized
    ordering = np.argsort(expected)
    chunks = np.array_split(ordering, bins)
    bin_error = [abs(float(error[index].mean())) for index in chunks if len(index)]
    return {
        "expected_return_mae": float(np.mean(np.abs(error))),
        "expected_return_rmse": float(np.sqrt(np.mean(error**2))),
        "expected_return_calibration_error": float(np.mean(bin_error)),
    }
