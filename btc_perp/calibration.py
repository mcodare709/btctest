"""Chronological probability calibration for multiclass model OOS forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class PriorOOSIsotonic:
    """Fit one-vs-rest calibrators only from forecasts of earlier OOS folds."""

    def __init__(self, minimum_rows: int = 100) -> None:
        self.minimum_rows = minimum_rows
        self._probabilities: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    @property
    def rows(self) -> int:
        return sum(len(item) for item in self._targets)

    def transform(self, probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
        raw = np.asarray(probabilities, dtype=float)
        if self.rows < self.minimum_rows:
            return raw.copy()
        prior_probabilities = np.vstack(self._probabilities)
        prior_targets = np.concatenate(self._targets)
        calibrated = raw.copy()
        for column, label in enumerate(classes):
            outcome = (prior_targets == label).astype(int)
            if outcome.min() != outcome.max():
                calibrated[:, column] = IsotonicRegression(out_of_bounds="clip").fit(
                    prior_probabilities[:, column], outcome
                ).predict(raw[:, column])
        denominator = calibrated.sum(axis=1, keepdims=True)
        return np.divide(calibrated, denominator, out=raw.copy(), where=denominator > 0)

    def observe(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        self._probabilities.append(np.asarray(probabilities, dtype=float).copy())
        self._targets.append(np.asarray(targets, dtype=int).copy())
