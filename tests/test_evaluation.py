from __future__ import annotations

import unittest

import numpy as np

from btc_perp.evaluation import calibration_metrics, chronological_train_validation_indices, expected_return_calibration, purged_walk_forward_splits


class EvaluationTests(unittest.TestCase):
    def test_walk_forward_has_no_train_test_overlap(self) -> None:
        folds = purged_walk_forward_splits(100, train_bars=30, test_bars=20, purge_bars=10)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(fold.purge_end, fold.test_start)
            self.assertLessEqual(fold.train_end, fold.purge_start)
            self.assertLess(fold.purge_end, fold.test_end)

    def test_train_validation_split_purges_next_open_horizon(self) -> None:
        fit_end, validation_start = chronological_train_validation_indices(
            100,
            horizon_bars=10,
            validation_fraction=0.2,
        )
        self.assertEqual(validation_start, 80)
        self.assertEqual(fit_end, 69)
        self.assertEqual(validation_start - fit_end, 11)

    def test_calibration_metrics(self) -> None:
        result = calibration_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertEqual(result["brier_score"], 0.0)
        self.assertEqual(result["expected_calibration_error"], 0.0)
    def test_expected_return_calibration(self) -> None:
        result = expected_return_calibration(np.array([0.01, -0.01]), np.array([0.01, -0.01]), bins=2)
        self.assertEqual(result["expected_return_mae"], 0.0)
        self.assertEqual(result["expected_return_rmse"], 0.0)
