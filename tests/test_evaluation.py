from __future__ import annotations

import unittest

import numpy as np

from btc_perp.evaluation import calibration_metrics, purged_walk_forward_splits


class EvaluationTests(unittest.TestCase):
    def test_walk_forward_has_no_train_test_overlap(self) -> None:
        folds = purged_walk_forward_splits(100, train_bars=30, test_bars=20, purge_bars=10)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(fold.purge_end, fold.test_start)
            self.assertLessEqual(fold.train_end, fold.purge_start)
            self.assertLess(fold.purge_end, fold.test_end)

    def test_calibration_metrics(self) -> None:
        result = calibration_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertEqual(result["brier_score"], 0.0)
        self.assertEqual(result["expected_calibration_error"], 0.0)
