import unittest

import numpy as np

from btc_perp.gate_ablation import (
    calibrated_gate_threshold,
    evaluate_non_overlapping_gate,
    side_metrics,
    validate_fold_layout,
)


class GateThresholdTests(unittest.TestCase):
    def test_quantile_threshold_uses_calibration_predictions_only(self) -> None:
        calibration_expected_net = np.array(
            [
                [0.01, -0.20],
                [0.02, -0.30],
                [0.03, -0.40],
                [0.04, -0.50],
            ]
        )
        unrelated_test_predictions = np.array([[1000.0, 900.0]])

        threshold = calibrated_gate_threshold(
            calibration_expected_net,
            safety_margin_bps=1.0,
            top_fraction=0.25,
        )
        actions, _ = evaluate_non_overlapping_gate(
            unrelated_test_predictions,
            np.zeros_like(unrelated_test_predictions),
            threshold=threshold,
            horizon_bars=1,
        )

        self.assertAlmostEqual(threshold, 0.0325)
        np.testing.assert_array_equal(actions, [1])

    def test_threshold_is_maximum_of_zero_safety_and_quantile(self) -> None:
        safety_dominates = calibrated_gate_threshold(
            np.array([[0.001, -0.01], [0.002, -0.02]]),
            safety_margin_bps=20.0,
            top_fraction=0.5,
        )
        zero_dominates = calibrated_gate_threshold(
            np.array([[-0.04, -0.05], [-0.02, -0.03]]),
            safety_margin_bps=0.0,
            top_fraction=0.5,
        )

        self.assertAlmostEqual(safety_dominates, 0.002)
        self.assertEqual(zero_dominates, 0.0)


class GateEvaluationTests(unittest.TestCase):
    def test_horizon_blocks_every_row_until_first_non_overlapping_index(self) -> None:
        expected_net = np.array(
            [
                [0.80, 0.10],
                [0.10, 0.90],
                [0.70, 0.20],
                [0.10, 0.85],
                [0.75, 0.20],
                [0.10, 0.95],
                [0.65, 0.10],
            ]
        )

        actions, metrics = evaluate_non_overlapping_gate(
            expected_net,
            np.zeros_like(expected_net),
            threshold=0.5,
            horizon_bars=2,
        )

        np.testing.assert_array_equal(actions, [1, 0, 0, -1, 0, 0, 1])
        self.assertEqual(metrics["flat_rows"], 4)

    def test_long_short_metrics_and_profit_factor_use_executed_side(self) -> None:
        expected_net = np.array(
            [
                [0.9, 0.1],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.1, 0.9],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.8, 0.1],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.1, 0.8],
            ]
        )
        actual_net = np.zeros_like(expected_net)
        actual_net[0] = [0.10, -9.0]
        actual_net[3] = [-9.0, 0.20]
        actual_net[6] = [-0.05, 9.0]
        actual_net[9] = [9.0, -0.10]

        _, metrics = evaluate_non_overlapping_gate(
            expected_net,
            actual_net,
            threshold=0.5,
            horizon_bars=2,
        )

        self.assertEqual(metrics["long"]["count"], 2)
        self.assertEqual(metrics["short"]["count"], 2)
        self.assertEqual(metrics["all"]["count"], 4)
        self.assertAlmostEqual(metrics["long"]["total_uncompounded_net_return"], 0.05)
        self.assertAlmostEqual(metrics["short"]["total_uncompounded_net_return"], 0.10)
        self.assertAlmostEqual(metrics["long"]["profit_factor"], 2.0)
        self.assertAlmostEqual(metrics["short"]["profit_factor"], 2.0)
        self.assertAlmostEqual(metrics["all"]["profit_factor"], 2.0)

    def test_profit_factor_is_none_without_losses(self) -> None:
        metrics = side_metrics(np.array([0.10, 0.00, 0.20]))

        self.assertIsNone(metrics["profit_factor"])
        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["win_rate"], 2.0 / 3.0)


class GateValidationTests(unittest.TestCase):
    def test_invalid_calibration_parameters_raise(self) -> None:
        valid = np.array([[0.01, 0.02]])
        invalid_cases = (
            (np.array([]), 0.0, 0.5),
            (np.array([0.01, 0.02]), 0.0, 0.5),
            (np.array([[np.nan, 0.02]]), 0.0, 0.5),
            (valid, -1.0, 0.5),
            (valid, 0.0, 0.0),
            (valid, 0.0, 1.01),
        )

        for predictions, safety_margin_bps, top_fraction in invalid_cases:
            with self.subTest(
                predictions=predictions,
                safety_margin_bps=safety_margin_bps,
                top_fraction=top_fraction,
            ):
                with self.assertRaises(ValueError):
                    calibrated_gate_threshold(
                        predictions,
                        safety_margin_bps=safety_margin_bps,
                        top_fraction=top_fraction,
                    )

    def test_invalid_gate_evaluation_parameters_raise(self) -> None:
        valid = np.zeros((2, 2))

        with self.assertRaisesRegex(ValueError, "share shape"):
            evaluate_non_overlapping_gate(
                valid,
                np.zeros((3, 2)),
                threshold=0.0,
                horizon_bars=1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_non_overlapping_gate(
                np.array([[np.inf, 0.0]]),
                np.zeros((1, 2)),
                threshold=0.0,
                horizon_bars=1,
            )
        with self.assertRaisesRegex(ValueError, "threshold cannot be negative"):
            evaluate_non_overlapping_gate(
                valid,
                valid,
                threshold=-0.01,
                horizon_bars=1,
            )
        with self.assertRaisesRegex(ValueError, "horizon_bars must be positive"):
            evaluate_non_overlapping_gate(
                valid,
                valid,
                threshold=0.0,
                horizon_bars=0,
            )

    def test_invalid_fold_layouts_raise(self) -> None:
        base = {
            "rows": 1_000,
            "train_bars": 300,
            "calibration_bars": 100,
            "test_bars": 100,
            "step_bars": 100,
            "purge_bars": 50,
            "fold_count": 3,
        }

        cases = (
            ({"purge_bars": 0}, "positive"),
            ({"train_bars": 249}, "fewer than 100 fit rows"),
            ({"rows": 649}, "does not fit available rows"),
        )
        for changes, message in cases:
            params = {**base, **changes}
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    validate_fold_layout(**params)


if __name__ == "__main__":
    unittest.main()
