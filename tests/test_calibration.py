from __future__ import annotations

import unittest

import numpy as np

from btc_perp.calibration import PriorOOSIsotonic


class PriorOOSIsotonicTests(unittest.TestCase):
    def test_does_not_calibrate_before_prior_oos_data_exists(self) -> None:
        calibrator = PriorOOSIsotonic(minimum_rows=2)
        current = np.array([[0.2, 0.5, 0.3]])
        np.testing.assert_allclose(calibrator.transform(current, np.array([0, 1, 2])), current)

    def test_uses_only_observed_prior_oos_rows(self) -> None:
        calibrator = PriorOOSIsotonic(minimum_rows=2)
        prior = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
        calibrator.observe(prior, np.array([0, 2]))
        calibrated = calibrator.transform(np.array([[0.8, 0.1, 0.1]]), np.array([0, 1, 2]))
        self.assertEqual(calibrator.rows, 2)
        self.assertAlmostEqual(float(calibrated.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
