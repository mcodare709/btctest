from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_perp.availability import feature_availability_report


class AvailabilityTests(unittest.TestCase):
    def test_nan_and_zero_are_reported_separately(self) -> None:
        index = pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC")
        frame = pd.DataFrame({"x": [0.0, np.nan, 2.0, np.nan, np.nan]}, index=index)
        row = feature_availability_report(frame).iloc[0]
        self.assertAlmostEqual(row["missing_rate"], 3 / 5)
        self.assertAlmostEqual(row["zero_rate"], 1 / 5)
        self.assertEqual(row["unique_count"], 2)
        self.assertEqual(row["coverage_start"], index[0].isoformat())
        self.assertEqual(row["coverage_end"], index[2].isoformat())
        self.assertEqual(row["continuous_missing_periods"], "0 days 00:02:00")


if __name__ == "__main__":
    unittest.main()
