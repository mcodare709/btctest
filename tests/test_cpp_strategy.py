from __future__ import annotations

import unittest

from dashboard.cpp_core import MarketFeatures, StrategyCore


class StrategyCoreTests(unittest.TestCase):
    def test_neutral_market_is_cost_blocked(self) -> None:
        decision = StrategyCore().evaluate(MarketFeatures(0.0, 0.0, 0.0, 5.0, 0.0), 0, 0)
        self.assertEqual(decision.direction, 0)
        self.assertLess(decision.expected_edge_bps, 0.0)

    def test_strong_signal_requires_three_confirmations(self) -> None:
        core = StrategyCore()
        features = MarketFeatures(0.002, 0.8, 0.8, 20.0, 1.0)
        first = core.evaluate(features, 0, 0)
        second = core.evaluate(features, first.candidate_direction, first.candidate_count)
        third = core.evaluate(features, second.candidate_direction, second.candidate_count)
        self.assertEqual(first.direction, 1)
        self.assertEqual(first.confirmed_direction, 0)
        self.assertEqual(third.confirmed_direction, 1)


if __name__ == "__main__":
    unittest.main()
