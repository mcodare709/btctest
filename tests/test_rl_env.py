import unittest

import numpy as np
import pandas as pd

from btc_perp.rl_env import PortfolioAction, RLPortfolioConfig, RLPortfolioEnv


LONG_ACTIONS = (PortfolioAction.LONG_1X, PortfolioAction.LONG_3X, PortfolioAction.LONG_5X)
SHORT_ACTIONS = (PortfolioAction.SHORT_1X, PortfolioAction.SHORT_3X, PortfolioAction.SHORT_5X)
ENTRY_ACTIONS = LONG_ACTIONS + SHORT_ACTIONS


def market(
    prices: list[float],
    *,
    frequency: str = "1min",
    **columns: list[float],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": [price * 1.001 for price in prices],
            "low": [price * 0.999 for price in prices],
            "close": prices,
            **columns,
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq=frequency, tz="UTC"),
    )
    return frame


class RLPortfolioEnvTests(unittest.TestCase):
    def test_expected_return_gate_masks_only_strict_winning_side(self) -> None:
        cases = (
            ("below_threshold", 0.0004, 0.0003, False, False),
            ("long_wins", 0.0010, 0.0007, True, False),
            ("short_wins", 0.0007, 0.0010, False, True),
            ("tie", 0.0010, 0.0010, False, False),
        )
        config = RLPortfolioConfig(expected_return_gate_enabled=True, min_expected_edge_bps=5.0)

        for name, long_edge, short_edge, allow_long, allow_short in cases:
            with self.subTest(name=name):
                frame = market(
                    [100.0] * 3,
                    long_expected_net_return=[long_edge] * 3,
                    short_expected_net_return=[short_edge] * 3,
                )
                env = RLPortfolioEnv(frame, config=config)
                env.reset(episode_max_leverage=3.0)

                mask = env.valid_action_mask()

                self.assertTrue(mask[PortfolioAction.HOLD])
                self.assertFalse(mask[PortfolioAction.CLOSE_ALL])
                self.assertEqual(mask[list(LONG_ACTIONS)].tolist(), [allow_long] * len(LONG_ACTIONS))
                self.assertEqual(mask[list(SHORT_ACTIONS)].tolist(), [allow_short] * len(SHORT_ACTIONS))

    def test_expected_return_gate_and_rate_limit_both_apply_to_entry_masks(self) -> None:
        frame = market(
            [100.0] * 4,
            long_expected_net_return=[0.0010] * 4,
            short_expected_net_return=[0.0001] * 4,
        )
        config = RLPortfolioConfig(
            expected_return_gate_enabled=True,
            min_expected_edge_bps=5.0,
            max_entries_per_window=1,
            entry_rate_window_minutes=5.0,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=3.0)

        edge_only_mask = env.valid_action_mask()
        self.assertTrue(edge_only_mask[list(LONG_ACTIONS)].all())
        self.assertFalse(edge_only_mask[list(SHORT_ACTIONS)].any())

        *_, opened = env.step(PortfolioAction.LONG_1X)
        self.assertTrue(opened["opened_lot"])
        combined_mask = env.valid_action_mask()
        self.assertFalse(combined_mask[list(ENTRY_ACTIONS)].any())
        self.assertTrue(combined_mask[PortfolioAction.HOLD])
        self.assertTrue(combined_mask[PortfolioAction.CLOSE_ALL])

    def test_expected_return_gate_does_not_block_close_all(self) -> None:
        frame = market(
            [100.0] * 4,
            long_expected_net_return=[0.0010, 0.0, 0.0, 0.0],
            short_expected_net_return=[0.0] * 4,
        )
        config = RLPortfolioConfig(
            expected_return_gate_enabled=True,
            min_expected_edge_bps=5.0,
            max_holding_bars=100,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=3.0)
        env.step(PortfolioAction.LONG_1X)

        mask = env.valid_action_mask()
        self.assertFalse(mask[list(ENTRY_ACTIONS)].any())
        self.assertTrue(mask[PortfolioAction.CLOSE_ALL])

        *_, closed = env.step(PortfolioAction.CLOSE_ALL)
        self.assertFalse(env.lots)
        self.assertEqual(env.trades[-1]["exit_reason"], "agent_close")
        self.assertFalse(closed["action_blocked"])

    def test_expected_return_gate_does_not_prevent_time_exit(self) -> None:
        frame = market(
            [100.0] * 4,
            long_expected_net_return=[0.0010, 0.0, 0.0, 0.0],
            short_expected_net_return=[0.0] * 4,
        )
        config = RLPortfolioConfig(
            expected_return_gate_enabled=True,
            min_expected_edge_bps=5.0,
            max_holding_bars=1,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=3.0)
        env.step(PortfolioAction.LONG_1X)

        *_, info = env.step(PortfolioAction.LONG_1X)

        self.assertFalse(info["opened_lot"])
        self.assertFalse(env.lots)
        self.assertEqual(len(env.trades), 1)
        self.assertEqual(env.trades[0]["exit_reason"], "time_exit")

    def test_open_rate_limit_blocks_sixth_fill_as_hold_until_window_rolls(self) -> None:
        config = RLPortfolioConfig(
            max_holding_bars=100,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(market([100.0] * 20, frequency="30s"), config=config)
        env.reset(episode_max_leverage=5.0)

        for _ in range(5):
            *_, info = env.step(PortfolioAction.LONG_1X)
            self.assertTrue(info["opened_lot"])

        *_, blocked = env.step(PortfolioAction.LONG_1X)
        self.assertFalse(blocked["opened_lot"])
        self.assertEqual(len(env.lots), 5)
        self.assertTrue(blocked["action_blocked"])
        self.assertEqual(blocked["action_block_reason"], "open_rate_limit")
        self.assertEqual(blocked["blocked_action_count"], 1)
        self.assertEqual(blocked["effective_action"], int(PortfolioAction.HOLD))

        for _ in range(5):
            env.step(PortfolioAction.HOLD)
        *_, allowed = env.step(PortfolioAction.LONG_1X)
        self.assertTrue(allowed["opened_lot"])
        self.assertFalse(allowed["action_blocked"])
        self.assertEqual(allowed["blocked_action_count"], 1)
        self.assertEqual(len(env.lots), 6)

    def test_agent_closes_do_not_consume_open_rate_limit(self) -> None:
        config = RLPortfolioConfig(
            max_holding_bars=100,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(market([100.0] * 20, frequency="20s"), config=config)
        env.reset(episode_max_leverage=5.0)

        for fill_number in range(1, 6):
            *_, opened = env.step(PortfolioAction.LONG_1X)
            self.assertTrue(opened["opened_lot"])
            *_, closed = env.step(PortfolioAction.CLOSE_ALL)
            self.assertFalse(closed["action_blocked"])
            self.assertEqual(closed["blocked_action_count"], 0)
            self.assertFalse(env.lots)
            self.assertEqual(len(env.trades), fill_number)
            self.assertEqual(env.trades[-1]["exit_reason"], "agent_close")

        *_, blocked = env.step(PortfolioAction.LONG_1X)
        self.assertFalse(blocked["opened_lot"])
        self.assertTrue(blocked["action_blocked"])
        self.assertEqual(blocked["blocked_action_count"], 1)
        self.assertFalse(env.lots)

    def test_blocked_open_does_not_prevent_or_count_time_exit(self) -> None:
        config = RLPortfolioConfig(
            max_holding_bars=1,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(market([100.0] * 10, frequency="30s"), config=config)
        env.reset(episode_max_leverage=5.0)

        for _ in range(5):
            *_, info = env.step(PortfolioAction.LONG_1X)
            self.assertTrue(info["opened_lot"])

        self.assertEqual(len(env.trades), 4)
        *_, blocked = env.step(PortfolioAction.LONG_1X)
        self.assertFalse(blocked["opened_lot"])
        self.assertTrue(blocked["action_blocked"])
        self.assertEqual(blocked["blocked_action_count"], 1)
        self.assertFalse(env.lots)
        self.assertEqual(len(env.trades), 5)
        self.assertTrue(all(trade["exit_reason"] == "time_exit" for trade in env.trades))

    def test_liquidation_precedes_open_rate_limit_and_does_not_count_as_blocked(self) -> None:
        frame = market(
            [100.0] * 8,
            frequency="30s",
            mark_open=[100.0] * 6 + [50.0, 100.0],
            mark_low=[100.0] * 6 + [50.0, 100.0],
            mark_high=[100.0] * 8,
            mark_close=[100.0] * 6 + [50.0, 100.0],
        )
        config = RLPortfolioConfig(
            base_allocation_fraction=0.1,
            max_holding_bars=100,
            max_drawdown=0.99,
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
        )
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=5.0)

        for _ in range(5):
            *_, info = env.step(PortfolioAction.LONG_5X)
            self.assertTrue(info["opened_lot"])

        *_, terminated, _, info = env.step(PortfolioAction.LONG_1X)
        self.assertTrue(terminated)
        self.assertTrue(info["liquidated"])
        self.assertFalse(info["action_blocked"])
        self.assertEqual(info["blocked_action_count"], 0)
        self.assertFalse(info["opened_lot"])
        self.assertFalse(env.lots)
        self.assertEqual(len(env.trades), 5)
        self.assertTrue(all(trade["exit_reason"] == "liquidation" for trade in env.trades))

    def test_action_at_close_executes_at_next_open_and_lots_overlap(self) -> None:
        frame = market([100.0, 101.0, 102.0, 103.0])
        env = RLPortfolioEnv(frame)
        env.reset(episode_max_leverage=3.0)
        env.step(PortfolioAction.LONG_1X)
        self.assertEqual(env.lots[0].entry_index, 1)
        self.assertEqual(env.lots[0].entry_price, 101.0)
        env.step(PortfolioAction.LONG_1X)
        self.assertEqual(len(env.lots), 2)
        self.assertEqual([lot.entry_index for lot in env.lots], [1, 2])

    def test_opposite_action_closes_existing_lots_before_reversal(self) -> None:
        env = RLPortfolioEnv(market([100.0, 101.0, 102.0, 103.0]))
        env.reset(episode_max_leverage=3.0)
        env.step(PortfolioAction.LONG_1X)
        env.step(PortfolioAction.SHORT_1X)
        self.assertEqual(len(env.lots), 1)
        self.assertEqual(env.lots[0].side, -1)
        self.assertEqual(env.trades[0]["exit_reason"], "agent_reversal")

    def test_missing_feature_has_separate_mask(self) -> None:
        frame = market([100.0, 100.0, 100.0])
        frame["macro"] = [np.nan, 0.0, 1.0]
        env = RLPortfolioEnv(frame, feature_names=("macro",))
        observation, _ = env.reset(episode_max_leverage=3.0)
        self.assertEqual(tuple(env.observation_names[:2]), ("macro", "macro__missing"))
        np.testing.assert_array_equal(observation[:2], [0.0, 1.0])
        observation, *_ = env.step(PortfolioAction.HOLD)
        np.testing.assert_array_equal(observation[:2], [0.0, 0.0])

    def test_funding_precedes_new_lot_so_only_existing_lot_is_charged(self) -> None:
        frame = market([100.0, 100.0, 100.0, 100.0], funding_rate=[np.nan, np.nan, 0.01, np.nan])
        config = RLPortfolioConfig(fee_rate=0.0, slippage_bps=0.0, default_spread_bps=0.0)
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=3.0)
        env.step(PortfolioAction.LONG_1X)
        env.step(PortfolioAction.LONG_1X)
        self.assertEqual(len(env.lots), 2)
        self.assertGreater(env.lots[0].funding_cost, 0.0)
        self.assertEqual(env.lots[1].funding_cost, 0.0)

    def test_mark_low_can_liquidate_when_trade_low_does_not(self) -> None:
        frame = market(
            [100.0, 100.0, 100.0],
            mark_open=[100.0, 100.0, 100.0],
            mark_low=[100.0, 70.0, 100.0],
            mark_high=[100.0, 100.0, 100.0],
            mark_close=[100.0, 70.0, 100.0],
        )
        config = RLPortfolioConfig(base_allocation_fraction=1.0, fee_rate=0.0, slippage_bps=0.0, default_spread_bps=0.0)
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=5.0)
        _, _, terminated, _, info = env.step(PortfolioAction.LONG_5X)
        self.assertTrue(terminated)
        self.assertTrue(info["liquidated"])
        self.assertEqual(env.trades[0]["exit_reason"], "liquidation")
        self.assertFalse(info["mark_price_fallback_used"])

    def test_time_exit_occurs_before_exit_bar_intrabar_path(self) -> None:
        frame = market(
            [100.0, 100.0, 100.0, 100.0],
            mark_open=[100.0] * 4,
            mark_low=[100.0, 100.0, 50.0, 100.0],
            mark_high=[100.0] * 4,
            mark_close=[100.0] * 4,
        )
        config = RLPortfolioConfig(max_holding_bars=1, base_allocation_fraction=1.0, fee_rate=0.0, slippage_bps=0.0, default_spread_bps=0.0)
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=5.0)
        env.step(PortfolioAction.LONG_5X)
        env.step(PortfolioAction.HOLD)
        self.assertEqual(env.trades[0]["exit_reason"], "time_exit")

    def test_entry_cost_cannot_push_gross_leverage_above_episode_cap(self) -> None:
        config = RLPortfolioConfig(base_allocation_fraction=1.0)
        env = RLPortfolioEnv(market([100.0, 100.0, 100.0]), config=config)
        env.reset(episode_max_leverage=5.0)
        _, _, _, _, info = env.step(PortfolioAction.LONG_5X)
        self.assertLessEqual(info["gross_leverage"], 5.0 + 1e-12)

    def test_drawdown_termination_realizes_open_lots_and_exit_cost(self) -> None:
        frame = market([100.0, 100.0, 95.0, 95.0])
        config = RLPortfolioConfig(
            base_allocation_fraction=1.0,
            max_drawdown=0.01,
            maintenance_margin_rate=0.0,
        )
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=3.0)
        _, _, terminated, _, _ = env.step(PortfolioAction.LONG_3X)
        self.assertFalse(terminated)
        _, _, terminated, _, _ = env.step(PortfolioAction.HOLD)
        self.assertTrue(terminated)
        self.assertFalse(env.lots)
        self.assertEqual(env.trades[-1]["exit_reason"], "risk_termination")

    def test_recovery_multiplier_is_capped_after_losses(self) -> None:
        frame = market([100.0, 100.0, 90.0, 90.0, 80.0, 80.0, 70.0])
        config = RLPortfolioConfig(
            fee_rate=0.0,
            slippage_bps=0.0,
            default_spread_bps=0.0,
            max_drawdown=0.99,
        )
        env = RLPortfolioEnv(frame, config=config)
        env.reset(episode_max_leverage=3.0)
        env.step(PortfolioAction.LONG_1X)
        env.step(PortfolioAction.CLOSE_ALL)
        self.assertEqual(env.recovery_layer, 1)
        self.assertEqual(env._recovery_multiplier(), 1.25)
        env.step(PortfolioAction.LONG_1X)
        env.step(PortfolioAction.CLOSE_ALL)
        self.assertEqual(env.recovery_layer, 2)
        self.assertEqual(env._recovery_multiplier(), 1.5)


if __name__ == "__main__":
    unittest.main()
