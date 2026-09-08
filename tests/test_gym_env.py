import importlib.util
import unittest

import pandas as pd


RL_INSTALLED = importlib.util.find_spec("gymnasium") is not None and importlib.util.find_spec("stable_baselines3") is not None


@unittest.skipUnless(RL_INSTALLED, "RL optional dependencies are not installed")
class GymRLPortfolioEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        from btc_perp.gym_env import GymRLPortfolioEnv

        self.environment_class = GymRLPortfolioEnv
        prices = [100.0 + index * 0.01 for index in range(100)]
        self.frame = pd.DataFrame(
            {
                "open": prices,
                "high": [price * 1.001 for price in prices],
                "low": [price * 0.999 for price in prices],
                "close": prices,
                "signal": [0.0] * len(prices),
                "long_expected_net_return": [0.001] * len(prices),
                "short_expected_net_return": [0.0] * len(prices),
            }
        )

    def test_environment_passes_stable_baselines_checker(self) -> None:
        from stable_baselines3.common.env_checker import check_env

        environment = self.environment_class(
            self.frame,
            feature_names=("signal",),
            episode_steps=20,
            fixed_max_leverage=3.0,
        )
        check_env(environment, warn=True)

    def test_fixed_validation_reset_is_reproducible(self) -> None:
        environment = self.environment_class(
            self.frame,
            feature_names=("signal",),
            episode_steps=20,
            random_start=False,
            fixed_start_index=10,
            fixed_max_leverage=3.0,
        )
        first, first_info = environment.reset(seed=7)
        second, second_info = environment.reset(seed=99)
        self.assertEqual(first_info["cursor"], 10)
        self.assertEqual(second_info["cursor"], 10)
        self.assertEqual(first.tolist(), second.tolist())

    def test_action_masks_match_core_before_and_after_step(self) -> None:
        from btc_perp.rl_env import PortfolioAction

        environment = self.environment_class(
            self.frame,
            feature_names=("signal",),
            episode_steps=20,
            random_start=False,
            fixed_start_index=10,
            fixed_max_leverage=3.0,
        )
        environment.reset()

        self.assertEqual(environment.action_masks().dtype, bool)
        self.assertEqual(environment.action_masks().shape, (environment.action_space.n,))
        self.assertEqual(environment.action_masks().tolist(), environment.core.valid_action_mask().tolist())

        environment.step(int(PortfolioAction.LONG_1X))
        self.assertEqual(environment.action_masks().tolist(), environment.core.valid_action_mask().tolist())


if __name__ == "__main__":
    unittest.main()
