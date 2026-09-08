"""Gymnasium adapter for :mod:`btc_perp.rl_env`."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd

from .rl_env import PortfolioAction, RLPortfolioConfig, RLPortfolioEnv


class GymRLPortfolioEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: pd.DataFrame,
        *,
        feature_names: Sequence[str],
        config: RLPortfolioConfig | None = None,
        episode_steps: int = 2_048,
        random_start: bool = True,
        fixed_start_index: int = 0,
        fixed_max_leverage: float | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if episode_steps <= 0 or episode_steps >= len(market_data):
            raise ValueError("episode_steps must be positive and smaller than market_data")
        self.core = RLPortfolioEnv(
            market_data,
            feature_names=feature_names,
            config=config,
            seed=seed,
        )
        self.episode_steps = episode_steps
        self.random_start = random_start
        self.fixed_start_index = fixed_start_index
        self.fixed_max_leverage = fixed_max_leverage
        self.action_space = gym.spaces.Discrete(len(PortfolioAction))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.core.observation_names),),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        options = options or {}
        max_start = len(self.core.frame) - self.episode_steps - 1
        if max_start < 0:
            raise RuntimeError("market_data no longer fits episode_steps")
        if "start_index" in options:
            start_index = int(options["start_index"])
        elif self.random_start:
            start_index = int(self.np_random.integers(0, max_start + 1))
        else:
            start_index = self.fixed_start_index
        if not 0 <= start_index <= max_start:
            raise ValueError("episode start does not leave enough rows")
        leverage = options.get("episode_max_leverage", self.fixed_max_leverage)
        if leverage is None:
            leverage = float(self.np_random.choice(self.core.config.episode_leverage_choices))
        return self.core.reset(
            start_index=start_index,
            max_steps=self.episode_steps,
            episode_max_leverage=float(leverage),
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return self.core.step(action)

    def action_masks(self) -> np.ndarray:
        """Return True for actions currently valid for MaskablePPO."""

        return self.core.valid_action_mask().copy()
