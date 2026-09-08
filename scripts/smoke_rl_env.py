"""Run a deterministic random-policy smoke test on the RL portfolio environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btc_perp.rl_env import RLPortfolioEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_1m_features.csv.gz"))
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--episode-max-leverage", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = ("return_1", "atr_pct", "realized_vol", "volume_z")
    columns = ["timestamp", "open", "high", "low", "close", *features]
    frame = pd.read_csv(args.input, compression="gzip", usecols=columns, nrows=args.rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    environment = RLPortfolioEnv(frame, feature_names=features, seed=args.seed)
    observation, info = environment.reset(
        start_index=args.start_index,
        max_steps=args.steps,
        episode_max_leverage=args.episode_max_leverage,
    )
    rng = np.random.default_rng(args.seed)
    terminated = False
    truncated = False
    reward_sum = 0.0
    executed_steps = 0
    while not (terminated or truncated):
        valid_actions = np.flatnonzero(environment.valid_action_mask())
        action = int(rng.choice(valid_actions))
        observation, reward, terminated, truncated, info = environment.step(action)
        reward_sum += reward
        executed_steps += 1
    result = {
        "seed": args.seed,
        "requested_steps": args.steps,
        "executed_steps": executed_steps,
        "terminated": terminated,
        "truncated": truncated,
        "equity": info["equity"],
        "trade_count": len(environment.trades),
        "open_lots": info["lot_count"],
        "drawdown": info["drawdown"],
        "liquidated": info["liquidated"],
        "mark_price_fallback_used": info["mark_price_fallback_used"],
        "observation_shape": list(observation.shape),
        "reward_sum": reward_sum,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
